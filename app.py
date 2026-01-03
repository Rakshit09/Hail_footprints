import os
import uuid
import json
import threading
import ssl
from pathlib import Path
from datetime import datetime
import io
import base64
import re
import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, jsonify, send_from_directory, url_for, session, send_file)
import warnings
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from pyproj import Transformer
from config import Config
from processing.footprint import Params, run_footprint, get_processing_status
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    # disable SSL verification 
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # disable SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # unverified SSL context for urllib
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("[SSL] Certificate verification disabled for tile fetching")
except Exception as e:
    print(f"[SSL] Could not disable SSL verification: {e}")

app = Flask(__name__)
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
socketio = SocketIO(
    app,
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    cors_allowed_origins="*",
    max_http_buffer_size=1024 * 1024 * 1024,  # 1GB
    async_handlers=True)


app.secret_key = 'supersecretkey'
# store processing jobs
processing_jobs = {}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def guess_column_type(col_name: str) -> str:
    """guess column type based on strict name patterns."""
    
    col_lower = col_name.lower().strip()
    
    # LATITUDE 
    if col_lower in ['lat', 'latitude', 'y', 'y_coord', 'lat_dd', 'slat', 'start_lat']:
        return 'latitude'
        
    if re.search(r'(^|[\s_])(lat|latitude)([\s_]|$)', col_lower):
        return 'latitude'
    
    # LONGITUDE 
    if col_lower in ['lon', 'long', 'lng', 'longitude', 'x', 'x_coord', 'lon_dd', 'slon', 'start_lon']:
        return 'longitude'
        
    if re.search(r'(^|[\s_])(lon|lng|long|longitude)([\s_]|$)', col_lower):
        return 'longitude'
    
    # HAIL SIZE 
    if col_lower in ['max_hail_diameter','max_size', 'hail_size', 'hailsize', 'maximum_hail_size', 'hail', 'size', 'diameter', 'diam', 'mag', 'magnitude', 'hail_size']:
        return 'hail_size'
    
    
    return 'unknown'


def sanitize_for_json(obj):
    """
    recursively sanitize an object for JSON serialization
    handles NaN, Inf, numpy types, etc
    """
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


@app.route('/')
def index():
    """main page with upload form"""
    max_upload_bytes = getattr(Config, 'MAX_CONTENT_LENGTH', 50 * 1024 * 1024)
    max_upload_mb = int(max_upload_bytes / (1024 * 1024))
    allowed_ext = sorted(list(getattr(Config, 'ALLOWED_EXTENSIONS', {'csv'})))
    return render_template(
        'index.html',
        max_upload_bytes=max_upload_bytes,
        max_upload_mb=max_upload_mb,
        allowed_ext=allowed_ext
    )


@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_file():
    """handle file upload and return column info"""
    try:
        # Check content length first
        content_length = request.content_length
        max_size = app.config.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024)
        
        if content_length and content_length > max_size:
            return jsonify({
                'error': f'File too large. Maximum size is {max_size // (1024*1024)}MB'
            }), 413
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            allowed = ', '.join(Config.ALLOWED_EXTENSIONS)
            return jsonify({'error': f'File type not allowed. Supported: {allowed}'}), 400
        
        # Generate unique ID for this session
        job_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        
        # Handle empty filename after sanitization
        if not filename:
            filename = f"upload_{job_id}.csv"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{job_id}_{timestamp}_{filename}"
        
        # Ensure upload folder exists
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        
        filepath = Config.UPLOAD_FOLDER / saved_filename
        
        # Save file
        try:
            file.save(str(filepath))
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        # Verify file was saved
        if not filepath.exists():
            return jsonify({'error': 'File was not saved correctly'}), 500
        
        file_size = filepath.stat().st_size
        if file_size == 0:
            filepath.unlink()  # Delete empty file
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        print(f"[Upload] Saved file: {saved_filename} ({file_size} bytes)")
        
        # Parse columns
        try:
            if filename.lower().endswith('.csv'):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df_preview = None
                
                for encoding in encodings:
                    try:
                        df_preview = pd.read_csv(filepath, nrows=25, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"[Upload] CSV parse error with {encoding}: {e}")
                        continue
                
                if df_preview is None:
                    return jsonify({'error': 'Could not parse CSV file. Check file encoding.'}), 400
                
                # Count total rows efficiently
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                except:
                    row_count = len(pd.read_csv(filepath, usecols=[0]))
                    
            else:
                import geopandas as gpd
                gdf = gpd.read_file(str(filepath))
                df_preview = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore')).head(25)
                row_count = len(gdf)
            
            columns = list(df_preview.columns)
            
            if len(columns) == 0:
                return jsonify({'error': 'No columns found in file'}), 400
            
            # Auto-detect with scoring/prioritization
            best_matches = {}
            
            for col in columns:
                col_type = guess_column_type(col)
                
                if col_type != 'unknown':
                    col_lower = col.lower()
                    score = 3  # Default score (worst)
                    
                    # Scoring rules
                    if col_type == 'latitude':
                        if col_lower in ['latitude', 'lat', 'y', 'lat_dd']:
                            score = 1
                        elif 'start' in col_lower or col_lower == 'slat':
                            score = 2
                            
                    elif col_type == 'longitude':
                        if col_lower in ['longitude', 'lon', 'long', 'x', 'lon_dd', 'lng']:
                            score = 1
                        elif 'start' in col_lower or col_lower == 'slon':
                            score = 2
                    
                    elif col_type == 'hail_size':
                        if col_lower in ['max_hail_diameter', 'hail_size', 'hailsize', 'maximum_hail_diameter']:
                            score = 1
                        elif 'max' in col_lower or 'diam' in col_lower:
                            score = 2

                    if col_type not in best_matches:
                        best_matches[col_type] = {'col': col, 'score': score}
                    else:
                        if score < best_matches[col_type]['score']:
                            best_matches[col_type] = {'col': col, 'score': score}
            
            column_suggestions = {k: v['col'] for k, v in best_matches.items()}
            
            # Convert to records and sanitize
            sample_data = df_preview.to_dict('records')
            sample_data = sanitize_for_json(sample_data)
            
            response_data = {
                'success': True,
                'job_id': job_id,
                'filename': saved_filename,
                'columns': columns,
                'suggestions': column_suggestions,
                'sample_data': sample_data,
                'row_count': row_count
            }
            
            print(f"[Upload] Success: {len(columns)} columns, {row_count} rows")
            
            return jsonify(response_data)
            
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'File is empty or has no data'}), 400
        except pd.errors.ParserError as e:
            return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


# Add error handlers for common HTTP errors
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum upload size exceeded.'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request error"""
    return jsonify({'error': 'Bad request. Please check your input.'}), 400


@app.route('/process', methods=['POST'])
def process_footprint():
    """start footprint processing"""
    data = request.json
    
    required_fields = ['filename', 'lon_col', 'lat_col', 'hail_col', 'event_name']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    job_id = data.get('job_id', str(uuid.uuid4())[:8])
    
    # create job-specific output folder
    output_folder = Config.OUTPUT_FOLDER / job_id
    output_folder.mkdir(exist_ok=True)
    
    # build params
    params = Params(
        input_points=str(Config.UPLOAD_FOLDER / data['filename']),
        hail_field=data['hail_col'],
        event_name=data['event_name'],
        lon_col=data['lon_col'],
        lat_col=data['lat_col'],
        grouping_threshold_km=float(data.get('grouping_threshold_km', 30.0)),
        large_buffer_km=float(data.get('large_buffer_km', 10.0)),
        small_buffer_km=float(data.get('small_buffer_km', 5.0)),
        out_folder=str(output_folder),
        job_id=job_id,
    )
    
    # store job info
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Initializing...',
        'params': params,
        'result': None,
        'error': None
    }
    
    # start processing in background thread
    thread = threading.Thread(
        target=run_processing_job,
        args=(job_id, params)
    )
    thread.start()
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Processing started'
    })


def run_processing_job(job_id: str, params: Params):
    """run processing in background thread"""
    def progress_callback(progress: int, message: str):
        processing_jobs[job_id]['progress'] = progress
        processing_jobs[job_id]['message'] = message
        socketio.emit('progress', {
            'job_id': job_id,
            'progress': progress,
            'message': message
        })
    
    try:
        processing_jobs[job_id]['status'] = 'processing'
        result = run_footprint(params, progress_callback=progress_callback)
        
        # sanitize result for JSON
        result = sanitize_for_json(result)
        
        processing_jobs[job_id]['status'] = 'completed'
        processing_jobs[job_id]['progress'] = 100
        processing_jobs[job_id]['result'] = result
        
        socketio.emit('completed', {
            'job_id': job_id,
            'result': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['error'] = str(e)
        
        socketio.emit('error', {
            'job_id': job_id,
            'error': str(e)
        })


@app.route('/status/<job_id>')
def get_status(job_id):
    """get processing status"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    return jsonify(sanitize_for_json({
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'result': job['result'],
        'error': job['error']
    }))

@app.route('/health')
def health_check():
    """Health check endpoint for Posit Connect"""
    return jsonify({
        'status': 'ok',
        'upload_folder_exists': Config.UPLOAD_FOLDER.exists(),
        'output_folder_exists': Config.OUTPUT_FOLDER.exists(),
        'upload_folder_writable': os.access(Config.UPLOAD_FOLDER, os.W_OK) if Config.UPLOAD_FOLDER.exists() else False,
    })
@app.route('/viewer/<job_id>')
def viewer(job_id):
    """interactive map viewer"""
    if job_id not in processing_jobs:
        return render_template('error.html', message='Job not found'), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed':
        return render_template('error.html', message='Processing not complete'), 400
    
    return render_template('viewer.html', 
                          job_id=job_id,
                          result=job['result'],
                          event_name=job['params'].event_name)


@app.route('/outputs/<job_id>/<filename>')
def serve_output(job_id, filename):
    """serve output files"""
    output_folder = Config.OUTPUT_FOLDER / job_id
    return send_from_directory(output_folder, filename)


@app.route('/geojson/<job_id>')
def get_geojson(job_id):
    """get GeoJSON data for the map"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'No data available'}), 400
    
    # get filename from result
    geojson_filename = job['result'].get('geojson', '')
    if not geojson_filename:
        return jsonify({'error': 'GeoJSON not found'}), 404
    
    # build full path using job's output folder
    geojson_path = Config.OUTPUT_FOLDER / job_id / geojson_filename
    
    if not geojson_path.exists():
        return jsonify({'error': f'GeoJSON file not found: {geojson_filename}'}), 404
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(sanitize_for_json(data))


@app.route('/footprint_geojson/<job_id>')
def get_footprint_geojson(job_id):
    """get dissolved footprint GeoJSON"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'No data available'}), 400
    
    geojson_filename = job['result'].get('footprint_geojson', '')
    if not geojson_filename:
        return jsonify({'error': 'Footprint GeoJSON not found'}), 404
    
    geojson_path = Config.OUTPUT_FOLDER / job_id / geojson_filename
    
    if not geojson_path.exists():
        return jsonify({'error': f'Footprint GeoJSON file not found: {geojson_filename}'}), 404
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(sanitize_for_json(data))


@app.route('/points_geojson/<job_id>')
def get_points_geojson(job_id):
    """get input points as GeoJSON"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'No data available'}), 400
    
    geojson_filename = job['result'].get('points_geojson', '')
    if not geojson_filename:
        return jsonify({'error': 'Points GeoJSON not found'}), 404
    
    geojson_path = Config.OUTPUT_FOLDER / job_id / geojson_filename
    
    if not geojson_path.exists():
        return jsonify({'error': f'Points GeoJSON file not found: {geojson_filename}'}), 404
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(sanitize_for_json(data))


# basemap configurations
BASEMAPS = {
    'osm': {
        'name': 'OpenStreetMap',
        'url': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap contributors'
    },
    'carto_light': {
        'name': 'Carto Light (Clean)',
        'url': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        'attribution': '© CARTO'
    },
    'carto_dark': {
        'name': 'Carto Dark',
        'url': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        'attribution': '© CARTO'
    },
    'carto_voyager': {
        'name': 'Carto Voyager',
        'url': 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
        'attribution': '© CARTO'
    },
    'carto_positron_nolabels': {
        'name': 'Carto Light (No Labels)',
        'url': 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
        'attribution': '© CARTO'
    },
    'stamen_toner': {
        'name': 'Stamen Toner (B&W)',
        'url': 'https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        'attribution': '© Stamen Design'
    },
    'stamen_toner_lite': {
        'name': 'Stamen Toner Lite',
        'url': 'https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png',
        'attribution': '© Stamen Design'
    },
    'esri_worldstreet': {
        'name': 'ESRI World Street',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'esri_worldtopo': {
        'name': 'ESRI World Topo',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'esri_worldimagery': {
        'name': 'ESRI Satellite',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'esri_gray': {
        'name': 'ESRI Light Gray',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'esri_darkgray': {
        'name': 'ESRI Dark Gray',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'esri_natgeo': {
        'name': 'ESRI National Geographic',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri, National Geographic'
    },
    'esri_boundaries': {
        'name': 'ESRI Boundaries Only',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
        'attribution': '© Esri'
    },
    'none': {
        'name': 'No Basemap (White)',
        'url': '',
        'attribution': ''
    }
}


@app.route('/basemaps')
def get_basemaps():
    """return available basemap options"""
    return jsonify(BASEMAPS)


# error template
@app.route('/error')
def error_page():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

@app.route('/debug/<job_id>')
def debug_job(job_id):
    """debug endpoint to check job status and files"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    output_folder = Config.OUTPUT_FOLDER / job_id
    
    # list files in output folder
    files = []
    if output_folder.exists():
        files = [f.name for f in output_folder.iterdir()]
    
    # check GeoJSON validity
    geojson_check = {}
    if job['result']:
        for key in ['geojson', 'footprint_geojson', 'points_geojson']:
            filename = job['result'].get(key, '')
            if filename:
                filepath = output_folder / filename
                if filepath.exists():
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            geojson_check[key] = {
                                'exists': True,
                                'features': len(data.get('features', [])),
                                'crs': data.get('crs', {}).get('properties', {}).get('name', 'unknown')
                            }
                            # check first feature coordinates
                            if data.get('features'):
                                first_geom = data['features'][0].get('geometry', {})
                                coords = first_geom.get('coordinates', [])
                                if coords:
                                    # get sample coordinate
                                    sample = coords
                                    while isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], list):
                                        sample = sample[0]
                                    geojson_check[key]['sample_coord'] = sample[:2] if len(sample) >= 2 else sample
                    except Exception as e:
                        geojson_check[key] = {'exists': True, 'error': str(e)}
                else:
                    geojson_check[key] = {'exists': False}
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'result': sanitize_for_json(job['result']),
        'files_in_folder': files,
        'geojson_check': geojson_check
    })

@app.route('/render_map/<job_id>', methods=['POST'])
def render_map(job_id):
    """render a map with user-specified settings"""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'Job not complete'}), 400
    
    data = request.json or {}
    
    # get parameters from request
    bounds = data.get('bounds')  # [minLon, minLat, maxLon, maxLat]
    basemap_id = data.get('basemap', 'carto_light')
    show_footprint = data.get('show_footprint', True)
    show_outline = data.get('show_outline', True)
    show_points = data.get('show_points', False)
    opacity = float(data.get('opacity', 0.6))
    # optional export sizing (pixels). if omitted, we pick a high-res default.
    width_px = int(data.get('width_px', 3200))
    height_px = int(data.get('height_px', 2000))
    outline_width = data.get('outline_width', 2.5)
    outline_color = data.get('outline_color', '#000000')
    # NEW: display mode and color map from viewer
    display_mode = data.get('display_mode', 'cells')  # 'cells' or 'continuous'
    color_map = data.get('color_map', 'ylOrRd')
    
    output_folder = Config.OUTPUT_FOLDER / job_id
    result = job['result']
    
    try:
        png_bytes = generate_map_png(
            output_folder=output_folder,
            geojson_file=result.get('geojson', ''),
            footprint_file=result.get('footprint_geojson', ''),
            points_file=result.get('points_geojson', ''),
            bounds=bounds,
            basemap_id=basemap_id,
            show_footprint=show_footprint,
            show_outline=show_outline,
            show_points=show_points,
            opacity=opacity,
            event_name=job['params'].event_name,
            hail_min=result.get('hail_min', 0),
            hail_max=result.get('hail_max', 1),
            width_px=width_px,
            height_px=height_px,
            outline_width=outline_width,
            outline_color=outline_color,
            display_mode=display_mode,
            color_map=color_map,
        )
        
        return send_file(
            io.BytesIO(png_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{job['params'].event_name}_custom_map.png"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def fetch_tiles_manually(ax, ext_x1, ext_y1, ext_x2, ext_y2, zoom, tile_url_template):
    """
    manually fetch map tiles and compose them into a basemap.
    This bypasses contextily and uses requests directly with SSL disabled.
    all tile servers used are free and open source:
    - openstreetmap: open database license (ODbL)
    - carto: free for most uses
    - esri: free for development/non-commercial
    """
    import requests
    import math
    from PIL import Image
    from io import BytesIO
    import numpy as np
    
    print(f"[ManualTiles] Fetching tiles from: {tile_url_template[:50]}...")
    
    def deg2num(lat_deg, lon_deg, zoom):
        """convert lat/lon to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def num2deg(xtile, ytile, zoom):
        """convert tile numbers to lat/lon of the NW corner"""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def latlon_to_mercator(lat, lon):
        """convert lat/lon to Web Mercator coordinates"""
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * 20037508.34 / 180.0
        return x, y
    
    # convert mercator extent back to lat/lon
    def mercator_to_latlon(x, y):
        lon = x * 180.0 / 20037508.34
        lat = math.atan(math.exp(y * math.pi / 20037508.34)) * 360.0 / math.pi - 90
        return lat, lon
    
    # get corner coordinates in lat/lon
    lat_min, lon_min = mercator_to_latlon(ext_x1, ext_y1)
    lat_max, lon_max = mercator_to_latlon(ext_x2, ext_y2)
    
    # get tile range
    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)
    
    # ensure correct order
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    # limit number of tiles to prevent huge downloads
    max_tiles = 100
    num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    if num_tiles > max_tiles:
        print(f"[ManualTiles] Too many tiles ({num_tiles}), reducing zoom...")
        return False
    
    print(f"[ManualTiles] Fetching {num_tiles} tiles (x: {x_min}-{x_max}, y: {y_min}-{y_max})")
    
    # create session with SSL disabled
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'HailFootprintApp/1.0 (Python/requests)',
        'Accept': 'image/png,image/*',
    })
    
    # fetch all tiles
    tile_size = 256
    tiles = {}
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # format URL - handle both {z}/{x}/{y} and {z}/{y}/{x} patterns
            url = tile_url_template.format(z=zoom, x=x, y=y)
            try:
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    tiles[(x, y)] = img
                else:
                    print(f"[ManualTiles] Tile {x},{y} returned {response.status_code}")
            except Exception as e:
                print(f"[ManualTiles] Failed to fetch tile {x},{y}: {str(e)[:50]}")
    
    if not tiles:
        print("[ManualTiles] No tiles fetched successfully")
        return False
    
    print(f"[ManualTiles] Successfully fetched {len(tiles)} tiles")
    
    # compose tiles into single image
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    basemap_img = Image.new('RGB', (width, height), (245, 245, 245))
    
    for (x, y), tile_img in tiles.items():
        px = (x - x_min) * tile_size
        py = (y - y_min) * tile_size
        if tile_img.mode != 'RGB':
            tile_img = tile_img.convert('RGB')
        basemap_img.paste(tile_img, (px, py))
    
    # calculate extent of the composed image in web mercator
    nw_lat, nw_lon = num2deg(x_min, y_min, zoom)
    se_lat, se_lon = num2deg(x_max + 1, y_max + 1, zoom)
    
    nw_x, nw_y = latlon_to_mercator(nw_lat, nw_lon)
    se_x, se_y = latlon_to_mercator(se_lat, se_lon)
    
    # display the basemap
    img_array = np.array(basemap_img)
    ax.imshow(img_array, extent=[nw_x, se_x, se_y, nw_y], zorder=1, aspect='auto')
    
    print(f"[ManualTiles] Basemap composed and added to plot")
    return True


def generate_map_png(
    output_folder,
    geojson_file,
    footprint_file,
    points_file,
    bounds,
    basemap_id,
    show_footprint,
    show_outline,
    show_points,
    opacity,
    event_name,
    hail_min,
    hail_max,
    width_px=3200,
    height_px=2000,
    outline_width=2.5,
    outline_color='#000000',
    display_mode='cells',
    color_map='ylOrRd',
):
    """generate a PNG map with specified settings"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import geopandas as gpd
    from pyproj import Transformer
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        import contextily as ctx
        HAS_CTX = True
        print(f"[MapGen] Contextily version: {ctx.__version__}")
    except ImportError:
        HAS_CTX = False
        print("[MapGen] Contextily not available!")
    
    print(f"[MapGen] ========================================")
    print(f"[MapGen] Basemap requested: '{basemap_id}'")
    print(f"[MapGen] Bounds: {bounds}")
    print(f"[MapGen] Show footprint: {show_footprint}, outline: {show_outline}, points: {show_points}")
    print(f"[MapGen] Opacity: {opacity}")
    print(f"[MapGen] Target export size: {width_px}x{height_px}px")
    
    # load GeoJSON files
    footprint_gdf = None
    outline_gdf = None
    points_gdf = None
    
    if show_footprint and geojson_file:
        geojson_path = output_folder / geojson_file
        if geojson_path.exists():
            footprint_gdf = gpd.read_file(geojson_path)
            print(f"[MapGen] Loaded footprint: {len(footprint_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Footprint file not found: {geojson_path}")
    
    if show_outline and footprint_file:
        outline_path = output_folder / footprint_file
        if outline_path.exists():
            outline_gdf = gpd.read_file(outline_path)
            print(f"[MapGen] Loaded outline: {len(outline_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Outline file not found: {outline_path}")
    
    if show_points and points_file:
        points_path = output_folder / points_file
        if points_path.exists():
            points_gdf = gpd.read_file(points_path)
            print(f"[MapGen] Loaded points: {len(points_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Points file not found: {points_path}")
    
    # validate we have at least some data to render
    has_data = any([
        footprint_gdf is not None and not footprint_gdf.empty,
        outline_gdf is not None and not outline_gdf.empty,
        points_gdf is not None and not points_gdf.empty
    ])
    
    if not has_data and (not bounds or len(bounds) != 4):
        raise ValueError("No data to render and no bounds provided")
    
    # determine extent in WGS84
    bounds_from_viewer = bounds and len(bounds) == 4 and all(b is not None for b in bounds)
    
    if bounds_from_viewer:
        extent_minx, extent_miny, extent_maxx, extent_maxy = bounds
    elif footprint_gdf is not None and not footprint_gdf.empty:
        extent_minx, extent_miny, extent_maxx, extent_maxy = footprint_gdf.total_bounds
    elif outline_gdf is not None and not outline_gdf.empty:
        extent_minx, extent_miny, extent_maxx, extent_maxy = outline_gdf.total_bounds
    else:
        raise ValueError("No bounds provided and no footprint data available")
    
    # validate bounds
    if extent_minx >= extent_maxx or extent_miny >= extent_maxy:
        raise ValueError(f"Invalid bounds: [{extent_minx}, {extent_miny}, {extent_maxx}, {extent_maxy}]")
    
    # Only add padding if bounds came from footprint (not from viewer)
    # Viewer bounds should match exactly what the user sees
    if not bounds_from_viewer:
        pad_x = (extent_maxx - extent_minx) * 0.05
        pad_y = (extent_maxy - extent_miny) * 0.05
        extent_minx -= pad_x
        extent_maxx += pad_x
        extent_miny -= pad_y
        extent_maxy += pad_y
    
    print(f"[MapGen] WGS84 extent: [{extent_minx:.4f}, {extent_miny:.4f}, {extent_maxx:.4f}, {extent_maxy:.4f}]")
    
    # convert extent to web mercator (EPSG:3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ext_x1, ext_y1 = transformer.transform(extent_minx, extent_miny)
    ext_x2, ext_y2 = transformer.transform(extent_maxx, extent_maxy)
    
    print(f"[MapGen] 3857 extent: [{ext_x1:.0f}, {ext_y1:.0f}, {ext_x2:.0f}, {ext_y2:.0f}]")
    
    # reproject data to 3857
    footprint_3857 = None
    outline_3857 = None
    points_3857 = None
    
    if footprint_gdf is not None and not footprint_gdf.empty:
        footprint_3857 = footprint_gdf.to_crs(epsg=3857)
    
    if outline_gdf is not None and not outline_gdf.empty:
        outline_3857 = outline_gdf.to_crs(epsg=3857)
    
    if points_gdf is not None and not points_gdf.empty:
        points_3857 = points_gdf.to_crs(epsg=3857)
    
    # create figure
    dpi = 150  # Reduced DPI for faster generation, still high quality
    fig_w_in = max(6, float(width_px) / dpi)
    fig_h_in = max(4, float(height_px) / dpi)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    
    # Map viewer colormap names to matplotlib colormaps
    COLORMAP_MAPPING = {
        'ylOrRd': 'YlOrRd',
        'viridis': 'viridis',
        'plasma': 'plasma',
        'inferno': 'inferno',
        'turbo': 'turbo',
        'blues': 'Blues',
        'greens': 'Greens',
        'rdYlGn': 'RdYlGn',
        'spectral': 'Spectral',
        'coolwarm': 'coolwarm',
        'hot': 'hot',
        'jet': 'jet',
    }
    
    # Get the matplotlib colormap name
    mpl_cmap_name = COLORMAP_MAPPING.get(color_map, 'YlOrRd')
    cmap = plt.cm.get_cmap(mpl_cmap_name)
    print(f"[MapGen] Using colormap: {mpl_cmap_name} (from viewer: {color_map})")
    print(f"[MapGen] Display mode: {display_mode}")
    
    vmin = hail_min if hail_min is not None else 0
    vmax = hail_max if hail_max is not None else 1
    if vmax <= vmin:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # plot data first, then set extent, then add basemap
    
    # plot footprint
    if show_footprint and footprint_3857 is not None and not footprint_3857.empty:
        if 'Hail_Size' in footprint_3857.columns:
            if display_mode == 'continuous':
                # CONTINUOUS MODE: Rasterize and apply Gaussian blur for smooth appearance
                try:
                    from scipy.ndimage import gaussian_filter
                    import numpy as np
                    from rasterio.features import rasterize
                    from rasterio.transform import from_bounds
                    
                    # Use the map extent for consistent positioning
                    rast_minx, rast_miny, rast_maxx, rast_maxy = ext_x1, ext_y1, ext_x2, ext_y2
                    rast_width = max(400, int(width_px * 0.5))
                    rast_height = max(300, int(height_px * 0.5))
                    
                    print(f"[MapGen] Rasterizing to {rast_width}x{rast_height}")
                    
                    # Create transform for rasterization
                    transform = from_bounds(rast_minx, rast_miny, rast_maxx, rast_maxy, 
                                           rast_width, rast_height)
                    
                    # Rasterize: create an array with hail values
                    shapes = [(geom, val) for geom, val in zip(footprint_3857.geometry, 
                                                                footprint_3857['Hail_Size'])]
                    raster = rasterize(shapes, out_shape=(rast_height, rast_width), 
                                       transform=transform, fill=0, dtype='float32')
                    
                    # Create binary mask from footprint boundary (1 = inside, 0 = outside)
                    boundary_shapes = [(geom, 1) for geom in footprint_3857.geometry]
                    boundary_mask = rasterize(boundary_shapes, out_shape=(rast_height, rast_width),
                                              transform=transform, fill=0, dtype='uint8')
                    
                    print(f"[MapGen] Raster non-zero pixels: {np.count_nonzero(raster)}")
                    
                    # Apply Gaussian blur for smoothing
                    sigma = max(3, min(rast_width, rast_height) // 40)
                    smoothed = gaussian_filter(raster, sigma=sigma)
                    
                    # CLIP: Apply boundary mask to keep colors INSIDE footprint only
                    # Set values outside the original boundary to 0
                    smoothed = np.where(boundary_mask > 0, smoothed, 0)
                    
                    print(f"[MapGen] Smoothed (clipped) non-zero: {np.count_nonzero(smoothed)}")
                    
                    # Mask zero values for transparency
                    masked = np.ma.masked_where(smoothed <= 0.001, smoothed)
                    
                    # Plot as image - rasterio uses origin='upper' (top-left)
                    ax.imshow(masked, extent=[rast_minx, rast_maxx, rast_miny, rast_maxy], 
                             origin='upper', cmap=cmap, norm=norm, alpha=opacity, 
                             zorder=2, interpolation='bilinear', aspect='auto')
                    
                    print(f"[MapGen] Plotted footprint in CONTINUOUS mode (sigma={sigma})")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[MapGen] Continuous mode failed, falling back to cells: {e}")
                    # Fallback to cells mode
                    footprint_3857.plot(
                        ax=ax, column='Hail_Size', cmap=cmap, norm=norm,
                        alpha=opacity, linewidth=0, edgecolor='none', zorder=2,
                    )
            else:
                # CELLS MODE: Plot as grid cells (polygons with thin borders)
                footprint_3857.plot(
                    ax=ax,
                    column='Hail_Size',
                    cmap=cmap,
                    norm=norm,
                    alpha=opacity,
                    linewidth=0.1,
                    edgecolor='#666666',
                    zorder=2,
                )
                print("[MapGen] Plotted footprint in CELLS mode")
        else:
            # If no Hail_Size, it falls back to a single color
            footprint_3857.plot(
                ax=ax,
                color='#fd8d3c',
                alpha=opacity,
                zorder=2,
            )
        print("[MapGen] Plotted footprint")
    
    # plot outline
    if show_outline and outline_3857 is not None and not outline_3857.empty:
        outline_3857.boundary.plot(
            ax=ax,
            color=outline_color,   # CHANGED: Use the variable
            linewidth=outline_width, # CHANGED: Use the variable
            zorder=5,              # Increased zorder to ensure it's on top
        )
        print(f"[MapGen] Plotted outline with color {outline_color} and width {outline_width}")
    
    # plot points
    if show_points and points_3857 is not None and not points_3857.empty:
        points_3857.plot(
            ax=ax,
            color='#ff7800',
            markersize=50,
            edgecolor='black',
            linewidth=0.8,
            zorder=4,
        )
        print("[MapGen] Plotted points")
    
    # set the extent
    ax.set_xlim(ext_x1, ext_x2)
    ax.set_ylim(ext_y1, ext_y2)
    
    # now add basemap after plotting data and setting extent
    basemap_added = False
    
    if HAS_CTX and basemap_id and basemap_id != 'none':
        import math
        import requests
        from PIL import Image
        from io import BytesIO
        
        # monkey-patch requests to disable SSL verification
        _original_request = requests.Session.request
        def _patched_request(self, method, url, **kwargs):
            kwargs['verify'] = False 
            return _original_request(self, method, url, **kwargs)
        requests.Session.request = _patched_request
        
        _original_get = requests.get
        def _patched_get(url, **kwargs):
            kwargs['verify'] = False
            return _original_get(url, **kwargs)
        requests.get = _patched_get
        
        # disable warnings
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        print("[MapGen] SSL verification disabled via requests monkey-patch")
        
        # calculate zoom level
        center_lat = (extent_miny + extent_maxy) / 2.0
        width_m = abs(ext_x2 - ext_x1)
        height_m = abs(ext_y2 - ext_y1)
        m_per_px = max(width_m / max(1, width_px), height_m / max(1, height_px))
        initial_res = 156543.03392804097
        adj = max(0.15, math.cos(math.radians(center_lat)))
        zoom_f = math.log2((initial_res * adj) / max(1e-9, m_per_px))
        zoom = int(round(zoom_f))
        zoom = max(1, min(zoom, 18))
        print(f"[MapGen] Calculated zoom level: {zoom}")
        
        # tile URL mapping - all open source and free
        tile_urls = {
            'osm': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            'carto_light': 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
            'carto_positron_nolabels': 'https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png',
            'carto_dark': 'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'carto_voyager': 'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png',
            'esri_gray': 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            'esri_darkgray': 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            'esri_worldstreet': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
            'esri_worldtopo': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            'esri_worldimagery': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        }
        
        url = tile_urls.get(basemap_id, tile_urls.get('carto_light'))
        
        # Try contextily 
        try:
            print(f"[MapGen] Trying contextily with patched requests: {url[:50]}...")
            ctx.add_basemap(
                ax,
                source=url,
                zoom=zoom,
                zorder=1,
                attribution='',
                reset_extent=False
            )
            basemap_added = True
            print(f"[MapGen] SUCCESS: Added basemap via contextily")
        except Exception as e:
            print(f"[MapGen] Contextily failed: {str(e)[:80]}")
        
        # Fallback 1: Manual tile fetching 
        if not basemap_added:
            print("[MapGen] Trying manual tile fetching...")
            try:
                basemap_added = fetch_tiles_manually(
                    ax, ext_x1, ext_y1, ext_x2, ext_y2, zoom, url
                )
            except Exception as e:
                print(f"[MapGen] Manual tile fetch failed: {e}")
        
        #  fallback 2: try different tile servers
        if not basemap_added:
            fallback_servers = ['osm', 'esri_gray', 'esri_worldstreet']
            for server_id in fallback_servers:
                if basemap_added:
                    break
                fallback_url = tile_urls.get(server_id)
                print(f"[MapGen] Trying fallback server: {server_id}...")
                try:
                    basemap_added = fetch_tiles_manually(
                        ax, ext_x1, ext_y1, ext_x2, ext_y2, zoom, fallback_url
                    )
                except Exception as e:
                    print(f"[MapGen] Fallback {server_id} failed: {e}")
    
    if basemap_id == 'none':
        ax.set_facecolor('#f0f0f0')
        print("[MapGen] Set light gray background (no basemap)")
    elif not basemap_added:
        ax.set_facecolor('#f5f5f5')
        print("[MapGen] WARNING: No basemap could be added, using gray background")
    
    # Title
    ax.set_title(f"Hail Footprint – {event_name}", fontsize=14, fontweight='bold', pad=12)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Hail Size (cm)', fontsize=10)
    
    # Scale bar
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(1, units='m', location='lower right', box_alpha=0.7, font_properties={'size': 9}))
    except ImportError:
        pass
    
    # North arrow
    ax.annotate(
        'N',
        xy=(0.97, 0.95), xytext=(0.97, 0.88),
        xycoords='axes fraction',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, linewidth=0))
    
    ax.set_axis_off()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout(pad=0.5)
    plt.savefig(buf, format='png', dpi=dpi, facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    result = buf.read()
    print(f"[MapGen] ========================================")
    print(f"[MapGen] Map generation complete. Size: {len(result)} bytes, Basemap added: {basemap_added}")
    
    if len(result) < 1000:
        raise ValueError("Generated image is too small - rendering may have failed")
    
    return result


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)