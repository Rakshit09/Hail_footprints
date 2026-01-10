import os
import uuid
import json
import threading
import ssl
import io
import re
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import urllib3
from flask import (Flask, render_template, request, jsonify, send_from_directory, 
url_for, session, send_file, Response, make_response)
from werkzeug.utils import secure_filename
from flask_cors import CORS
import geopandas as gpd
from config import Config
from processing.footprint import Params, run_footprint
from export_map import generate_map_png

# SSL setup 
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
    print(f"[SSL] Using certifi certificates: {certifi.where()}")
except ImportError:
    print("[SSL] Warning: certifi not installed, using system certificates")
except Exception as e:
    print(f"[SSL] Error setting up SSL: {e}")

# setup
app = Flask(__name__)
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

CORS(app, resources={r"/*": {"origins": "*"}})

# ensure folders exist
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# state management
def get_status_file(job_id):
    """path to the status JSON file for a specific job."""
    return Config.OUTPUT_FOLDER / job_id / 'status.json'

def save_job_state(job_id, data):
    """save job state to disk atomically."""
    try:
        file_path = get_status_file(job_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        #  ensure data is realizable
        clean_data = sanitize_for_json(data)
        
        # write to temp file then rename
        temp_path = file_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(clean_data, f)
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Error saving state for {job_id}: {e}")

def load_job_state(job_id):
    """load job state from disk."""
    try:
        file_path = get_status_file(job_id)
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# utility functions
def sanitize_for_json(obj):
    """recursively clean data for JSON response."""
    if obj is None: return None
    # Handle Path objects (convert to string)
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, datetime): return obj.isoformat() 
    if isinstance(obj, dict): return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray): return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.bool_): return bool(obj)
    if pd.isna(obj): return None
    return obj
    

def make_json_response(data, status_code=200):
    """Helper to force proper JSON headers."""
    response = make_response(json.dumps(sanitize_for_json(data)), status_code)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def guess_column_type(col_name):
    """
    guess the column type and return (type, priority).
    Priority 1 = exact match, Priority 2 = partial match.
    """
    col_lower = str(col_name).lower().strip()
    
    # Lat - exact  (priority 1)
    if col_lower in ['lat', 'latitude', 'y', 'y_coord', 'lat_dd', 'slat']: 
        return ('latitude', 1)
    # Lat - partial  (priority 2)
    if re.search(r'(^|[\s_])(lat|latitude)([\s_]|$)', col_lower): 
        return ('latitude', 2)
    
    # Lon - exact  (priority 1)
    if col_lower in ['lon', 'long', 'lng', 'longitude', 'x', 'x_coord', 'lon_dd', 'slon']: 
        return ('longitude', 1)
    # Lon - partial  (priority 2)
    if re.search(r'(^|[\s_])(lon|lng|long|longitude)([\s_]|$)', col_lower): 
        return ('longitude', 2)
    
    # Hail - exact  (priority 1)
    if col_lower in ['hail', 'hail_size', 'hailsize', 'size', 'diameter', 'mag', 'magnitude']:
        return ('hail_size', 1)
    # Hail - partial  (priority 2)
    if any(x in col_lower for x in ['hail', 'size', 'diameter', 'mag']): 
        return ('hail_size', 2)
    
    # QC - exact  (priority 1)
    if col_lower in ['qc', 'qclevel', 'qc_level', 'quality', 'quality_level', 'qualitylevel', 'qc_flag', 'qcflag']:
        return ('qc_level', 1)
    # QC - partial  (priority 2)
    if re.search(r'(^|[\s_])(qc|quality)([\s_]|level)', col_lower):
        return ('qc_level', 2)
    
    return ('unknown', 99)

# error handlers
@app.after_request
def add_header(response):
    if request.path.startswith('/upload') or request.path.startswith('/process'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.errorhandler(413)
def request_entity_too_large(error):
    return make_json_response({'error': 'File too large'}, 413)

@app.errorhandler(500)
def internal_error(error):
    return make_json_response({'error': 'Internal server error'}, 500)

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/status') or request.path.startswith('/process'):
        return make_json_response({'error': 'Resource not found'}, 404)
    return render_template('error.html', message='Page not found'), 404

# main routes
@app.route('/')
def index():
    max_bytes = getattr(Config, 'MAX_CONTENT_LENGTH', 50 * 1024 * 1024)
    return render_template('index.html', max_upload_bytes=max_bytes)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return make_json_response({'error': 'No file provided'}, 400)
        
        file = request.files['file']
        if not file or file.filename == '':
            return make_json_response({'error': 'No file selected'}, 400)
        
        if not allowed_file(file.filename):
            return make_json_response({'error': 'File type not supported'}, 400)
            
        job_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename) or f"upload_{job_id}.csv"
        saved_name = f"{job_id}_{datetime.now().strftime('%H%M%S')}_{filename}"
        
        file_path = Config.UPLOAD_FOLDER / saved_name
        file.save(str(file_path))
        
        # Parse preview
        if filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path, nrows=25)
                # Quick row count
                with open(file_path, 'rb') as f:
                    row_count = sum(1 for _ in f) - 1
            except:
                df = pd.read_csv(file_path, nrows=25, encoding='latin1')
                with open(file_path, 'rb') as f:
                    row_count = sum(1 for _ in f) - 1
        else:
            gdf = gpd.read_file(str(file_path))
            df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore')).head(25)
            row_count = len(gdf)
            
        columns = list(df.columns)
        suggestions = {}
        priorities = {}  # Track priority 
        for c in columns:
            col_type, priority = guess_column_type(c)
            if col_type != 'unknown':
                if col_type not in suggestions or priority < priorities[col_type]:
                    suggestions[col_type] = c
                    priorities[col_type] = priority
        
        print(f"  Final suggestions: {suggestions}")
            
        return make_json_response({
            'success': True,
            'job_id': job_id,
            'filename': saved_name,
            'columns': columns,
            'suggestions': suggestions,
            'sample_data': df.to_dict('records'),
            'row_count': row_count
        })
        
    except Exception as e:
        traceback.print_exc()
        return make_json_response({'error': str(e)}, 500)

@app.route('/process', methods=['POST'])
def process_footprint():
    """start processing job."""
    try:
        data = request.json
        if not data:
            return make_json_response({'error': 'No JSON data'}, 400)

        job_id = data.get('job_id') or str(uuid.uuid4())[:8]
        
        # Create output folder
        out_dir = Config.OUTPUT_FOLDER / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        params = Params(
            input_points=str(Config.UPLOAD_FOLDER / data['filename']),
            hail_field=data['hail_col'],
            event_name=data['event_name'],
            lon_col=data['lon_col'],
            lat_col=data['lat_col'],
            qc_col=data.get('qc_col') or None,
            proxy_hail_size=float(data.get('proxy_hail_size')) if data.get('proxy_hail_size') not in (None, '', 0) else None,
            grouping_threshold_km=float(data.get('grouping_threshold_km', 30.0)),
            large_buffer_km=float(data.get('large_buffer_km', 10.0)),
            small_buffer_km=float(data.get('small_buffer_km', 5.0)),
            out_folder=str(out_dir),
            job_id=job_id,
        )
        
        # Initialize Status File
        initial_state = {
            'status': 'queued',
            'progress': 0,
            'message': 'Initializing...',
            'params': {'event_name': data['event_name']},
            'result': None
        }
        save_job_state(job_id, initial_state)
        
        # Start Thread
        thread = threading.Thread(target=run_processing_job, args=(job_id, params))
        thread.daemon = True
        thread.start()
        
        return make_json_response({'success': True, 'job_id': job_id})
        
    except Exception as e:
        traceback.print_exc()
        return make_json_response({'error': str(e)}, 500)

def run_processing_job(job_id, params):
    """background worker function."""
    
    # load existing or create new state
    state = load_job_state(job_id) or {}
    
    def progress_callback(progress, message):
        # update state
        state['progress'] = progress
        state['message'] = message
        state['status'] = 'processing'
        save_job_state(job_id, state)
            
    try:
        progress_callback(1, "Starting analysis...")
        
        # run logic
        result = run_footprint(params, progress_callback=progress_callback)
        
        # completion
        state['status'] = 'completed'
        state['progress'] = 100
        state['message'] = 'Done'
        state['result'] = result
        save_job_state(job_id, state)
        
    except Exception as e:
        traceback.print_exc()
        state['status'] = 'error'
        state['error'] = str(e)
        save_job_state(job_id, state)

@app.route('/status/<job_id>')
def get_status(job_id):
    """check status via file system."""
    state = load_job_state(job_id)
    if not state:
        return make_json_response({'status': 'queued', 'progress': 0, 'message': 'Starting...'})
    return make_json_response(state)

@app.route('/viewer/<job_id>')
def viewer(job_id):
    state = load_job_state(job_id)
    if not state:
        return render_template('error.html', message='Job not found'), 404
    
    event_name = state.get('params', {}).get('event_name', 'Event')
    return render_template('viewer.html', job_id=job_id, result=state.get('result'), event_name=event_name)


@app.route('/outputs/<job_id>/<filename>')
def serve_output(job_id, filename):
    """serve output files"""
    output_folder = Config.OUTPUT_FOLDER / job_id
    return send_from_directory(output_folder, filename)


@app.route('/geojson/<job_id>')
def get_geojson(job_id):
    state = load_job_state(job_id)
    if not state or not state.get('result'):
        print(f"[404 Error] Job {job_id} not ready or has no result")
        return make_json_response({'error': 'Not ready'}, 404)
        
    # get raw filename/path from result
    raw_fname = state['result'].get('geojson')
    if not raw_fname: 
        print(f"[404 Error] Job {job_id} result has no 'geojson' key")
        return make_json_response({'error': 'No file recorded'}, 404)
    
    # ensure filename
    filename = Path(raw_fname).name
    
    # construct path
    file_path = Config.OUTPUT_FOLDER / job_id / filename
    
    if not file_path.exists(): 
        print(f"[404 Error] File not found at: {file_path}")
        return make_json_response({'error': 'File missing on disk'}, 404)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return make_json_response(json.load(f))
    except Exception as e:
        print(f"[500 Error] Failed to read JSON: {e}")
        return make_json_response({'error': 'Corrupt file'}, 500)

@app.route('/footprint_geojson/<job_id>')
def get_footprint_geojson(job_id):
    state = load_job_state(job_id)
    if not state or not state.get('result'): 
        return make_json_response({'error': 'Not ready'}, 404)
    
    raw_fname = state['result'].get('footprint_geojson')
    if not raw_fname: return make_json_response({'error': 'No file'}, 404)
    
    # use filename
    file_path = Config.OUTPUT_FOLDER / job_id / Path(raw_fname).name
    
    if not file_path.exists(): return make_json_response({'error': 'File missing'}, 404)
    with open(file_path, 'r', encoding='utf-8') as f: return make_json_response(json.load(f))

@app.route('/points_geojson/<job_id>')
def get_points_geojson(job_id):
    state = load_job_state(job_id)
    if not state or not state.get('result'): 
        return make_json_response({'error': 'Not ready'}, 404)
    
    raw_fname = state['result'].get('points_geojson')
    if not raw_fname: return make_json_response({'error': 'No file'}, 404)
    
    file_path = Config.OUTPUT_FOLDER / job_id / Path(raw_fname).name
    
    if not file_path.exists(): return make_json_response({'error': 'File missing'}, 404)
    with open(file_path, 'r', encoding='utf-8') as f: return make_json_response(json.load(f))

@app.route('/grid_csv/<job_id>')
def get_grid_csv(job_id):
    """get grid cells as CSV with centroid coordinates and hail size"""
    state = load_job_state(job_id)
    if not state or not state.get('result'):
        return make_json_response({'error': 'Not ready'}, 404)
    
    raw_fname = state['result'].get('geojson')
    if not raw_fname:
        return make_json_response({'error': 'No file'}, 404)
    
    file_path = Config.OUTPUT_FOLDER / job_id / Path(raw_fname).name
    
    if not file_path.exists():
        return make_json_response({'error': 'File missing'}, 404)
    
    try:
        # read geojson
        gdf = gpd.read_file(str(file_path))
        
        # calculate centroids
        gdf['centroid'] = gdf.geometry.centroid
        gdf['longitude'] = gdf['centroid'].x
        gdf['latitude'] = gdf['centroid'].y
        
        # create csv with columns
        csv_columns = ['longitude', 'latitude', 'Hail_Size']
        if 'gridcode' in gdf.columns:
            csv_columns.append('gridcode')
        csv_df = gdf[csv_columns].copy()
        
        # generate csv string
        csv_output = csv_df.to_csv(index=False)
        
        # return as downloadable 
        event_name = state.get('params', {}).get('event_name', 'grid')
        filename = f"{event_name}_grid.csv"
        
        return send_file(
            io.BytesIO(csv_output.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        traceback.print_exc()
        return make_json_response({'error': str(e)}, 500)

# basemap 
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
    state = load_job_state(job_id)
    if not state:
        return make_json_response({'error': 'Job not found'}, 404)
    
    output_folder = Config.OUTPUT_FOLDER / job_id
    
    # list files in output folder
    files = []
    if output_folder.exists():
        files = [f.name for f in output_folder.iterdir()]
    
    # check GeoJSON validity
    geojson_check = {}
    if state.get('result'):
        for key in ['geojson', 'footprint_geojson', 'points_geojson']:
            filename = state['result'].get(key, '')
            if filename:
                filepath = output_folder / Path(filename).name
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
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
        'status': state.get('status'),
        'result': sanitize_for_json(state.get('result')),
        'files_in_folder': files,
        'geojson_check': geojson_check
    })

@app.route('/render_map/<job_id>', methods=['POST'])
def render_map(job_id):
    """render a map with user-specified settings"""
    state = load_job_state(job_id)
    if not state:
        return make_json_response({'error': 'Job not found'}, 404)
    
    if state.get('status') != 'completed' or not state.get('result'):
        return make_json_response({'error': 'Job not complete'}, 400)
    
    data = request.json or {}
    
    # get parameters 
    bounds = data.get('bounds')  # [minLon, minLat, maxLon, maxLat]
    basemap_id = data.get('basemap', 'carto_light')
    show_footprint = data.get('show_footprint', True)
    show_outline = data.get('show_outline', True)
    show_points = data.get('show_points', False)
    opacity = float(data.get('opacity', 0.7))
    width_px = int(data.get('width_px', 3200))
    height_px = int(data.get('height_px', 2000))
    outline_width = data.get('outline_width', 1.5)
    outline_color = data.get('outline_color', '#000000')
    display_mode = data.get('display_mode', 'cells')
    color_map = data.get('color_map', 'ylOrRd')
    
    output_folder = Config.OUTPUT_FOLDER / job_id
    result = state['result']
    
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
            event_name=state.get('params', {}).get('event_name', 'Event'),
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
            download_name=f"{state.get('params', {}).get('event_name', 'map')}_custom_map.png"
        )
        
    except Exception as e:
        traceback.print_exc()
        return make_json_response({'error': str(e)}, 500)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)