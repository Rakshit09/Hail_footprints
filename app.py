import os
import uuid
import json
import threading
import ssl
from pathlib import Path
from datetime import datetime
import io
import base64

import numpy as np
import pandas as pd

from flask import (
    Flask, render_template, request, jsonify, 
    send_from_directory, url_for, session, send_file
)

from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from config import Config
from processing.footprint import Params, run_footprint, get_processing_status

# ============================================================================
# SSL Certificate Fix for Windows
# This disables SSL verification for tile fetching (contextily/requests)
# Necessary on some Windows systems with corporate firewalls or outdated certs
# ============================================================================
try:
    # Disable SSL verification globally for requests library
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Disable SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create an unverified SSL context for urllib
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print("[SSL] Certificate verification disabled for tile fetching")
except Exception as e:
    print(f"[SSL] Could not disable SSL verification: {e}")

app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store processing jobs
processing_jobs = {}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def guess_column_type(col_name: str) -> str:
    """Guess column type based on name patterns."""
    col_lower = col_name.lower()
    
    # Latitude patterns
    lat_patterns = ['lat', 'latitude', 'y_coord', 'ycoord', 'lat_dd', 'y']
    if any(p in col_lower for p in lat_patterns):
        return 'latitude'
    
    # Longitude patterns
    lon_patterns = ['lon', 'long', 'longitude', 'x_coord', 'xcoord', 'lon_dd', 'x', 'lng']
    if any(p in col_lower for p in lon_patterns):
        return 'longitude'
    
    # Hail size patterns
    hail_patterns = ['hail', 'size', 'diameter', 'diam', 'max_hail', 'hailsize']
    if any(p in col_lower for p in hail_patterns):
        return 'hail_size'
    
    return 'unknown'


def sanitize_for_json(obj):
    """
    Recursively sanitize an object for JSON serialization.
    Handles NaN, Inf, numpy types, etc.
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
    """Main page with upload form."""
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
def upload_file():
    """Handle file upload and return column info."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: CSV, GPKG, GeoJSON, SHP'}), 400
    
    # Generate unique ID for this session
    job_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{job_id}_{timestamp}_{filename}"
    filepath = Config.UPLOAD_FOLDER / saved_filename
    
    file.save(filepath)
    
    # Parse columns
    try:
        if filename.lower().endswith('.csv'):
            # Read with flexible parsing
            df_preview = pd.read_csv(filepath, nrows=25)
            
            # Count total rows (efficiently)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
        else:
            import geopandas as gpd
            gdf = gpd.read_file(filepath)
            df_preview = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore')).head(5)
            row_count = len(gdf)
        
        columns = list(df_preview.columns)
        
        # Auto-detect column types
        column_suggestions = {}
        for col in columns:
            col_type = guess_column_type(col)
            if col_type != 'unknown':
                column_suggestions[col_type] = col
        
        # Convert to records with proper sanitization
        sample_data = df_preview.to_dict('records')
        sample_data = sanitize_for_json(sample_data)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'filename': saved_filename,
            'columns': columns,
            'suggestions': column_suggestions,
            'sample_data': sample_data,
            'row_count': row_count
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400


@app.route('/process', methods=['POST'])
def process_footprint():
    """Start footprint processing."""
    data = request.json
    
    required_fields = ['filename', 'lon_col', 'lat_col', 'hail_col', 'event_name']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    job_id = data.get('job_id', str(uuid.uuid4())[:8])
    
    # Create job-specific output folder
    output_folder = Config.OUTPUT_FOLDER / job_id
    output_folder.mkdir(exist_ok=True)
    
    # Build params
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
    
    # Store job info
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Initializing...',
        'params': params,
        'result': None,
        'error': None
    }
    
    # Start processing in background thread
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
    """Run processing in background thread."""
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
        
        # Sanitize result for JSON
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
    """Get processing status."""
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


@app.route('/viewer/<job_id>')
def viewer(job_id):
    """Interactive map viewer."""
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
    """Serve output files."""
    output_folder = Config.OUTPUT_FOLDER / job_id
    return send_from_directory(output_folder, filename)


@app.route('/geojson/<job_id>')
def get_geojson(job_id):
    """Get GeoJSON data for the map."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'No data available'}), 400
    
    # Get just the filename from result
    geojson_filename = job['result'].get('geojson', '')
    if not geojson_filename:
        return jsonify({'error': 'GeoJSON not found'}), 404
    
    # Build the full path using the job's output folder
    geojson_path = Config.OUTPUT_FOLDER / job_id / geojson_filename
    
    if not geojson_path.exists():
        return jsonify({'error': f'GeoJSON file not found: {geojson_filename}'}), 404
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    return jsonify(sanitize_for_json(data))


@app.route('/footprint_geojson/<job_id>')
def get_footprint_geojson(job_id):
    """Get dissolved footprint GeoJSON."""
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
    """Get input points as GeoJSON."""
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


# Basemap configurations
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
    """Return available basemap options."""
    return jsonify(BASEMAPS)


# Error template
@app.route('/error')
def error_page():
    message = request.args.get('message', 'An error occurred')
    return render_template('error.html', message=message)

@app.route('/debug/<job_id>')
def debug_job(job_id):
    """Debug endpoint to check job status and files."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    output_folder = Config.OUTPUT_FOLDER / job_id
    
    # List files in output folder
    files = []
    if output_folder.exists():
        files = [f.name for f in output_folder.iterdir()]
    
    # Check GeoJSON validity
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
                            # Check first feature coordinates
                            if data.get('features'):
                                first_geom = data['features'][0].get('geometry', {})
                                coords = first_geom.get('coordinates', [])
                                if coords:
                                    # Get sample coordinate
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
    """Render a map with user-specified settings."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    if job['status'] != 'completed' or not job['result']:
        return jsonify({'error': 'Job not complete'}), 400
    
    data = request.json or {}
    
    # Get parameters from request
    bounds = data.get('bounds')  # [minLon, minLat, maxLon, maxLat]
    basemap_id = data.get('basemap', 'carto_light')
    show_footprint = data.get('show_footprint', True)
    show_outline = data.get('show_outline', True)
    show_points = data.get('show_points', False)
    opacity = float(data.get('opacity', 0.6))
    # Optional export sizing (pixels). If omitted, we pick a high-res default.
    width_px = int(data.get('width_px', 3200))
    height_px = int(data.get('height_px', 2000))
    
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
    Manually fetch map tiles and compose them into a basemap.
    This bypasses contextily and uses requests directly with SSL disabled.
    
    All tile servers used are FREE and OPEN SOURCE:
    - OpenStreetMap: Open Database License (ODbL)
    - Carto: Free for most uses
    - ESRI: Free for development/non-commercial
    """
    import requests
    import math
    from PIL import Image
    from io import BytesIO
    import numpy as np
    
    print(f"[ManualTiles] Fetching tiles from: {tile_url_template[:50]}...")
    
    def deg2num(lat_deg, lon_deg, zoom):
        """Convert lat/lon to tile numbers."""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def num2deg(xtile, ytile, zoom):
        """Convert tile numbers to lat/lon of the NW corner."""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def latlon_to_mercator(lat, lon):
        """Convert lat/lon to Web Mercator coordinates."""
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        y = y * 20037508.34 / 180.0
        return x, y
    
    # Convert mercator extent back to lat/lon
    def mercator_to_latlon(x, y):
        lon = x * 180.0 / 20037508.34
        lat = math.atan(math.exp(y * math.pi / 20037508.34)) * 360.0 / math.pi - 90
        return lat, lon
    
    # Get corner coordinates in lat/lon
    lat_min, lon_min = mercator_to_latlon(ext_x1, ext_y1)
    lat_max, lon_max = mercator_to_latlon(ext_x2, ext_y2)
    
    # Get tile range
    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)
    
    # Ensure correct order
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    
    # Limit number of tiles to prevent huge downloads
    max_tiles = 100
    num_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    if num_tiles > max_tiles:
        print(f"[ManualTiles] Too many tiles ({num_tiles}), reducing zoom...")
        return False
    
    print(f"[ManualTiles] Fetching {num_tiles} tiles (x: {x_min}-{x_max}, y: {y_min}-{y_max})")
    
    # Create session with SSL disabled
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'HailFootprintApp/1.0 (Python/requests)',
        'Accept': 'image/png,image/*',
    })
    
    # Fetch all tiles
    tile_size = 256
    tiles = {}
    
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # Format URL - handle both {z}/{x}/{y} and {z}/{y}/{x} patterns
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
    
    # Compose tiles into single image
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    basemap_img = Image.new('RGB', (width, height), (245, 245, 245))
    
    for (x, y), tile_img in tiles.items():
        px = (x - x_min) * tile_size
        py = (y - y_min) * tile_size
        if tile_img.mode != 'RGB':
            tile_img = tile_img.convert('RGB')
        basemap_img.paste(tile_img, (px, py))
    
    # Calculate extent of the composed image in Web Mercator
    nw_lat, nw_lon = num2deg(x_min, y_min, zoom)
    se_lat, se_lon = num2deg(x_max + 1, y_max + 1, zoom)
    
    nw_x, nw_y = latlon_to_mercator(nw_lat, nw_lon)
    se_x, se_y = latlon_to_mercator(se_lat, se_lon)
    
    # Display the basemap
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
):
    """Generate a PNG map with specified settings."""
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
    
    # Load GeoJSON files
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
    
    # Validate we have at least some data to render
    has_data = any([
        footprint_gdf is not None and not footprint_gdf.empty,
        outline_gdf is not None and not outline_gdf.empty,
        points_gdf is not None and not points_gdf.empty
    ])
    
    if not has_data and (not bounds or len(bounds) != 4):
        raise ValueError("No data to render and no bounds provided")
    
    # Determine extent in WGS84
    if bounds and len(bounds) == 4 and all(b is not None for b in bounds):
        extent_minx, extent_miny, extent_maxx, extent_maxy = bounds
    elif footprint_gdf is not None and not footprint_gdf.empty:
        extent_minx, extent_miny, extent_maxx, extent_maxy = footprint_gdf.total_bounds
    elif outline_gdf is not None and not outline_gdf.empty:
        extent_minx, extent_miny, extent_maxx, extent_maxy = outline_gdf.total_bounds
    else:
        raise ValueError("No bounds provided and no footprint data available")
    
    # Validate bounds
    if extent_minx >= extent_maxx or extent_miny >= extent_maxy:
        raise ValueError(f"Invalid bounds: [{extent_minx}, {extent_miny}, {extent_maxx}, {extent_maxy}]")
    
    # Add padding (5%)
    pad_x = (extent_maxx - extent_minx) * 0.05
    pad_y = (extent_maxy - extent_miny) * 0.05
    extent_minx -= pad_x
    extent_maxx += pad_x
    extent_miny -= pad_y
    extent_maxy += pad_y
    
    print(f"[MapGen] WGS84 extent: [{extent_minx:.4f}, {extent_miny:.4f}, {extent_maxx:.4f}, {extent_maxy:.4f}]")
    
    # Convert extent to Web Mercator (EPSG:3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ext_x1, ext_y1 = transformer.transform(extent_minx, extent_miny)
    ext_x2, ext_y2 = transformer.transform(extent_maxx, extent_maxy)
    
    print(f"[MapGen] 3857 extent: [{ext_x1:.0f}, {ext_y1:.0f}, {ext_x2:.0f}, {ext_y2:.0f}]")
    
    # Reproject data to 3857
    footprint_3857 = None
    outline_3857 = None
    points_3857 = None
    
    if footprint_gdf is not None and not footprint_gdf.empty:
        footprint_3857 = footprint_gdf.to_crs(epsg=3857)
    
    if outline_gdf is not None and not outline_gdf.empty:
        outline_3857 = outline_gdf.to_crs(epsg=3857)
    
    if points_gdf is not None and not points_gdf.empty:
        points_3857 = points_gdf.to_crs(epsg=3857)
    
    # Create figure
    dpi = 150  # Reduced DPI for faster generation, still high quality
    fig_w_in = max(6, float(width_px) / dpi)
    fig_h_in = max(4, float(height_px) / dpi)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    
    # Color scale - use the same interpolation as the frontend
    cmap = plt.cm.YlOrRd
    vmin = hail_min if hail_min is not None else 0
    vmax = hail_max if hail_max is not None else 1
    if vmax <= vmin:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot data FIRST, then set extent, then add basemap
    
    # Plot footprint
    if show_footprint and footprint_3857 is not None and not footprint_3857.empty:
        # Check if Hail_Size column exists
        if 'Hail_Size' in footprint_3857.columns:
            footprint_3857.plot(
                ax=ax,
                column='Hail_Size',
                cmap=cmap,
                norm=norm,
                alpha=opacity,
                linewidth=0.3,
                edgecolor='#666666',
                zorder=2,
            )
        else:
            # Fallback: plot with single color
            footprint_3857.plot(
                ax=ax,
                color='#fd8d3c',
                alpha=opacity,
                linewidth=0.3,
                edgecolor='#666666',
                zorder=2,
            )
        print("[MapGen] Plotted footprint")
    
    # Plot outline
    if show_outline and outline_3857 is not None and not outline_3857.empty:
        outline_3857.boundary.plot(
            ax=ax,
            color='black',
            linewidth=2.5,
            zorder=3,
        )
        print("[MapGen] Plotted outline")
    
    # Plot points
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
    
    # Set the extent
    ax.set_xlim(ext_x1, ext_x2)
    ax.set_ylim(ext_y1, ext_y2)
    
    # Now add basemap AFTER plotting data and setting extent
    basemap_added = False
    
    if HAS_CTX and basemap_id and basemap_id != 'none':
        import math
        import requests
        from PIL import Image
        from io import BytesIO
        
        # ================================================================
        # CRITICAL: Monkey-patch requests to disable SSL verification
        # This is necessary on Windows systems with SSL certificate issues
        # ================================================================
        _original_request = requests.Session.request
        def _patched_request(self, method, url, **kwargs):
            kwargs['verify'] = False  # Disable SSL verification
            return _original_request(self, method, url, **kwargs)
        requests.Session.request = _patched_request
        
        # Also patch requests.get directly
        _original_get = requests.get
        def _patched_get(url, **kwargs):
            kwargs['verify'] = False
            return _original_get(url, **kwargs)
        requests.get = _patched_get
        
        # Disable SSL warnings
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        print("[MapGen] SSL verification disabled via requests monkey-patch")
        
        # Calculate appropriate zoom level
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
        
        # Tile URL mapping - all open source and free
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
        
        # Try contextily first (it should now use patched requests)
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
        
        # Fallback: Manual tile fetching with explicit SSL disabled
        if not basemap_added:
            print("[MapGen] Trying manual tile fetching...")
            try:
                basemap_added = fetch_tiles_manually(
                    ax, ext_x1, ext_y1, ext_x2, ext_y2, zoom, url
                )
            except Exception as e:
                print(f"[MapGen] Manual tile fetch failed: {e}")
        
        # Ultimate fallback: try different tile servers
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
    
    # Scale bar (optional)
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
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, linewidth=0),
    )
    
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