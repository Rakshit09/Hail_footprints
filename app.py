import os
import uuid
import json
import threading
import ssl
import time
import io
import base64
import re
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import urllib3
from flask import (Flask, render_template, request, jsonify, send_from_directory, 
                   url_for, session, send_file, Response, make_response)
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd

# -- Import your local modules --
from config import Config
from processing.footprint import Params, run_footprint, get_processing_status

# ==========================================
# SSL / NETWORK SETUP
# ==========================================
try:
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    ssl._create_default_https_context = ssl._create_unverified_context
    print("[SSL] Certificate verification disabled")
except Exception as e:
    print(f"[SSL] Error disabling SSL: {e}")

# Monkey patch requests for internal logic
_original_request = requests.Session.request
def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)
requests.Session.request = _patched_request

# ==========================================
# APP SETUP
# ==========================================
app = Flask(__name__)
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(
    app,
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    cors_allowed_origins="*",
    max_http_buffer_size=100 * 1024 * 1024, 
    async_handlers=True
)

# Ensure folders exist
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ==========================================
# STATE MANAGEMENT (FILE BASED)
# ==========================================
def get_status_file(job_id):
    """Path to the status JSON file for a specific job."""
    return Config.OUTPUT_FOLDER / job_id / 'status.json'

def save_job_state(job_id, data):
    """
    Save job state to disk atomically.
    This enables sharing state between multiple Gunicorn workers.
    """
    try:
        file_path = get_status_file(job_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sanitize data to ensure it's JSON serializable
        clean_data = sanitize_for_json(data)
        
        # Atomic write: write to temp file then rename
        temp_path = file_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(clean_data, f)
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Error saving state for {job_id}: {e}")

def load_job_state(job_id):
    """Load job state from disk."""
    try:
        file_path = get_status_file(job_id)
        if not file_path.exists():
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def sanitize_for_json(obj):
    """Recursively clean data for JSON response."""
    if obj is None: return None
    # Handle Path objects (convert to string)
    if isinstance(obj, Path): return str(obj)
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
    col_lower = str(col_name).lower().strip()
    # Lat
    if col_lower in ['lat', 'latitude', 'y', 'y_coord', 'lat_dd', 'slat']: return 'latitude'
    if re.search(r'(^|[\s_])(lat|latitude)([\s_]|$)', col_lower): return 'latitude'
    # Lon
    if col_lower in ['lon', 'long', 'lng', 'longitude', 'x', 'x_coord', 'lon_dd', 'slon']: return 'longitude'
    if re.search(r'(^|[\s_])(lon|lng|long|longitude)([\s_]|$)', col_lower): return 'longitude'
    # Hail
    if any(x in col_lower for x in ['hail', 'size', 'diameter', 'mag']): return 'hail_size'
    return 'unknown'

# ==========================================
# ERROR HANDLERS
# ==========================================
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

# ==========================================
# MAIN ROUTES
# ==========================================
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
                row_count = 0
        else:
            gdf = gpd.read_file(str(file_path))
            df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore')).head(25)
            row_count = len(gdf)
            
        columns = list(df.columns)
        suggestions = {}
        for c in columns:
            t = guess_column_type(c)
            if t != 'unknown': suggestions[t] = c
            
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
    """Start processing job."""
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
            grouping_threshold_km=float(data.get('grouping_threshold_km', 30.0)),
            large_buffer_km=float(data.get('large_buffer_km', 10.0)),
            small_buffer_km=float(data.get('small_buffer_km', 5.0)),
            out_folder=str(out_dir),
            job_id=job_id
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
    """Background worker function."""
    
    # Load existing or create new state
    state = load_job_state(job_id) or {}
    
    def progress_callback(progress, message):
        # Update state
        state['progress'] = progress
        state['message'] = message
        state['status'] = 'processing'
        save_job_state(job_id, state)
            
    try:
        progress_callback(1, "Starting analysis...")
        
        # Run the heavy logic
        result = run_footprint(params, progress_callback=progress_callback)
        
        # Completion
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
    """Check status via file system."""
    state = load_job_state(job_id)
    if not state:
        # If ID is valid format but file not found, likely starting up
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
        
    # Get raw filename/path from result
    raw_fname = state['result'].get('geojson')
    if not raw_fname: 
        print(f"[404 Error] Job {job_id} result has no 'geojson' key")
        return make_json_response({'error': 'No file recorded'}, 404)
    
    # FIX: Ensure we only use the filename, not a full path from a previous environment
    filename = Path(raw_fname).name
    
    # Construct expected path
    file_path = Config.OUTPUT_FOLDER / job_id / filename
    
    if not file_path.exists(): 
        print(f"[404 Error] File not found at: {file_path}")
        return make_json_response({'error': 'File missing on disk'}, 404)
    
    try:
        with open(file_path, 'r') as f:
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
    
    # FIX: Use .name to strip directories
    file_path = Config.OUTPUT_FOLDER / job_id / Path(raw_fname).name
    
    if not file_path.exists(): return make_json_response({'error': 'File missing'}, 404)
    with open(file_path, 'r') as f: return make_json_response(json.load(f))

@app.route('/points_geojson/<job_id>')
def get_points_geojson(job_id):
    state = load_job_state(job_id)
    if not state or not state.get('result'): 
        return make_json_response({'error': 'Not ready'}, 404)
    
    raw_fname = state['result'].get('points_geojson')
    if not raw_fname: return make_json_response({'error': 'No file'}, 404)
    
    # FIX: Use .name to strip directories
    file_path = Config.OUTPUT_FOLDER / job_id / Path(raw_fname).name
    
    if not file_path.exists(): return make_json_response({'error': 'File missing'}, 404)
    with open(file_path, 'r') as f: return make_json_response(json.load(f))

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
        p = output_folder / Path(geojson_file).name 
        if p.exists(): 
            footprint_gdf = gpd.read_file(p)
            print(f"[MapGen] Loaded footprint: {len(footprint_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Footprint file not found: {p}")
    
    if show_outline and footprint_file:
        p = output_folder / Path(footprint_file).name 
        if p.exists(): 
            outline_gdf = gpd.read_file(p)
            print(f"[MapGen] Loaded outline: {len(outline_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Outline file not found: {p}")
    
    if show_points and points_file:
        p = output_folder / Path(points_file).name 
        if p.exists(): 
            points_gdf = gpd.read_file(p)
            print(f"[MapGen] Loaded points: {len(points_gdf)} features")
        else:
            print(f"[MapGen] WARNING: Points file not found: {p}")
    
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
                    # CONTINUOUS MODE: Normalized Convolution for correct value preservation
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
                    
                    # Estimate reasonable sigma based on CELL SIZE
                    # Calculate average cell area in map units (approx)
                    # We can use the bounds of the first few polygons to guess size
                    avg_cell_width = 0
                    if not footprint_3857.empty:
                        # take a sample
                        sample = footprint_3857.geometry.iloc[:20]
                        total_area = sum(g.area for g in sample)
                        avg_area = total_area / len(sample)
                        avg_cell_width = np.sqrt(avg_area)
                    
                    # Convert map units to pixels
                    # width_m / rast_width = meters_per_pixel
                    map_width_m = abs(rast_maxx - rast_minx)
                    m_per_px = map_width_m / rast_width
                    
                    cell_size_px = avg_cell_width / m_per_px if m_per_px > 0 else 10
                    
                    # Heuristic: sigma approx 0.6x cell size preserves >95% of peak value (matches viewer opaque center)
                    sigma = max(2.0, cell_size_px * 0.6)
                    print(f"[MapGen] Calc sigma: {sigma:.2f} (cell_size_px: {cell_size_px:.1f})")

                    # Create transform for rasterization
                    transform = from_bounds(rast_minx, rast_miny, rast_maxx, rast_maxy, 
                                           rast_width, rast_height)
                    
                    # 1. Rasterize Values (Signal) * Mask
                    # shapes = (geometry, value)
                    shapes_val = [(geom, val) for geom, val in zip(footprint_3857.geometry, 
                                                                footprint_3857['Hail_Size'])]
                    raster_val = rasterize(shapes_val, out_shape=(rast_height, rast_width), 
                                       transform=transform, fill=0, dtype='float32')
                    
                    # 2. Rasterize Weights (Binary Mask)
                    # shapes = (geometry, 1)
                    shapes_weight = [(geom, 1.0) for geom in footprint_3857.geometry]
                    raster_weight = rasterize(shapes_weight, out_shape=(rast_height, rast_width),
                                              transform=transform, fill=0, dtype='float32')
                    
                    print(f"[MapGen] Raster non-zero pixels: {np.count_nonzero(raster_weight)}")
                    
                    # 3. Apply Gaussian Blur to both (Normalized Convolution)
                    smoothed_val = gaussian_filter(raster_val, sigma=sigma, mode='constant', cval=0)
                    smoothed_weight = gaussian_filter(raster_weight, sigma=sigma, mode='constant', cval=0)
                    
                    # 4. Normalize
                    # Avoid division by zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        normalized = smoothed_val / smoothed_weight
                        # Fill undefined (0/0) or unstable areas with 0
                        normalized[smoothed_weight < 1e-6] = 0
                        
                    print(f"[MapGen] Normalized non-zero: {np.count_nonzero(normalized)}")
                    
                    # 5. Create Alpha Mask for Soft Edges
                    # Instead of hard clipping, use the smoothed_weight as the alpha map
                    # Scale weight to 0-1 range for opacity
                    # Weights > 0.5 usually mean "inside" a cell, < 0.5 is "fade out"
                    # We boost it slightly so the center of cells is fully opaque
                    alpha_map = smoothed_weight  # raw weight is approx 0..1 (locally)
                    
                    # Normalize alpha map to verify max is near 1
                    max_weight = np.max(smoothed_weight)
                    if max_weight > 0:
                        alpha_map = alpha_map / max_weight
                        
                    # Apply global opacity
                    # We can use a colormap that handles alpha, or manually create RGBA
                    
                    # Get RGBA from colormap
                    norm_data = norm(normalized)
                    rgba_img = cmap(norm_data) # (H, W, 4)
                    
                    # Replace Alpha channel
                    # Combine local softness (alpha_map) with global opacity settings
                    # alpha_map gives the shape/fading
                    final_alpha = alpha_map * opacity
                    
                    # Threshold very low alpha to keep file size sanity? maybe not needed
                    rgba_img[..., 3] = final_alpha
                    
                    # Plot as image
                    ax.imshow(rgba_img, extent=[rast_minx, rast_maxx, rast_miny, rast_maxy], 
                             origin='upper', zorder=2, interpolation='bicubic', aspect='auto')
                             
                    print(f"[MapGen] Plotted footprint in CONTINUOUS mode (Normalized Conv)")
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