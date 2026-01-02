import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Use environment variable for base directory (Posit Connect uses different paths)
    BASE_DIR = Path(os.environ.get('APP_BASE_DIR', Path(__file__).parent))
    
    # For Posit Connect, use a writable directory
    # Posit Connect typically allows writing to /tmp or app-specific directories
    if os.environ.get('RSTUDIO_PRODUCT') or os.environ.get('CONNECT_SERVER'):
        # Running on Posit Connect
        UPLOAD_FOLDER = Path('/tmp/hail_app/uploads')
        OUTPUT_FOLDER = Path('/tmp/hail_app/outputs')
    else:
        # Local development
        UPLOAD_FOLDER = BASE_DIR / 'uploads'
        OUTPUT_FOLDER = BASE_DIR / 'outputs'
    
    # Reduced max upload for Posit Connect (10MB default, adjust as needed)
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_UPLOAD_SIZE', 10 * 1024 * 1024))
    
    ALLOWED_EXTENSIONS = {'csv', 'gpkg', 'shp', 'geojson'}
    
    # Create folders if they don't exist
    try:
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")