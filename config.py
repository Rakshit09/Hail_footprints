import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    OUTPUT_FOLDER = BASE_DIR / 'outputs'
    
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    ALLOWED_EXTENSIONS = {'csv', 'gpkg', 'shp', 'geojson'}
    
    # Create folders if they don't exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)