import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    BASE_DIR = Path(os.environ.get('APP_BASE_DIR', Path(__file__).parent))
    
   #ocal development
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    OUTPUT_FOLDER = BASE_DIR / 'outputs'
    
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB
    
    ALLOWED_EXTENSIONS = {'csv', 'gpkg', 'shp', 'geojson'}
    
    # Create folders 
    try:
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")