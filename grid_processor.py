"""
Grid Processor 
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def process_grid_with_hail_footprint(grid_shapefile: str, hail_geojson: str, output_csv: str, hail_size_field: str = 'Hail_Size') -> dict:
    """process grid with hail footprint"""
    try:
        # load shapefile
        print(f"[GridProcessor] Loading grid shapefile: {grid_shapefile}")
        grid_gdf = gpd.read_file(grid_shapefile)
        print(f"[GridProcessor] Grid loaded: {len(grid_gdf)} cells")
        
        # load hail footprint
        print(f"[GridProcessor] Loading hail footprint: {hail_geojson}")
        hail_gdf = gpd.read_file(hail_geojson)
        print(f"[GridProcessor] Hail footprint loaded: {len(hail_gdf)} features")
        
        # ensure both are in the same CRS
        if grid_gdf.crs != hail_gdf.crs:
            print(f"[GridProcessor] Reprojecting hail footprint from {hail_gdf.crs} to {grid_gdf.crs}")
            hail_gdf = hail_gdf.to_crs(grid_gdf.crs)
        
        # clip grid to hail footprint extent
        hail_bounds = hail_gdf.total_bounds  # [minx, miny, maxx, maxy]
        print(f"[GridProcessor] Hail footprint bounds: {hail_bounds}")
        
        #bounding box filter
        grid_gdf = grid_gdf.cx[hail_bounds[0]:hail_bounds[2], hail_bounds[1]:hail_bounds[3]]
        print(f"[GridProcessor] Grid cells after bbox filter: {len(grid_gdf)}")
        
        if len(grid_gdf) == 0:
            return {
                'success': False,
                'output_csv': None,
                'n_grid_cells': 0,
                'message': 'No grid cells intersect with the hail footprint extent'
            }
        
        # spatial join 
        print("[GridProcessor] Performing spatial join...")
        
        # ensure hail_size_field exists
        if hail_size_field not in hail_gdf.columns:
            available_cols = [c for c in hail_gdf.columns if c != 'geometry']
            return {
                'success': False,
                'output_csv': None,
                'n_grid_cells': 0,
                'message': f"Field '{hail_size_field}' not found. Available: {available_cols}"
            }
        
        # aggregate to get maximum hail size per grid cell
        print("[GridProcessor] Aggregating maximum hail size per grid cell...")
        
        # keep track of original grid geometry
        grid_gdf_indexed = grid_gdf.copy()
        grid_gdf_indexed['_grid_idx'] = range(len(grid_gdf_indexed))
        
        # spatial join with index tracking
        joined = gpd.sjoin(
            grid_gdf_indexed,
            hail_gdf[[hail_size_field, 'geometry']],
            how='inner',
            predicate='intersects'
        )
        
        print(f"[GridProcessor] Spatial join result: {len(joined)} intersections")
        
        if len(joined) == 0:
            return {
                'success': False,
                'output_csv': None,
                'n_grid_cells': 0,
                'message': 'No grid cells intersect with hail footprint polygons'
            }
        
        # aggregate max hail size per grid cell
        agg_df = joined.groupby('_grid_idx').agg({
            hail_size_field: 'max'
        }).reset_index()
        
        # merge back with grid geometry
        result_gdf = grid_gdf_indexed.merge(agg_df, on='_grid_idx', how='inner', suffixes=('_orig', ''))
        
        # handle column naming if there was a conflict
        if f'{hail_size_field}_orig' in result_gdf.columns:
            result_gdf = result_gdf.drop(columns=[f'{hail_size_field}_orig'])
        
        print(f"[GridProcessor] Unique grid cells with hail values: {len(result_gdf)}")
        
        # calculate centroids and prepare CSV output
        print("[GridProcessor] Calculating centroids...")
        
        # Ensure WGS84 for lat/lon output
        if result_gdf.crs and result_gdf.crs.to_epsg() != 4326:
            result_gdf_wgs84 = result_gdf.to_crs(epsg=4326)
        else:
            result_gdf_wgs84 = result_gdf
        
        result_gdf_wgs84['centroid'] = result_gdf_wgs84.geometry.centroid
        result_gdf_wgs84['longitude'] = result_gdf_wgs84['centroid'].x
        result_gdf_wgs84['latitude'] = result_gdf_wgs84['centroid'].y
        
        # create output CSV
        csv_columns = ['longitude', 'latitude', hail_size_field]
        
        # add gridcode if it exists
        if 'gridcode' in result_gdf_wgs84.columns:
            csv_columns.append('gridcode')
        elif 'GRIDCODE' in result_gdf_wgs84.columns:
            result_gdf_wgs84['gridcode'] = result_gdf_wgs84['GRIDCODE']
            csv_columns.append('gridcode')
        
        # select only needed columns
        output_df = result_gdf_wgs84[csv_columns].copy()
        
        # sort by hail size descending for convenience
        output_df = output_df.sort_values(by=hail_size_field, ascending=False)
        
        # export to CSV
        print(f"[GridProcessor] Exporting to CSV: {output_csv}")
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_csv, index=False)
        
        print(f"[GridProcessor] Complete! {len(output_df)} grid cells exported")
        
        return {
            'success': True,
            'output_csv': str(output_csv),
            'n_grid_cells': len(output_df),
            'message': f'Successfully processed {len(output_df)} grid cells'
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'output_csv': None,
            'n_grid_cells': 0,
            'message': f'Error: {str(e)}'
        }


def validate_shapefile(shapefile_path: str) -> dict:
    """validate shapefile """
    try:
        gdf = gpd.read_file(shapefile_path)
        
        return {
            'valid': True,
            'n_features': len(gdf),
            'crs': str(gdf.crs) if gdf.crs else 'Unknown',
            'geometry_type': gdf.geometry.geom_type.unique().tolist(),
            'bounds': gdf.total_bounds.tolist(),
            'columns': [c for c in gdf.columns if c != 'geometry'],
            'message': 'Shapefile is valid'
        }
        
    except Exception as e:
        return {
            'valid': False,
            'n_features': 0,
            'crs': None,
            'geometry_type': None,
            'bounds': None,
            'columns': [],
            'message': f'Error reading shapefile: {str(e)}'
        }


# testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python grid_processor.py <grid_shapefile> <hail_geojson> <output_csv>")
        sys.exit(1)
    
    result = process_grid_with_hail_footprint(
        grid_shapefile=sys.argv[1],
        hail_geojson=sys.argv[2],
        output_csv=sys.argv[3]
    )
    
    print(f"\nResult: {result}")

