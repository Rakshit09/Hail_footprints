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

# Path to proxy ID lookup parquet file
PROXY_PARQUET_PATH = Path(__file__).parent / 'proxy_id_grid_code_hail.parquet'


def process_grid_with_hail_footprint(grid_shapefile: str, hail_geojson: str, output_csv: str, hail_size_field: str = 'Hail_Size') -> dict:
    """process grid with hail footprint"""
    import time
    try:
        # load hail footprint FIRST to get bounds for filtering
        t0 = time.time()
        print(f"[GridProcessor] Loading hail footprint: {hail_geojson}")
        hail_gdf = gpd.read_file(hail_geojson)
        print(f"[GridProcessor] Hail footprint loaded: {len(hail_gdf)} features ({time.time()-t0:.1f}s)")
        
        # get bounds for bbox filter
        hail_bounds = hail_gdf.total_bounds  # [minx, miny, maxx, maxy]
        print(f"[GridProcessor] Hail footprint bounds: {hail_bounds}")
        
        # load shapefile with bbox filter for efficiency
        t1 = time.time()
        print(f"[GridProcessor] Loading grid shapefile with bbox filter: {grid_shapefile}")
        grid_gdf = gpd.read_file(grid_shapefile, bbox=tuple(hail_bounds))
        print(f"[GridProcessor] Grid loaded with bbox filter: {len(grid_gdf)} cells ({time.time()-t1:.1f}s)")
        
        # ensure both are in the same CRS
        if grid_gdf.crs != hail_gdf.crs:
            print(f"[GridProcessor] Reprojecting hail footprint from {hail_gdf.crs} to {grid_gdf.crs}")
            hail_gdf = hail_gdf.to_crs(grid_gdf.crs)
        
        if len(grid_gdf) == 0:
            return {
                'success': False,
                'output_csv': None,
                'n_grid_cells': 0,
                'message': 'No grid cells intersect with the hail footprint extent'
            }
        
        # spatial join 
        t2 = time.time()
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
        
        print(f"[GridProcessor] Spatial join result: {len(joined)} intersections ({time.time()-t2:.1f}s)")
        
        # aggregate to get maximum hail size per grid cell
        print("[GridProcessor] Aggregating maximum hail size per grid cell...")
        
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
        
        # add gridcode if it exists (check various column names)
        if 'gridcode' in result_gdf_wgs84.columns:
            csv_columns.append('gridcode')
        elif 'GRIDCODE' in result_gdf_wgs84.columns:
            result_gdf_wgs84['gridcode'] = result_gdf_wgs84['GRIDCODE']
            csv_columns.append('gridcode')
        
        # add proxy gridcode from 'Code' column if it exists
        if 'Code' in result_gdf_wgs84.columns:
            csv_columns.append('Code')
        elif 'code' in result_gdf_wgs84.columns:
            result_gdf_wgs84['Code'] = result_gdf_wgs84['code']
            csv_columns.append('Code')
        elif 'CODE' in result_gdf_wgs84.columns:
            result_gdf_wgs84['Code'] = result_gdf_wgs84['CODE']
            csv_columns.append('Code')
        
        # select only needed columns
        output_df = result_gdf_wgs84[csv_columns].copy()
        
        # Ensure proper data types
        output_df['longitude'] = output_df['longitude'].astype(float)
        output_df['latitude'] = output_df['latitude'].astype(float)
        output_df[hail_size_field] = output_df[hail_size_field].astype(float)
        if 'gridcode' in output_df.columns:
            output_df['gridcode'] = output_df['gridcode'].astype(int)
        if 'Code' in output_df.columns:
            output_df['Code'] = output_df['Code'].astype(int)
        
        # Join with parquet to get ProxyId
        if 'Code' in output_df.columns and PROXY_PARQUET_PATH.exists():
            try:
                print("[GridProcessor] Loading ProxyId lookup from parquet...")
                unique_codes = output_df['Code'].unique()
                
                # load only Grid_Id and ProxyId columns
                proxy_df = pd.read_parquet(PROXY_PARQUET_PATH, columns=['Grid_Id', 'ProxyId'])
                
                # filter to only matching Grid_Ids
                proxy_df = proxy_df[proxy_df['Grid_Id'].isin(unique_codes)]
                
                # merge on Code = Grid_Id
                output_df = output_df.merge(
                    proxy_df,
                    left_on='Code',
                    right_on='Grid_Id',
                    how='left'
                )
                
                # drop the duplicate Grid_Id column
                if 'Grid_Id' in output_df.columns:
                    output_df = output_df.drop(columns=['Grid_Id'])
                
                # ensure ProxyId is integer
                if 'ProxyId' in output_df.columns:
                    output_df['ProxyId'] = output_df['ProxyId'].fillna(0).astype(int)
                    print(f"[GridProcessor] added ProxyId for {(output_df['ProxyId'] > 0).sum()} cells")
                
            except Exception as e:
                print(f"[GridProcessor] warning: could not load ProxyId: {e}")
        
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

