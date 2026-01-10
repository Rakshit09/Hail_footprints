from __future__ import annotations
import os
import math
import time
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import CRS
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, GeometryCollection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Numba check
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _idw_core_numba(dists: np.ndarray, vals: np.ndarray, power: float) -> np.ndarray:
        n = dists.shape[0]
        k = dists.shape[1]
        out = np.empty(n, dtype=np.float64)
        
        for i in prange(n):
            has_hit = False
            for j in range(k):
                if dists[i, j] == 0.0:
                    out[i] = vals[i, j]
                    has_hit = True
                    break
            
            if not has_hit:
                w_sum = 0.0
                wv_sum = 0.0
                for j in range(k):
                    w = 1.0 / (dists[i, j] ** power)
                    w_sum += w
                    wv_sum += w * vals[i, j]
                out[i] = wv_sum / w_sum
        
        return out


def _split_gdb_path(path: str) -> Tuple[str, Optional[str]]:
    lower = path.lower()
    if ".gdb" not in lower:
        return path, None
    idx = lower.rfind(".gdb")
    gdb = path[: idx + 4]
    rest = path[idx + 4:].lstrip("\\/")
    return gdb, rest if rest else None


def read_vector(path: str) -> gpd.GeoDataFrame:
    ds, layer = _split_gdb_path(path)
    return gpd.read_file(ds, layer=layer) if layer else gpd.read_file(ds)


def read_points_csv(
    csv_path: str,
    lon_col: str,
    lat_col: str,
    hail_col: str,
    qc_col: Optional[str] = None,
    proxy_hail_size: Optional[float] = None
) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    
    print(f"  Initial rows: {len(df)}")
    
    for c in (lon_col, lat_col, hail_col):
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")
    
    # show sample values
    print(f"  Sample values before conversion:")
    print(f"    {lon_col}: {df[lon_col].head(3).tolist()}")
    print(f"    {lat_col}: {df[lat_col].head(3).tolist()}")
    print(f"    {hail_col}: {df[hail_col].head(3).tolist()}")
    
    # track rows 
    initial_count = len(df)
    
    for col in (lon_col, lat_col, hail_col):
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        converted_to_na = after_na - before_na
        if converted_to_na > 0:
            print(f"  WARNING: {converted_to_na} values in '{col}' could not be converted to numbers")
    
    # use proxy_hail_size if provided
    if proxy_hail_size is not None:
        missing_count = df[hail_col].isna().sum()
        if missing_count > 0:
            print(f"  Filling {missing_count} missing hail values with proxy value: {proxy_hail_size}")
            df[hail_col] = df[hail_col].fillna(proxy_hail_size)
    
    # track dropped
    lon_invalid = df[lon_col].isna().sum()
    lat_invalid = df[lat_col].isna().sum()
    hail_invalid = df[hail_col].isna().sum()
    
    if lon_invalid > 0 or lat_invalid > 0 or hail_invalid > 0:
        print(f"  Invalid values found:")
        print(f"    {lon_col}: {lon_invalid} invalid")
        print(f"    {lat_col}: {lat_invalid} invalid")
        print(f"    {hail_col}: {hail_invalid} invalid")
    
    df = df.dropna(subset=[lon_col, lat_col, hail_col])
    
    if len(df) == 0:
        error_msg = (
            f"No valid data points after cleaning. "
            f"Started with {initial_count} rows, all were removed due to invalid data.\n"
            f"Check that:\n"
            f"  1. Column '{lon_col}' contains valid longitude values (numbers)\n"
            f"  2. Column '{lat_col}' contains valid latitude values (numbers)\n"
            f"  3. Column '{hail_col}' contains valid hail size values (numbers)"
        )
        if proxy_hail_size is None:
            error_msg += f"\n  4. Consider setting a proxy hail size if '{hail_col}' has missing values"
        raise ValueError(error_msg)
    
    print(f"  Valid rows after cleaning: {len(df)} (removed {initial_count - len(df)})")
    
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )


def choose_utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def ensure_projected_meters(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, CRS, CRS]:
    crs_in = CRS.from_user_input(gdf.crs) if gdf.crs else CRS.from_epsg(4326)
    
    if gdf.crs is None:
        gdf = gdf.set_crs(crs_in)
    
    if crs_in.is_projected:
        unit = crs_in.axis_info[0].unit_name.lower()
        if unit in ("metre", "meter"):
            return gdf, crs_in, crs_in
    
    gdf_ll = gdf.to_crs(4326)
    centroid = gdf_ll.geometry.union_all().centroid
    crs_work = choose_utm_crs_from_lonlat(centroid.x, centroid.y)
    
    return gdf.to_crs(crs_work), crs_in, crs_work


def sample_polygon_boundary_points_vectorized(geom, spacing_m: float) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros((0, 2), dtype=np.float64)
    
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(g.geoms)
    else:
        polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    
    if not polys:
        return np.zeros((0, 2), dtype=np.float64)
    
    spacing = max(float(spacing_m), 1.0)
    all_coords = []
    
    for poly in polys:
        exterior = poly.exterior
        length = exterior.length
        if length <= 0:
            continue
        
        n = max(8, int(math.ceil(length / spacing)))
        distances = np.linspace(0.0, length, num=n, endpoint=False)
        coords = np.array([exterior.interpolate(d).coords[0] for d in distances])
        all_coords.append(coords)
    
    if not all_coords:
        return np.zeros((0, 2), dtype=np.float64)
    
    return np.vstack(all_coords).astype(np.float64)


def idw_knn_optimized(
    xy_known: np.ndarray,
    v_known: np.ndarray,
    xy_query: np.ndarray,
    k: int,
    power: float = 2.0,
    chunk_size: int = 50000) -> np.ndarray: 
    n_query = xy_query.shape[0]
    
    if n_query == 0:
        return np.array([], dtype=np.float64)
    
    # k doesn't exceed number of known points
    k = min(k, len(xy_known))
    if k < 1:
        k = 1
    
    tree = cKDTree(xy_known)
    
    if n_query <= chunk_size:
        return _idw_single_chunk(tree, v_known, xy_query, k, power)
    
    out = np.empty(n_query, dtype=np.float64)
    
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        out[start:end] = _idw_single_chunk(tree, v_known, xy_query[start:end], k, power)
    
    return out


def _idw_single_chunk(tree, v_known, xy_query, k, power):
    try:
        dists, idx = tree.query(xy_query, k=k, workers=-1)
    except TypeError:
        dists, idx = tree.query(xy_query, k=k)
    
    if k == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    
    vals = v_known[idx]
    
    if HAS_NUMBA and xy_query.shape[0] > 1000:
        return _idw_core_numba(dists, vals, power)
    
    return _idw_core_numpy(dists, vals, power)


def _idw_core_numpy(dists, vals, power):
    hit = dists == 0.0
    out = np.empty(dists.shape[0], dtype=np.float64)
    
    if np.any(hit):
        has_hit = hit.any(axis=1)
        first_hit = hit.argmax(axis=1)
        out[has_hit] = vals[has_hit, first_hit[has_hit]]
        
        nh = ~has_hit
        if np.any(nh):
            w = 1.0 / np.power(dists[nh], power)
            out[nh] = (w * vals[nh]).sum(axis=1) / w.sum(axis=1)
    else:
        w = 1.0 / np.power(dists, power)
        out = (w * vals).sum(axis=1) / w.sum(axis=1)
    
    return out


def heuristic_cell_size_m(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 500.0
    
    tree = cKDTree(points_xy)
    try:
        dists, _ = tree.query(points_xy, k=2, workers=-1)
    except TypeError:
        dists, _ = tree.query(points_xy, k=2)
    
    nn = dists[:, 1]
    valid_nn = nn[nn > 0]
    
    if valid_nn.size == 0:
        return 500.0
    
    med = float(np.median(valid_nn))
    if not np.isfinite(med) or med <= 0:
        return 500.0
    
    return float(np.clip(med / 3.0, 250.0, 2000.0))


def sanitize_value(val):
    """sanitize a single value for JSON serialization"""
    if val is None:
        return None
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return val


@dataclass
class Params:
    input_points: str
    hail_field: str
    event_name: str
    lon_col: str = "LONGITUDE"
    lat_col: str = "LATITUDE"
    qc_col: Optional[str] = None
    proxy_hail_size: Optional[float] = None
    
    grouping_threshold_km: float = 30.0
    large_buffer_km: float = 10.0
    small_buffer_km: float = 5.0
    
    boundary_spacing_small_m: float = 2200.0
    boundary_spacing_large_m: float = 3300.0
    
    idw_k_small: int = 2
    idw_k_large: int = 3
    idw_power: float = 2.0
    
    out_folder: str = "."
    job_id: str = ""


def get_processing_status(job_id: str) -> Dict:
    return {}


def run_footprint(
    params: Params,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, any]:
    """main processing function with progress callbacks"""
    
    def update_progress(pct: int, msg: str):
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[{pct}%] {msg}")
    
    t0 = time.time()
    
    os.makedirs(params.out_folder, exist_ok=True)
    
    # output filenames
    raster_filename = f"Ras_{params.event_name}.tif"
    view_raster_filename = f"Ras_{params.event_name}_HailSize.tif"
    poly_filename = f"Pg_{params.event_name}.gpkg"
    geojson_filename = f"Ras_{params.event_name}.geojson"
    footprint_geojson_filename = f"Footprint_{params.event_name}.geojson"
    points_geojson_filename = f"points_{params.event_name}.geojson"
    png_filename = f"Footprint_{params.event_name}_ClientMap.png"
    out_poly_layer = f"Pg_{params.event_name}"
    
    # Full paths
    out_raster = os.path.join(params.out_folder, raster_filename)
    view_raster = os.path.join(params.out_folder, view_raster_filename)
    out_poly = os.path.join(params.out_folder, poly_filename)
    out_geojson = os.path.join(params.out_folder, geojson_filename)
    footprint_geojson = os.path.join(params.out_folder, footprint_geojson_filename)
    points_geojson_path = os.path.join(params.out_folder, points_geojson_filename)
    out_png = os.path.join(params.out_folder, png_filename)
    
    # read inputs
    update_progress(5, "Reading input data...")
    
    if params.input_points.lower().endswith(".csv"):
        pts = read_points_csv(
            params.input_points,
            lon_col=params.lon_col,
            lat_col=params.lat_col,
            hail_col=params.hail_field,
            qc_col=params.qc_col,
            proxy_hail_size=params.proxy_hail_size
        )
    else:
        pts = read_vector(params.input_points)
    
    # WGS84 input for points export
    pts_4326_input = pts.to_crs(epsg=4326) if pts.crs and pts.crs.to_epsg() != 4326 else pts
    
    # project for processing
    pts_proj, crs_in, crs_work = ensure_projected_meters(pts)
    
    pts_proj[params.hail_field] = pd.to_numeric(pts_proj[params.hail_field], errors="coerce")
    pts_proj = pts_proj.dropna(subset=[params.hail_field]).copy()
    
    if len(pts_proj) == 0:
        raise ValueError("No valid points with hail data found")
    
    update_progress(10, f"Loaded {len(pts_proj)} valid points")
    
    # print coordinate ranges
    pts_debug = pts_proj.to_crs(epsg=4326)
    lon_range = (pts_debug.geometry.x.min(), pts_debug.geometry.x.max())
    lat_range = (pts_debug.geometry.y.min(), pts_debug.geometry.y.max())
    print(f"  Longitude range: {lon_range[0]:.4f} to {lon_range[1]:.4f}")
    print(f"  Latitude range: {lat_range[0]:.4f} to {lat_range[1]:.4f}")
    
    xy = np.column_stack([
        pts_proj.geometry.x.to_numpy(),
        pts_proj.geometry.y.to_numpy()
    ])
    
    # export points GeoJSON
    pts_export = pts_proj.to_crs(epsg=4326)
    pts_export.to_file(points_geojson_path, driver="GeoJSON")
    print(f"  Saved points GeoJSON: {points_geojson_path}")
    
    # DBSCAN clustering
    update_progress(15, "Clustering points...")
    
    eps_m = params.grouping_threshold_km * 1000.0
    db = DBSCAN(eps=eps_m, min_samples=2, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(xy)
    
    raw_group = np.where(
        labels != -1,
        (1000000 + labels).astype(np.int64),
        -(np.arange(len(labels)) + 1)
    )
    seq, _ = pd.factorize(raw_group, sort=True)
    pts_proj["Sequential_ID"] = (seq + 1).astype(np.int32)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    update_progress(20, f"Found {n_clusters} clusters")
    
    # prepare raster grid
    cell_size = heuristic_cell_size_m(xy)
    print(f"  Cell size: {cell_size:.1f} m")
    
    groups = pts_proj.groupby("Sequential_ID", sort=True)
    group_data_list = []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    
    for gid, g in groups:
        n = len(g)
        buf_m = (params.small_buffer_km if n == 1 else params.large_buffer_km) * 1000.0
        geom = unary_union(g.geometry.values)
        buf_poly = geom.buffer(buf_m)
        
        if buf_poly.is_empty:
            continue
        
        bx0, by0, bx1, by1 = buf_poly.bounds
        minx, miny = min(minx, bx0), min(miny, by0)
        maxx, maxy = max(maxx, bx1), max(maxy, by1)
        
        g_xy = np.column_stack([
            g.geometry.x.to_numpy(),
            g.geometry.y.to_numpy()
        ]).astype(np.float64)
        g_v = g[params.hail_field].to_numpy(dtype=np.float64)
        
        group_data_list.append((gid, g_xy, g_v, buf_poly, n))
    
    if not group_data_list:
        raise RuntimeError("No valid groups/buffers produced")
    
    minx -= cell_size
    miny -= cell_size
    maxx += cell_size
    maxy += cell_size
    
    width = int(math.ceil((maxx - minx) / cell_size))
    height = int(math.ceil((maxy - miny) / cell_size))
    
    print(f"  Raster size: {width} x {height}")
    
    transform = from_origin(minx, maxy, cell_size, cell_size)
    
    # process groups
    update_progress(25, f"Processing {len(group_data_list)} groups...")
    
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "int32",
        "crs": crs_work.to_wkt(),
        "transform": transform,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }
    
    raster_data = np.zeros((height, width), dtype=np.int32)
    
    total_groups = len(group_data_list)
    for i, group_data in enumerate(group_data_list):
        gid, g_xy, g_v, buf_poly, n = group_data
        
        progress_pct = 25 + int(50 * (i + 1) / total_groups)
        update_progress(progress_pct, f"Processing group {i+1}/{total_groups}")
        
        buf_spacing = params.boundary_spacing_small_m if n == 1 else params.boundary_spacing_large_m
        k = params.idw_k_small if n == 1 else params.idw_k_large
        
        boundary_xy = sample_polygon_boundary_points_vectorized(buf_poly, spacing_m=buf_spacing)
        boundary_v = np.zeros(boundary_xy.shape[0], dtype=np.float64)
        
        known_xy = np.vstack([g_xy, boundary_xy]) if boundary_xy.shape[0] > 0 else g_xy
        known_v = np.concatenate([g_v, boundary_v]) if boundary_xy.shape[0] > 0 else g_v
        
        bx0, by0, bx1, by1 = buf_poly.bounds
        col0 = max(0, int(math.floor((bx0 - minx) / cell_size)))
        col1 = min(width, int(math.ceil((bx1 - minx) / cell_size)))
        row0 = max(0, int(math.floor((maxy - by1) / cell_size)))
        row1 = min(height, int(math.ceil((maxy - by0) / cell_size)))
        
        if col1 <= col0 or row1 <= row0:
            continue
        
        win_w = col1 - col0
        win_h = row1 - row0
        
        win_transform = rasterio.transform.Affine(
            cell_size, 0.0, minx + col0 * cell_size,
            0.0, -cell_size, maxy - row0 * cell_size,
        )
        
        mask = rasterize(
            [(mapping(buf_poly), 1)],
            out_shape=(win_h, win_w),
            transform=win_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        
        if mask.max() == 0:
            continue
        
        rows, cols = np.where(mask == 1)
        xs = (minx + (col0 + cols + 0.5) * cell_size)
        ys = (maxy - (row0 + rows + 0.5) * cell_size)
        query_xy = np.column_stack([xs, ys]).astype(np.float64)
        
        idw_vals = idw_knn_optimized(
            xy_known=known_xy,
            v_known=known_v,
            xy_query=query_xy,
            k=k,
            power=params.idw_power,
        )
        
        scaled = np.clip(np.rint(idw_vals * 10000.0), 0, np.iinfo(np.int32).max).astype(np.int32)
        
        global_rows = row0 + rows
        global_cols = col0 + cols
        raster_data[global_rows, global_cols] = np.maximum(
            raster_data[global_rows, global_cols],
            scaled
        )
    
    # write raster
    update_progress(75, "Writing raster...")
    with rasterio.open(out_raster, "w", **profile) as dst:
        dst.write(raster_data, 1)
    
    # view raster
    with rasterio.open(out_raster) as src:
        a = src.read(1).astype(np.float32)
        a = np.where(a == 0, -9999.0, a / 10000.0)
        
        profile2 = src.profile.copy()
        profile2.update(dtype="float32", nodata=-9999.0)
        
        with rasterio.open(view_raster, "w", **profile2) as dst:
            dst.write(a, 1)
    
    # polygonize
    update_progress(80, "Polygonizing raster...")
    
    with rasterio.open(out_raster) as src:
        band = src.read(1)
        mask = band != 0
        
        geoms = []
        vals = []
        for geom, val in shapes(band, mask=mask, transform=src.transform):
            if val != 0:
                geoms.append(shape(geom))
                vals.append(int(val))
    
    if not geoms:
        raise RuntimeError("Polygonization produced no shapes")
    
        # create GeoDataFrame in the working CRS
    fpt = gpd.GeoDataFrame(
        {
            "gridcode": vals,
            "Hail_Size": np.array(vals, dtype=np.float64) / 10000.0
        },
        geometry=gpd.GeoSeries(geoms, crs=crs_work),
    )
    
    # convert to WGS84 for output
    fpt_4326 = fpt.to_crs(epsg=4326)
    fpt_out = fpt.to_crs(crs_in)
    
    # save outputs
    update_progress(85, "Saving outputs...")
    
    if os.path.exists(out_poly):
        os.remove(out_poly)
    fpt_out.to_file(out_poly, layer=out_poly_layer, driver="GPKG")
    
    # save GeoJSON in WGS84
    fpt_4326.to_file(out_geojson, driver="GeoJSON")
    print(f"  Saved footprint GeoJSON: {out_geojson}")
    
    # footprint (dissolved)
    min_hail_for_footprint = 0.1
    footprint = fpt_4326.loc[fpt_4326["Hail_Size"] >= min_hail_for_footprint].copy()
    
    png_created = False
    if not footprint.empty:
        update_progress(90, "Creating footprint...")
        footprint_dissolved = footprint.dissolve()
        
        footprint_layer = f"Footprint_{params.event_name}"
        footprint_dissolved.to_crs(crs_in).to_file(out_poly, layer=footprint_layer, driver="GPKG")
        footprint_dissolved.to_file(footprint_geojson, driver="GeoJSON")
        print(f"  Saved dissolved footprint GeoJSON: {footprint_geojson}")
        
        # png export
        try:
            export_client_png(
                fpt_out=fpt_out,
                footprint=footprint_dissolved.to_crs(crs_in),
                out_png=out_png,
                event_name=params.event_name,
            )
            png_created = True
            print(f"  Saved PNG: {out_png}")
        except Exception as e:
            print(f"Warning: PNG export failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        # create empty footprint file
        footprint_dissolved = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        footprint_dissolved.to_file(footprint_geojson, driver="GeoJSON")
    
    # calculate bounds in WGS84
    bounds_arr = fpt_4326.total_bounds  # [minx, miny, maxx, maxy]
    
    # validate bounds
    if any(not np.isfinite(b) for b in bounds_arr):
        print("Warning: Invalid bounds detected, using point extent")
        bounds_arr = pts_export.total_bounds
    
    bounds = [float(b) for b in bounds_arr]  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
    
    # center: [lat, lon] for Leaflet
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    center = [center_lat, center_lon]
    
    print(f"  Bounds (lon/lat): [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    print(f"  Center (lat/lon): [{center[0]:.4f}, {center[1]:.4f}]")
    
    # compute stats
    hail_min = float(fpt_4326["Hail_Size"].min())
    hail_max = float(fpt_4326["Hail_Size"].max())
    
    dt = time.time() - t0
    update_progress(100, f"Completed in {dt:.1f}s")
    
    # return filenames 
    return {
        "raster": view_raster_filename,
        "raw_raster": raster_filename,
        "polygons": poly_filename,
        "polygon_layer": out_poly_layer,
        "geojson": geojson_filename,
        "footprint_geojson": footprint_geojson_filename,
        "points_geojson": points_geojson_filename,
        "footprint_png": png_filename if png_created else "",
        "cell_size_m": float(cell_size),
        "n_groups": len(group_data_list),
        "n_points": len(pts_proj),
        "processing_seconds": round(dt, 2),
        "bounds": bounds,  
        "center": center, 
        "hail_min": sanitize_value(hail_min),
        "hail_max": sanitize_value(hail_max),
        "job_id": params.job_id,
    }


def export_client_png(
    fpt_out: gpd.GeoDataFrame,
    footprint: gpd.GeoDataFrame,
    out_png: str,
    event_name: str,
    cmap: str = "YlOrRd"
) -> None:
    """export png map"""
    fpt_3857 = fpt_out.to_crs(epsg=3857)
    foot_3857 = footprint.to_crs(epsg=3857)
    
    vals = fpt_3857["Hail_Size"].to_numpy()
    vals = vals[np.isfinite(vals)]
    vmin = float(np.nanmin(vals)) if vals.size else 0.0
    vmax = float(np.nanmax(vals)) if vals.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    fpt_3857.plot(
        ax=ax,
        column="Hail_Size",
        cmap=cmap,
        linewidth=0.0,
        alpha=0.55,
        vmin=vmin,
        vmax=vmax,
    )
    
    foot_3857.boundary.plot(ax=ax, color="black", linewidth=1.0)
    
    minx, miny, maxx, maxy = foot_3857.total_bounds
    pad_x = (maxx - minx) * 0.10 if maxx > minx else 5000
    pad_y = (maxy - miny) * 0.10 if maxy > miny else 5000
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    
    if HAS_CONTEXTILY:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")
    
    ax.set_title(f"Hail Footprint – {event_name}", fontsize=18, weight="bold", pad=14)
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colormaps.get_cmap(cmap))
    sm.set_array([])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Hail size (cm)", fontsize=10)
    
    if HAS_SCALEBAR:
        ax.add_artist(ScaleBar(dx=1, units="m", location="lower right", box_alpha=0.75))
    
    ax.annotate(
        "N",
        xy=(0.96, 0.94),
        xytext=(0.96, 0.86),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black"),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0),
    )
    
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)