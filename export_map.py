
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import io

def generate_map_png(output_folder, geojson_file, footprint_file, points_file, bounds, basemap_id,
    show_footprint, show_outline, show_points, opacity, event_name, hail_min, hail_max,
    width_px=3200, height_px=2000, outline_width=1, outline_color='#000000',
    display_mode='cells', color_map='ylOrRd', point_radius=3, point_color='#ff7800'):
    """Generate a PNG map with specified settings - FIXED aspect ratio handling"""

    
    try:
        import contextily as ctx
        HAS_CTX = True
    except ImportError:
        HAS_CTX = False
    
    print(f"[MapGen] ========================================")
    print(f"[MapGen] Basemap: '{basemap_id}', Bounds: {bounds}")
    print(f"[MapGen] Target size: {width_px}x{height_px}px")
    
    # load data
    footprint_gdf = None
    outline_gdf = None
    points_gdf = None
    
    if show_footprint and geojson_file:
        geojson_path = output_folder / geojson_file
        if geojson_path.exists():
            footprint_gdf = gpd.read_file(geojson_path)
            print(f"[MapGen] Loaded footprint: {len(footprint_gdf)} features")
    
    if show_outline and footprint_file:
        outline_path = output_folder / footprint_file
        if outline_path.exists():
            outline_gdf = gpd.read_file(outline_path)
            print(f"[MapGen] Loaded outline: {len(outline_gdf)} features")
    
    if show_points and points_file:
        points_path = output_folder / points_file
        if points_path.exists():
            points_gdf = gpd.read_file(points_path)
            print(f"[MapGen] Loaded points: {len(points_gdf)} features")
    
    # determine extent (WGS84)
    bounds_from_viewer = bounds and len(bounds) == 4 and all(b is not None for b in bounds)
    
    if bounds_from_viewer:
        extent_minx, extent_miny, extent_maxx, extent_maxy = bounds
        print(f"[MapGen] Using viewer bounds (WGS84): [{extent_minx:.4f}, {extent_miny:.4f}, {extent_maxx:.4f}, {extent_maxy:.4f}]")
    elif footprint_gdf is not None and not footprint_gdf.empty:
        extent_minx, extent_miny, extent_maxx, extent_maxy = footprint_gdf.total_bounds
        #  padding when using data bounds
        pad_x = (extent_maxx - extent_minx) * 0.05
        pad_y = (extent_maxy - extent_miny) * 0.05
        extent_minx -= pad_x
        extent_maxx += pad_x
        extent_miny -= pad_y
        extent_maxy += pad_y
    else:
        raise ValueError("No bounds provided and no footprint data available")
    
    # validate bounds
    if extent_minx >= extent_maxx or extent_miny >= extent_maxy:
        raise ValueError(f"Invalid bounds: [{extent_minx}, {extent_miny}, {extent_maxx}, {extent_maxy}]")
    
    # convert to web mercator (EPSG:3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    ext_x1, ext_y1 = transformer.transform(extent_minx, extent_miny)
    ext_x2, ext_y2 = transformer.transform(extent_maxx, extent_maxy)
    
    print(f"[MapGen] Initial 3857 extent: [{ext_x1:.0f}, {ext_y1:.0f}, {ext_x2:.0f}, {ext_y2:.0f}]")
    
    # fix aspect ratio - expand bounds to match target image ratio
    bounds_width_m = abs(ext_x2 - ext_x1)
    bounds_height_m = abs(ext_y2 - ext_y1)
    
    if bounds_width_m == 0 or bounds_height_m == 0:
        raise ValueError("Invalid bounds: zero width or height")
    
    bounds_aspect = bounds_width_m / bounds_height_m  # W/H ratio of geographic bounds
    target_aspect = float(width_px) / float(height_px)  # W/H ratio of output image
    
    print(f"[MapGen] Bounds aspect: {bounds_aspect:.3f}, Target aspect: {target_aspect:.3f}")
    
    # adjust bounds to match target aspect ratio
    if abs(bounds_aspect - target_aspect) > 0.001:
        center_x = (ext_x1 + ext_x2) / 2
        center_y = (ext_y1 + ext_y2) / 2
        
        if bounds_aspect > target_aspect:
            # Bounds are WIDER than target - need to expand HEIGHT
            new_height = bounds_width_m / target_aspect
            ext_y1 = center_y - new_height / 2
            ext_y2 = center_y + new_height / 2
            print(f"[MapGen] Expanded height: {bounds_height_m:.0f} -> {new_height:.0f}")
        else:
            # Bounds are TALLER than target - need to expand WIDTH
            new_width = bounds_height_m * target_aspect
            ext_x1 = center_x - new_width / 2
            ext_x2 = center_x + new_width / 2
            print(f"[MapGen] Expanded width: {bounds_width_m:.0f} -> {new_width:.0f}")
    
    print(f"[MapGen] Adjusted 3857 extent: [{ext_x1:.0f}, {ext_y1:.0f}, {ext_x2:.0f}, {ext_y2:.0f}]")
    
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
    
    # create figure with correct dimensions
    dpi = 150
    fig_w_in = float(width_px) / dpi
    fig_h_in = float(height_px) / dpi
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(ext_x1, ext_x2)
    ax.set_ylim(ext_y1, ext_y2)
    ax.set_aspect('auto')
    
    # colormap setup    
    COLORMAP_MAPPING = {
        'ylOrRd': 'YlOrRd', 'orRd': 'OrRd', 'reds': 'Reds',
        'ylGnBu': 'YlGnBu', 'blues': 'Blues', 'greens': 'Greens',
        'purples': 'Purples', 'greys': 'Greys', 'viridis': 'viridis',
        'plasma': 'plasma', 'inferno': 'inferno', 'magma': 'magma',
        'cividis': 'cividis', 'turbo': 'turbo', 'rdYlGn': 'RdYlGn',
        'spectral': 'Spectral', 'coolwarm': 'coolwarm', 'hot': 'hot', 'jet': 'jet',
    }
    
    mpl_cmap_name = COLORMAP_MAPPING.get(color_map, 'YlOrRd')
    cmap = plt.cm.get_cmap(mpl_cmap_name)
    
    vmin = hail_min if hail_min is not None else 0
    vmax = hail_max if hail_max is not None else 1
    if vmax <= vmin:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    print(f"[MapGen] Colormap: {mpl_cmap_name}, Display: {display_mode}")
    
    # plot footprint
    if show_footprint and footprint_3857 is not None and not footprint_3857.empty:
        if 'Hail_Size' in footprint_3857.columns:
            if display_mode == 'continuous':
                try:
                    from scipy.ndimage import gaussian_filter
                    from rasterio.features import rasterize
                    from rasterio.transform import from_bounds
                    
                    #  use adjusted extent for rasterization
                    rast_minx, rast_miny, rast_maxx, rast_maxy = ext_x1, ext_y1, ext_x2, ext_y2
                    rast_width = max(400, int(width_px * 0.5))
                    rast_height = max(300, int(height_px * 0.5))
                    
                    #  calculate sigma based on cell size
                    avg_cell_width = 0
                    if not footprint_3857.empty:
                        sample = footprint_3857.geometry.iloc[:20]
                        total_area = sum(g.area for g in sample)
                        avg_area = total_area / len(sample)
                        avg_cell_width = np.sqrt(avg_area)
                    
                    map_width_m = abs(rast_maxx - rast_minx)
                    m_per_px = map_width_m / rast_width
                    cell_size_px = avg_cell_width / m_per_px if m_per_px > 0 else 10
                    sigma = max(2.0, cell_size_px * 0.6)
                    
                    transform = from_bounds(rast_minx, rast_miny, rast_maxx, rast_maxy, 
                                           rast_width, rast_height)
                    
                    shapes_val = [(geom, val) for geom, val in 
                                  zip(footprint_3857.geometry, footprint_3857['Hail_Size'])]
                    raster_val = rasterize(shapes_val, out_shape=(rast_height, rast_width), 
                                           transform=transform, fill=0, dtype='float32')
                    
                    shapes_weight = [(geom, 1.0) for geom in footprint_3857.geometry]
                    raster_weight = rasterize(shapes_weight, out_shape=(rast_height, rast_width),
                                              transform=transform, fill=0, dtype='float32')
                    
                    smoothed_val = gaussian_filter(raster_val, sigma=sigma, mode='constant', cval=0)
                    smoothed_weight = gaussian_filter(raster_weight, sigma=sigma, mode='constant', cval=0)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        normalized = smoothed_val / smoothed_weight
                        normalized[smoothed_weight < 1e-6] = 0
                    
                    alpha_map = smoothed_weight
                    max_weight = np.max(smoothed_weight)
                    if max_weight > 0:
                        alpha_map = alpha_map / max_weight
                    
                    norm_data = norm(normalized)
                    rgba_img = cmap(norm_data)
                    rgba_img[..., 3] = alpha_map * opacity
                    
                    # Plot with EXACT adjusted extent
                    ax.imshow(rgba_img, 
                              extent=[rast_minx, rast_maxx, rast_miny, rast_maxy], 
                              origin='upper', zorder=2, interpolation='bicubic', 
                              aspect='auto')
                    
                    print(f"[MapGen] Plotted continuous mode")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[MapGen] Continuous mode failed: {e}")
                    # Fallback
                    footprint_3857.plot(ax=ax, column='Hail_Size', cmap=cmap, norm=norm,
                                        alpha=opacity, linewidth=0, edgecolor='none', zorder=2)
                    ax.set_aspect('auto')  # Reset after geopandas
            else:
                # CELLS MODE
                footprint_3857.plot(ax=ax, column='Hail_Size', cmap=cmap, norm=norm,
                                    alpha=opacity, linewidth=0.1, edgecolor='#666666', zorder=2)
                ax.set_aspect('auto')  # CRITICAL: Reset after geopandas plotting
                print("[MapGen] Plotted cells mode")
        else:
            footprint_3857.plot(ax=ax, color='#fd8d3c', alpha=opacity, zorder=2)
            ax.set_aspect('auto')
    
    # plot outline
    if show_outline and outline_3857 is not None and not outline_3857.empty:
        outline_3857.boundary.plot(ax=ax, color=outline_color, linewidth=outline_width, zorder=5)
        ax.set_aspect('auto')
        print(f"[MapGen] Plotted outline")
    
    # plot points
    if show_points and points_3857 is not None and not points_3857.empty:
        marker_size = (point_radius ** 2) * 1.5
        points_3857.plot(ax=ax, color=point_color, markersize=marker_size,
                         edgecolor='black', linewidth=0.8, zorder=4)
        ax.set_aspect('auto')
        print(f"[MapGen] Plotted points")
    
    # ensure correct extent (after all plotting)
    ax.set_xlim(ext_x1, ext_x2)
    ax.set_ylim(ext_y1, ext_y2)
    ax.set_aspect('auto')  # final override
    
    # add basemap
    basemap_added = False
    
    if HAS_CTX and basemap_id and basemap_id != 'none':
        import math
        import requests
        
        # Disable SSL verification
        _original_get = requests.get
        def _patched_get(url, **kwargs):
            kwargs['verify'] = False
            return _original_get(url, **kwargs)
        requests.get = _patched_get
        
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        # Calculate zoom
        center_lat_3857 = (ext_y1 + ext_y2) / 2
        # Convert center back to lat for zoom calculation
        rev_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        _, center_lat = rev_transformer.transform((ext_x1 + ext_x2) / 2, center_lat_3857)
        
        width_m = abs(ext_x2 - ext_x1)
        height_m = abs(ext_y2 - ext_y1)
        m_per_px = max(width_m / width_px, height_m / height_px)
        initial_res = 156543.03392804097
        adj = max(0.15, math.cos(math.radians(center_lat)))
        zoom_f = math.log2((initial_res * adj) / max(1e-9, m_per_px))
        zoom = int(round(zoom_f))
        zoom = max(1, min(zoom, 18))
        
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
        
        try:
            ctx.add_basemap(ax, source=url, zoom=zoom, zorder=1, 
                           attribution='', reset_extent=False)
            basemap_added = True
            print(f"[MapGen] Added basemap via contextily")
        except Exception as e:
            print(f"[MapGen] Contextily failed: {e}")
        
        # Fallback to manual tiles
        if not basemap_added:
            try:
                basemap_added = fetch_tiles_manually(ax, ext_x1, ext_y1, ext_x2, ext_y2, zoom, url)
            except Exception as e:
                print(f"[MapGen] Manual tiles failed: {e}")
    
    if basemap_id == 'none' or not basemap_added:
        ax.set_facecolor('#f5f5f5')
    
    # final extent enforcement (after basemap)
    ax.set_xlim(ext_x1, ext_x2)
    ax.set_ylim(ext_y1, ext_y2)
    ax.set_aspect('auto')
    ax.set_axis_off()
    
    # add map elements (title, colorbar, scale)
    ax.set_position([0.02, 0.02, 0.90, 0.93]) 
    
    # Title
    ax.set_title(f"Hail Footprint – {event_name}", fontsize=14, fontweight='bold', pad=12)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.15, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Hail Size (cm)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    
    # Scale bar
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(1, units='m', location='lower right', 
                               box_alpha=0.7, font_properties={'size': 12}))
    except ImportError:
        pass
    
    # North arrow
    ax.annotate('N', xy=(0.97, 0.95), xytext=(0.97, 0.88), xycoords='axes fraction',
                ha='center', va='center', fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, linewidth=0))
    
    # save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor='white', 
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    
    result = buf.read()
    print(f"[MapGen] Complete. Size: {len(result)} bytes")
    
    if len(result) < 1000:
        raise ValueError("Generated image too small")
    
    return result