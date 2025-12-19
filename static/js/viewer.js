// Inline-capable Leaflet viewer for a completed processing job.
// Usage:
//   window.HailViewer.init({ jobId, eventName, result })

(function () {
  function el(id) { return document.getElementById(id); }

  function safeNum(x, fallback) {
    const n = Number(x);
    return Number.isFinite(n) ? n : fallback;
  }

  function showInlineError(msg) {
    const container = el('errorContainer');
    if (!container) {
      window.HailUI?.toast?.(msg, { type: 'danger', title: 'Viewer error' });
      return;
    }
    container.innerHTML = `
      <div class="mt-4 rounded-2xl border border-rose-200/70 bg-rose-50 px-4 py-3 text-rose-900 dark:border-rose-900/50 dark:bg-rose-950/40 dark:text-rose-100">
        <div class="flex items-start gap-3">
          <div class="mt-0.5 text-lg"><i class="bi bi-x-circle"></i></div>
          <div class="min-w-0">
            <div class="text-sm font-semibold">Viewer error</div>
            <div class="text-sm opacity-90">${msg}</div>
          </div>
        </div>
      </div>
    `;
  }

  async function ensureDomToImage() {
    if (window.domtoimage) return true;
    return new Promise((resolve) => {
      const s = document.createElement('script');
      s.src = 'https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/2.6.0/dom-to-image.min.js';
      s.onload = () => resolve(true);
      s.onerror = () => resolve(false);
      document.head.appendChild(s);
    });
  }

  async function init(opts) {
    const { jobId, eventName, result } = opts || {};
    if (!jobId) throw new Error('Missing jobId');
    if (!window.L) throw new Error('Leaflet not loaded');

    const mapEl = el('map');
    const mapContainerEl = el('map-container');
    if (!mapEl || !mapContainerEl) throw new Error('Map container not found in DOM');

    // Re-init support (Change file): tear down any previous map instance
    const prev = mapEl._hfMap;
    if (prev && typeof prev.remove === 'function') {
      try { prev.remove(); } catch (e) {}
      mapEl._hfMap = null;
    }
    mapEl.dataset.hfInitialized = '1';

    // Parse result data
    const resultData = result || {};

    let bounds = Array.isArray(resultData.bounds) ? resultData.bounds : [-10, 40, 10, 60];
    if (!Array.isArray(bounds) || bounds.length !== 4 || bounds.some(b => b === null || isNaN(b))) {
      bounds = [-10, 40, 10, 60];
    }

    const hailMin = safeNum(resultData.hail_min, 0);
    const hailMax = safeNum(resultData.hail_max, 1);

    // Update legend
    const legendMin = el('legendMin');
    const legendMax = el('legendMax');
    if (legendMin) legendMin.textContent = (hailMin || 0).toFixed(1);
    if (legendMax) legendMax.textContent = (hailMax || 1).toFixed(1);

    // Panel toggle
    const controlPanel = el('controlPanel');
    const panelToggle = el('panelToggle');
    const panelHeader = el('panelHeader');
    const toggleIcon = el('toggleIcon');

    function togglePanel() {
      if (!controlPanel) return;
      controlPanel.classList.toggle('hf-collapsed');
      const collapsed = controlPanel.classList.contains('hf-collapsed');
      if (toggleIcon) toggleIcon.className = collapsed ? 'bi bi-chevron-left' : 'bi bi-chevron-right';
    }

    panelToggle?.addEventListener('click', (e) => {
      e.stopPropagation();
      togglePanel();
    });
    panelHeader?.addEventListener('click', togglePanel);

    // Default basemap based on theme
    const isDark = document.documentElement.classList.contains('dark');
    const basemapSelect = el('basemapSelect');
    if (basemapSelect && !basemapSelect.value) basemapSelect.value = isDark ? 'carto_dark' : 'carto_light';

    // =====================
    // Initialize Map with SVG renderer for consistent colors
    // =====================
    const center = Array.isArray(resultData.center) && resultData.center.length === 2
      ? resultData.center
      : [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2];

    // Use SVG renderer explicitly to prevent color issues on zoom
    const map = L.map('map', { 
      center, 
      zoom: 8, 
      zoomControl: true,
      renderer: L.svg({ padding: 0.5 })  // SVG renderer with padding to prevent clipping
    });
    mapEl._hfMap = map;

    const basemapConfigs = {
      osm: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      carto_light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      carto_positron_nolabels: 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
      carto_dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      carto_voyager: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      esri_gray: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      esri_darkgray: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      esri_worldstreet: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      esri_worldtopo: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      esri_worldimagery: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      none: ''
    };

    const basemaps = {};
    for (const [key, url] of Object.entries(basemapConfigs)) {
      basemaps[key] = url ? L.tileLayer(url, { attribution: '', maxZoom: 19 }) : null;
    }

    let currentBasemapId = basemapSelect?.value || (isDark ? 'carto_dark' : 'carto_light');
    let currentBasemap = basemaps[currentBasemapId];
    if (currentBasemap) currentBasemap.addTo(map);

    const leafletBounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]];
    try { map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 12 }); }
    catch (e) { map.setView(center, 8); }

    // Ensure proper sizing if we were revealed with animation
    setTimeout(() => map.invalidateSize(), 50);

    // =====================
    // Color Scale - Improved interpolation
    // =====================
    const colorStops = [
      { pos: 0.0, color: [255, 255, 178] },   // #ffffb2
      { pos: 0.17, color: [254, 217, 118] },  // #fed976
      { pos: 0.33, color: [254, 178, 76] },   // #feb24c
      { pos: 0.5, color: [253, 141, 60] },    // #fd8d3c
      { pos: 0.67, color: [252, 78, 42] },    // #fc4e2a
      { pos: 0.83, color: [227, 26, 28] },    // #e31a1c
      { pos: 1.0, color: [189, 0, 38] }       // #bd0026
    ];

    function getColor(value) {
      const range = (hailMax - hailMin) || 1;
      const normalized = Math.max(0, Math.min(1, (value - hailMin) / range));
      
      // Find the two color stops to interpolate between
      let lower = colorStops[0];
      let upper = colorStops[colorStops.length - 1];
      
      for (let i = 0; i < colorStops.length - 1; i++) {
        if (normalized >= colorStops[i].pos && normalized <= colorStops[i + 1].pos) {
          lower = colorStops[i];
          upper = colorStops[i + 1];
          break;
        }
      }
      
      // Interpolate between the two colors
      const range2 = upper.pos - lower.pos || 1;
      const t = (normalized - lower.pos) / range2;
      
      const r = Math.round(lower.color[0] + t * (upper.color[0] - lower.color[0]));
      const g = Math.round(lower.color[1] + t * (upper.color[1] - lower.color[1]));
      const b = Math.round(lower.color[2] + t * (upper.color[2] - lower.color[2]));
      
      return `rgb(${r},${g},${b})`;
    }

    let footprintLayer = null;
    let outlineLayer = null;
    let pointsLayer = null;
    let currentOpacity = 0.6;

    const opacityValue = el('opacityValue');
    const opacitySlider = el('opacitySlider');
    if (opacitySlider) opacitySlider.value = String(currentOpacity);
    if (opacityValue) opacityValue.textContent = currentOpacity.toFixed(1);

    // Store styles per feature to ensure consistency
    const featureStyles = new WeakMap();

    function getFeatureStyle(feature) {
      // Cache the color for each feature to prevent recalculation issues
      if (!featureStyles.has(feature)) {
        featureStyles.set(feature, {
          fillColor: getColor(feature.properties?.Hail_Size || 0)
        });
      }
      const cached = featureStyles.get(feature);
      return {
        fillColor: cached.fillColor,
        weight: 0.5,
        opacity: 0.8,
        color: '#666',
        fillOpacity: currentOpacity
      };
    }

    function footprintStyle(feature) {
      return getFeatureStyle(feature);
    }

    // Function to refresh all layer styles (call after opacity change or zoom)
    function refreshLayerStyles() {
      if (!footprintLayer) return;
      footprintLayer.eachLayer(layer => {
        if (layer.feature) {
          layer.setStyle(getFeatureStyle(layer.feature));
        }
      });
    }

    async function loadLayers() {
      try {
        // Footprint polygons
        const footprintRes = await fetch(`/geojson/${jobId}`);
        if (footprintRes.ok) {
          const data = await footprintRes.json();
          if (data.features?.length) {
            footprintLayer = L.geoJSON(data, {
              style: footprintStyle,
              onEachFeature: (feature, layer) => {
                // Pre-cache the color for this feature
                getFeatureStyle(feature);
                
                layer.on({
                  mouseover: () => {
                    const val = feature.properties?.Hail_Size || 0;
                    const hoverInfo = el('hoverInfo');
                    if (hoverInfo) hoverInfo.innerHTML = `<strong>${Number(val).toFixed(2)} cm</strong>`;
                    layer.setStyle({ weight: 2, color: '#fff' });
                  },
                  mouseout: () => {
                    // Restore full style including cached color
                    layer.setStyle(getFeatureStyle(feature));
                    const hoverInfo = el('hoverInfo');
                    if (hoverInfo) hoverInfo.textContent = 'Hover over map';
                  },
                  click: (e) => {
                    const val = feature.properties?.Hail_Size || 0;
                    L.popup().setLatLng(e.latlng).setContent(`<b>Hail Size:</b> ${Number(val).toFixed(2)} cm`).openOn(map);
                  }
                });
              }
            }).addTo(map);
          }
        }

        // Outline
        const outlineRes = await fetch(`/footprint_geojson/${jobId}`);
        if (outlineRes.ok) {
          const data = await outlineRes.json();
          if (data.features?.length) {
            outlineLayer = L.geoJSON(data, { style: { fill: false, weight: 2.5, color: '#000', opacity: 1 } }).addTo(map);
          }
        }

        // Points (hidden initially)
        const pointsRes = await fetch(`/points_geojson/${jobId}`);
        if (pointsRes.ok) {
          const data = await pointsRes.json();
          if (data.features?.length) {
            pointsLayer = L.geoJSON(data, {
              pointToLayer: (_f, latlng) => L.circleMarker(latlng, {
                radius: 6, fillColor: '#ff7800', color: '#000', weight: 1, opacity: 1, fillOpacity: 0.8
              }),
              onEachFeature: (feature, layer) => {
                let html = '<b>Point</b><br>';
                for (const [k, v] of Object.entries(feature.properties || {})) {
                  if (v !== null && v !== undefined) html += `${k}: ${v}<br>`;
                }
                layer.bindPopup(html);
              }
            });
          }
        }

        const loading = el('loading-overlay');
        if (loading) loading.style.display = 'none';
      } catch (err) {
        console.error('Load error:', err);
        const loading = el('loading-overlay');
        if (loading) loading.style.display = 'none';
        showInlineError('Failed to load map data.');
      }
    }

    await loadLayers();

    // Refresh styles on zoom to ensure colors persist
    map.on('zoomend', refreshLayerStyles);
    map.on('moveend', refreshLayerStyles);

    // =====================
    // Controls
    // =====================
    const showFootprint = el('showFootprint');
    const showOutline = el('showOutline');
    const showPoints = el('showPoints');

    showFootprint?.addEventListener('change', function () {
      if (!footprintLayer) return;
      if (this.checked) {
        footprintLayer.addTo(map);
        refreshLayerStyles();  // Refresh colors when re-adding layer
      } else {
        map.removeLayer(footprintLayer);
      }
    });
    showOutline?.addEventListener('change', function () {
      if (!outlineLayer) return;
      this.checked ? outlineLayer.addTo(map) : map.removeLayer(outlineLayer);
    });
    showPoints?.addEventListener('change', function () {
      if (!pointsLayer) return;
      if (this.checked) pointsLayer.addTo(map);
      else map.removeLayer(pointsLayer);
      el('togglePointsBtn')?.classList.toggle('hf-active', this.checked);
    });

    opacitySlider?.addEventListener('input', function () {
      currentOpacity = safeNum(this.value, 0.6);
      if (opacityValue) opacityValue.textContent = currentOpacity.toFixed(1);
      // Refresh all styles with new opacity while preserving colors
      refreshLayerStyles();
    });

    basemapSelect?.addEventListener('change', function () {
      currentBasemapId = this.value;
      if (currentBasemap) map.removeLayer(currentBasemap);
      currentBasemap = basemaps[currentBasemapId];
      if (currentBasemap) {
        currentBasemap.addTo(map);
        currentBasemap.bringToBack();
      }
    });

    el('resetViewBtn')?.addEventListener('click', () => {
      try { map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 12 }); }
      catch (e) { /* noop */ }
    });

    el('toggleLayersBtn')?.addEventListener('click', function () {
      if (!showFootprint || !showOutline) return;
      const allVisible = showFootprint.checked && showOutline.checked;
      showFootprint.checked = !allVisible;
      showOutline.checked = !allVisible;
      showFootprint.dispatchEvent(new Event('change'));
      showOutline.dispatchEvent(new Event('change'));
      this.classList.toggle('hf-active', !allVisible);
    });

    el('togglePointsBtn')?.addEventListener('click', function () {
      if (!showPoints) return;
      showPoints.checked = !showPoints.checked;
      showPoints.dispatchEvent(new Event('change'));
    });

    map.on('mousemove', (e) => {
      const c = el('coordsDisplay');
      if (c) c.textContent = `${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`;
    });

    L.control.scale({ metric: true, imperial: false }).addTo(map);

    // =====================
    // Export
    // =====================
    function getCurrentBounds() {
      const b = map.getBounds();
      return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    }

    let isExporting = false;

    async function exportServerRendered() {
      if (isExporting) {
        window.HailUI?.toast?.('Export already in progress…', { type: 'warning', title: 'Please wait' });
        return;
      }

      const exportBtn = el('exportCurrentViewBtn');
      const originalContent = exportBtn?.innerHTML;
      
      try {
        isExporting = true;
        
        // Update button to show loading state
        if (exportBtn) {
          exportBtn.disabled = true;
          exportBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin"></i> Generating…';
        }

        window.HailUI?.toast?.('Generating high-quality map…', { type: 'info', title: 'Export' });
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

        const res = await fetch(`/render_map/${jobId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            bounds: getCurrentBounds(),
            basemap: currentBasemapId,
            show_footprint: !!showFootprint?.checked,
            show_outline: !!showOutline?.checked,
            show_points: !!showPoints?.checked,
            opacity: currentOpacity,
            width_px: 3200,
            height_px: 2000
          }),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
          let err = 'Export failed';
          try { 
            const errData = await res.json();
            err = errData.error || err; 
          } catch (e) {
            try { err = await res.text(); } catch (e2) {}
          }
          throw new Error(err);
        }

        const ct = (res.headers.get('content-type') || '').toLowerCase();
        if (!ct.includes('image/png')) {
          let body = '';
          try { body = await res.text(); } catch (e) {}
          const preview = body ? body.slice(0, 300) : '';
          throw new Error(`Export did not return a PNG (content-type: ${ct || 'unknown'}). ${preview ? `Response: ${preview}` : ''}`.trim());
        }

        const blob = await res.blob();
        
        if (blob.size < 1000) {
          throw new Error('Generated image is too small - export may have failed');
        }

        // Use a more reliable download method
        const url = URL.createObjectURL(blob);
        const filename = `${eventName || 'hail_footprint'}_map.png`;
        
        // Try using the download attribute first
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        
        // Use a slight delay to ensure the link is in the DOM
        await new Promise(resolve => setTimeout(resolve, 100));
        a.click();
        
        // Clean up after a delay to ensure download starts
        setTimeout(() => {
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }, 1000);

        window.HailUI?.toast?.('Map exported successfully!', { type: 'success', title: 'Export Complete' });

      } catch (e) {
        console.error('Export error:', e);
        if (e.name === 'AbortError') {
          showInlineError('Export timed out. Try with a smaller view area.');
        } else {
          showInlineError(e.message || 'Export failed. Please try again.');
        }
        window.HailUI?.toast?.(e.message || 'Export failed', { type: 'danger', title: 'Export Error' });
      } finally {
        isExporting = false;
        if (exportBtn) {
          exportBtn.disabled = false;
          exportBtn.innerHTML = originalContent || '<i class="bi bi-camera"></i> HQ export';
        }
      }
    }

    async function exportScreenshot() {
      const ok = await ensureDomToImage();
      if (!ok) throw new Error('Screenshot export unavailable (dom-to-image failed to load)');
      window.HailUI?.toast?.('Capturing screenshot…', { type: 'info', title: 'Export' });
      
      try {
        const dataUrl = await window.domtoimage.toPng(mapEl, { 
          quality: 1, 
          bgcolor: '#fff',
          style: {
            'transform': 'none'  // Prevent transform issues
          }
        });
        const a = document.createElement('a');
        a.href = dataUrl;
        a.download = `${eventName || 'hail_footprint'}_screenshot.png`;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => document.body.removeChild(a), 100);
        window.HailUI?.toast?.('Screenshot saved!', { type: 'success', title: 'Done' });
      } catch (e) {
        throw new Error('Screenshot capture failed: ' + e.message);
      }
    }

    el('exportCurrentViewBtn')?.addEventListener('click', async () => {
      try { await exportServerRendered(); }
      catch (e) { showInlineError(e.message); }
    });

    el('exportScreenshotBtn')?.addEventListener('click', async () => {
      try { await exportScreenshot(); }
      catch (e) { showInlineError(e.message); }
    });
  }

  window.HailViewer = { init };
})();