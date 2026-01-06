// Inline-capable Leaflet viewer for a completed processing job.
(function () {
  "use strict";

  function el(id) {
    return document.getElementById(id);
  }

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
      <div class="mt-4 rounded-2xl border border-rose-200/70 bg-rose-50 px-4 py-3 text-rose-900">
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

  // ═══════════════════════════════════════════════════════════════════════════
  // COLOR MAPS DEFINITIONS
  // ═══════════════════════════════════════════════════════════════════════════
  const COLOR_MAPS = {
    // Sequential colormaps
    ylOrRd: {
      name: 'Yellow-Orange-Red',
      stops: [
        { pos: 0.0, color: [255, 255, 178] },
        { pos: 0.17, color: [254, 217, 118] },
        { pos: 0.33, color: [254, 178, 76] },
        { pos: 0.5, color: [253, 141, 60] },
        { pos: 0.67, color: [252, 78, 42] },
        { pos: 0.83, color: [227, 26, 28] },
        { pos: 1.0, color: [189, 0, 38] }
      ]
    },
    orRd: {
      name: 'Orange-Red',
      stops: [
        { pos: 0.0, color: [255, 247, 236] },
        { pos: 0.17, color: [254, 232, 200] },
        { pos: 0.33, color: [253, 212, 158] },
        { pos: 0.5, color: [253, 187, 132] },
        { pos: 0.67, color: [252, 141, 89] },
        { pos: 0.83, color: [239, 101, 72] },
        { pos: 1.0, color: [179, 0, 0] }
      ]
    },
    reds: {
      name: 'Reds',
      stops: [
        { pos: 0.0, color: [255, 245, 240] },
        { pos: 0.17, color: [254, 224, 210] },
        { pos: 0.33, color: [252, 187, 161] },
        { pos: 0.5, color: [252, 146, 114] },
        { pos: 0.67, color: [251, 106, 74] },
        { pos: 0.83, color: [222, 45, 38] },
        { pos: 1.0, color: [165, 15, 21] }
      ]
    },
    ylGnBu: {
      name: 'Yellow-Green-Blue',
      stops: [
        { pos: 0.0, color: [255, 255, 217] },
        { pos: 0.17, color: [237, 248, 177] },
        { pos: 0.33, color: [199, 233, 180] },
        { pos: 0.5, color: [127, 205, 187] },
        { pos: 0.67, color: [65, 182, 196] },
        { pos: 0.83, color: [29, 145, 192] },
        { pos: 1.0, color: [8, 29, 88] }
      ]
    },
    blues: {
      name: 'Blues',
      stops: [
        { pos: 0.0, color: [247, 251, 255] },
        { pos: 0.17, color: [222, 235, 247] },
        { pos: 0.33, color: [198, 219, 239] },
        { pos: 0.5, color: [158, 202, 225] },
        { pos: 0.67, color: [107, 174, 214] },
        { pos: 0.83, color: [49, 130, 189] },
        { pos: 1.0, color: [8, 81, 156] }
      ]
    },
    greens: {
      name: 'Greens',
      stops: [
        { pos: 0.0, color: [247, 252, 245] },
        { pos: 0.17, color: [229, 245, 224] },
        { pos: 0.33, color: [199, 233, 192] },
        { pos: 0.5, color: [161, 217, 155] },
        { pos: 0.67, color: [116, 196, 118] },
        { pos: 0.83, color: [49, 163, 84] },
        { pos: 1.0, color: [0, 109, 44] }
      ]
    },
    purples: {
      name: 'Purples',
      stops: [
        { pos: 0.0, color: [252, 251, 253] },
        { pos: 0.17, color: [239, 237, 245] },
        { pos: 0.33, color: [218, 218, 235] },
        { pos: 0.5, color: [188, 189, 220] },
        { pos: 0.67, color: [158, 154, 200] },
        { pos: 0.83, color: [128, 125, 186] },
        { pos: 1.0, color: [84, 39, 143] }
      ]
    },
    greys: {
      name: 'Greys',
      stops: [
        { pos: 0.0, color: [255, 255, 255] },
        { pos: 0.17, color: [240, 240, 240] },
        { pos: 0.33, color: [217, 217, 217] },
        { pos: 0.5, color: [189, 189, 189] },
        { pos: 0.67, color: [150, 150, 150] },
        { pos: 0.83, color: [99, 99, 99] },
        { pos: 1.0, color: [37, 37, 37] }
      ]
    },
    // Perceptually uniform colormaps
    viridis: {
      name: 'Viridis',
      stops: [
        { pos: 0.0, color: [68, 1, 84] },
        { pos: 0.17, color: [72, 40, 120] },
        { pos: 0.33, color: [62, 74, 137] },
        { pos: 0.5, color: [49, 104, 142] },
        { pos: 0.67, color: [38, 130, 142] },
        { pos: 0.83, color: [31, 158, 137] },
        { pos: 1.0, color: [253, 231, 37] }
      ]
    },
    plasma: {
      name: 'Plasma',
      stops: [
        { pos: 0.0, color: [13, 8, 135] },
        { pos: 0.17, color: [126, 3, 168] },
        { pos: 0.33, color: [204, 71, 120] },
        { pos: 0.5, color: [248, 149, 64] },
        { pos: 0.67, color: [255, 204, 92] },
        { pos: 0.83, color: [252, 255, 164] },
        { pos: 1.0, color: [240, 249, 33] }
      ]
    },
    inferno: {
      name: 'Inferno',
      stops: [
        { pos: 0.0, color: [0, 0, 4] },
        { pos: 0.17, color: [40, 11, 84] },
        { pos: 0.33, color: [101, 21, 110] },
        { pos: 0.5, color: [159, 42, 99] },
        { pos: 0.67, color: [212, 72, 66] },
        { pos: 0.83, color: [245, 125, 21] },
        { pos: 1.0, color: [252, 255, 164] }
      ]
    },
    magma: {
      name: 'Magma',
      stops: [
        { pos: 0.0, color: [0, 0, 4] },
        { pos: 0.17, color: [28, 16, 68] },
        { pos: 0.33, color: [79, 18, 123] },
        { pos: 0.5, color: [129, 37, 129] },
        { pos: 0.67, color: [181, 54, 122] },
        { pos: 0.83, color: [229, 80, 100] },
        { pos: 1.0, color: [252, 253, 191] }
      ]
    },
    cividis: {
      name: 'Cividis',
      stops: [
        { pos: 0.0, color: [0, 32, 77] },
        { pos: 0.17, color: [59, 60, 98] },
        { pos: 0.33, color: [94, 88, 111] },
        { pos: 0.5, color: [128, 118, 119] },
        { pos: 0.67, color: [163, 150, 119] },
        { pos: 0.83, color: [202, 185, 109] },
        { pos: 1.0, color: [253, 231, 37] }
      ]
    },
    turbo: {
      name: 'Turbo',
      stops: [
        { pos: 0.0, color: [48, 18, 59] },
        { pos: 0.17, color: [65, 68, 214] },
        { pos: 0.33, color: [35, 139, 251] },
        { pos: 0.5, color: [18, 203, 163] },
        { pos: 0.67, color: [139, 229, 63] },
        { pos: 0.83, color: [243, 186, 47] },
        { pos: 1.0, color: [122, 4, 3] }
      ]
    },
    // Legacy/Extra colormaps
    rdYlGn: {
      name: 'Red-Yellow-Green',
      stops: [
        { pos: 0.0, color: [0, 104, 55] },
        { pos: 0.17, color: [26, 152, 80] },
        { pos: 0.33, color: [145, 207, 96] },
        { pos: 0.5, color: [255, 255, 191] },
        { pos: 0.67, color: [252, 141, 89] },
        { pos: 0.83, color: [215, 48, 39] },
        { pos: 1.0, color: [165, 0, 38] }
      ]
    },
    spectral: {
      name: 'Spectral',
      stops: [
        { pos: 0.0, color: [94, 79, 162] },
        { pos: 0.17, color: [50, 136, 189] },
        { pos: 0.33, color: [102, 194, 165] },
        { pos: 0.5, color: [254, 254, 189] },
        { pos: 0.67, color: [254, 224, 139] },
        { pos: 0.83, color: [252, 141, 89] },
        { pos: 1.0, color: [213, 62, 79] }
      ]
    },
    coolwarm: {
      name: 'Cool-Warm',
      stops: [
        { pos: 0.0, color: [59, 76, 192] },
        { pos: 0.17, color: [98, 130, 234] },
        { pos: 0.33, color: [141, 176, 254] },
        { pos: 0.5, color: [221, 221, 221] },
        { pos: 0.67, color: [245, 152, 130] },
        { pos: 0.83, color: [219, 96, 76] },
        { pos: 1.0, color: [180, 4, 38] }
      ]
    },
    hot: {
      name: 'Hot',
      stops: [
        { pos: 0.0, color: [10, 0, 0] },
        { pos: 0.17, color: [128, 0, 0] },
        { pos: 0.33, color: [230, 0, 0] },
        { pos: 0.5, color: [255, 128, 0] },
        { pos: 0.67, color: [255, 230, 0] },
        { pos: 0.83, color: [255, 255, 128] },
        { pos: 1.0, color: [255, 255, 255] }
      ]
    },
    jet: {
      name: 'Jet',
      stops: [
        { pos: 0.0, color: [0, 0, 127] },
        { pos: 0.17, color: [0, 0, 255] },
        { pos: 0.33, color: [0, 255, 255] },
        { pos: 0.5, color: [128, 255, 0] },
        { pos: 0.67, color: [255, 255, 0] },
        { pos: 0.83, color: [255, 128, 0] },
        { pos: 1.0, color: [127, 0, 0] }
      ]
    }
  };





  function updateColorMapPreview(colorMapId) {
    const preview = document.getElementById('colorMapPreview');
    if (!preview) return;

    const cmap = COLOR_MAPS[colorMapId];
    if (!cmap) return;

    const gradientStops = cmap.stops.map(s =>
      `rgb(${s.color[0]},${s.color[1]},${s.color[2]}) ${s.pos * 100}%`
    ).join(', ');

    preview.style.background = `linear-gradient(to right, ${gradientStops})`;
  }

  async function init(opts) {
    const { jobId, eventName, result } = opts || {};
    if (!jobId) throw new Error('Missing jobId');
    if (!window.L) throw new Error('Leaflet not loaded');

    const mapEl = el('map');
    const mapContainerEl = el('map-container');
    if (!mapEl || !mapContainerEl) throw new Error('Map container not found in DOM');

    // Clean up any existing map instance before creating a new one
    if (mapEl._hfMap) {
      try {
        mapEl._hfMap.off();
        mapEl._hfMap.remove();
      } catch (e) {
        console.warn('[Viewer] Error cleaning up previous map:', e);
      }
      mapEl._hfMap = null;
    }

    // Ensure loading overlay is visible at the start
    const loadingOverlay = el('loading-overlay');
    if (loadingOverlay) loadingOverlay.style.display = 'flex';

    // Initialize color map preview
    updateColorMapPreview('ylOrRd');

    // ═══════════════════════════════════════════════════════════════════════════
    // UI ELEMENTS
    // ═══════════════════════════════════════════════════════════════════════════
    const showFootprint = el('showFootprint');
    const displayModeSelect = el('displayMode');
    const cellOptions = el('cellOptions');
    const continuousOptions = el('continuousOptions');
    const showCellBorders = el('showCellBorders');
    const cellBorderOptions = el('cellBorderOptions');
    const footprintOptions = el('footprintOptions');
    const showOutline = el('showOutline');
    const showPoints = el('showPoints');
    const outlineOptions = el('outlineOptions');
    const opacitySlider = el('opacitySlider');
    const opacityValue = el('opacityValue');
    const basemapSelect = el('basemapSelect');
    const legendMin = el('legendMin');
    const legendMax = el('legendMax');
    const controlPanel = el('controlPanel');
    const panelToggle = el('panelToggle');
    const panelHeader = el('panelHeader');
    const toggleIcon = el('toggleIcon');
    const smoothnessLevel = el('smoothnessLevel');

    // ═══════════════════════════════════════════════════════════════════════════
    // OPTIONS VISIBILITY
    // ═══════════════════════════════════════════════════════════════════════════
    function updateFootprintOptionsVisibility() {
      const fpBtn = document.getElementById('showFootprint');
      const fpOpts = document.getElementById('footprintOptions');
      if (fpOpts && fpBtn) {
        fpOpts.style.display = fpBtn.checked ? 'block' : 'none';
      }
    }

    function updateDisplayModeOptions() {
      const dMode = document.getElementById('displayMode');
      const mode = dMode?.value || 'cells';
      const cellOpts = document.getElementById('cellOptions');
      const contOpts = document.getElementById('continuousOptions');
      if (cellOpts) cellOpts.style.display = mode === 'cells' ? 'block' : 'none';
      if (contOpts) contOpts.style.display = mode === 'continuous' ? 'block' : 'none';
    }

    function updateCellBorderOptionsVisibility() {
      const cbBtn = document.getElementById('showCellBorders');
      const cbOpts = document.getElementById('cellBorderOptions');
      if (cbOpts && cbBtn) {
        cbOpts.style.display = cbBtn.checked ? 'block' : 'none';
      }
    }

    function updateOutlineOptionsVisibility() {
      const olBtn = document.getElementById('showOutline');
      const olOpts = document.getElementById('outlineOptions');
      if (olOpts && olBtn) {
        olOpts.style.display = olBtn.checked ? 'block' : 'none';
      }
    }

    // Force initial update
    setTimeout(() => {
      updateFootprintOptionsVisibility();
      updateDisplayModeOptions();
      updateCellBorderOptionsVisibility();
      updateOutlineOptionsVisibility();
    }, 50);

    // ═══════════════════════════════════════════════════════════════════════════
    // PARSE RESULT DATA
    // ═══════════════════════════════════════════════════════════════════════════
    const resultData = result || {};
    let bounds = Array.isArray(resultData.bounds) ? resultData.bounds : [-10, 40, 10, 60];
    if (!Array.isArray(bounds) || bounds.length !== 4 || bounds.some(b => b === null || isNaN(b))) {
      bounds = [-10, 40, 10, 60];
    }

    const hailMin = safeNum(resultData.hail_min, 0);
    const hailMax = safeNum(resultData.hail_max, 1);

    if (legendMin) legendMin.textContent = (hailMin || 0).toFixed(1);
    if (legendMax) legendMax.textContent = (hailMax || 1).toFixed(1);

    // ═══════════════════════════════════════════════════════════════════════════
    // MAP SETUP
    // ═══════════════════════════════════════════════════════════════════════════
    const isDark = document.documentElement.classList.contains('dark');
    if (basemapSelect && !basemapSelect.value) basemapSelect.value = isDark ? 'carto_dark' : 'carto_light';

    const center = Array.isArray(resultData.center) && resultData.center.length === 2
      ? resultData.center
      : [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2];

    const map = L.map('map', {
      center,
      zoom: 8,
      zoomControl: true,
      renderer: L.svg({ padding: 1.0 })
    });
    mapEl._hfMap = map;

    const basemapConfigs = {
      osm: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      carto_light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
      carto_dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      esri_worldstreet: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      esri_worldimagery: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
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

    // ═══════════════════════════════════════════════════════════════════════════
    // COLOR SCALE - Dynamic based on selected colormap
    // ═══════════════════════════════════════════════════════════════════════════
    let currentColorMap = 'ylOrRd';

    function getColorStops() {
      return COLOR_MAPS[currentColorMap]?.stops || COLOR_MAPS.ylOrRd.stops;
    }

    function getColorRGB(value) {
      const colorStops = getColorStops();
      const range = (hailMax - hailMin) || 1;
      const normalized = Math.max(0, Math.min(1, (value - hailMin) / range));

      let lower = colorStops[0];
      let upper = colorStops[colorStops.length - 1];

      for (let i = 0; i < colorStops.length - 1; i++) {
        if (normalized >= colorStops[i].pos && normalized <= colorStops[i + 1].pos) {
          lower = colorStops[i];
          upper = colorStops[i + 1];
          break;
        }
      }

      const range2 = upper.pos - lower.pos || 1;
      const t = (normalized - lower.pos) / range2;
      const r = Math.round(lower.color[0] + t * (upper.color[0] - lower.color[0]));
      const g = Math.round(lower.color[1] + t * (upper.color[1] - lower.color[1]));
      const b = Math.round(lower.color[2] + t * (upper.color[2] - lower.color[2]));

      return { r, g, b };
    }

    function getColor(value) {
      const { r, g, b } = getColorRGB(value);
      return `rgb(${r},${g},${b})`;
    }

    function updateLegendGradient() {
      const legendGradient = el('legendGradient');
      const colorMapPreview = el('colorMapPreview');

      const colorStops = getColorStops();
      const gradientStops = colorStops.map(s =>
        `rgb(${s.color[0]},${s.color[1]},${s.color[2]}) ${s.pos * 100}%`
      ).join(', ');

      // Update legend (horizontal)
      if (legendGradient) {
        legendGradient.style.background = `linear-gradient(to right, ${gradientStops})`;
      }

      // Also update colormap preview in control panel
      if (colorMapPreview) {
        colorMapPreview.style.background = `linear-gradient(to right, ${gradientStops})`;
      }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════════
    let footprintLayer = null;
    let canvasSmoothLayer = null;
    let outlineLayer = null;
    let pointsLayer = null;
    let currentOpacity = 0.6;
    let displayMode = 'cells';
    let smoothness = 3;
    let geoJsonData = null;

    // Cell border settings
    let showCellBordersEnabled = true;
    let cellBorderWidth = 0.5;
    let cellBorderColor = '#666666';

    // Outline settings
    let outlineWidth = 2.5;
    let outlineColor = '#000000';

    // Points settings
    let pointRadius = 6;
    let pointColor = '#ff7800';

    if (opacitySlider) opacitySlider.value = String(currentOpacity);
    if (opacityValue) opacityValue.textContent = currentOpacity.toFixed(1);

    const featureStyles = new WeakMap();

    // ═══════════════════════════════════════════════════════════════════════════
    // CANVAS CONTINUOUS LAYER
    // ═══════════════════════════════════════════════════════════════════════════
    const CanvasContinuousLayer = L.Layer.extend({
      initialize: function (geojson, options) {
        this._geojson = geojson;
        this._options = options || {};
        this._canvas = null;
      },

      onAdd: function (map) {
        this._map = map;
        this._canvas = L.DomUtil.create('canvas', 'hf-smooth-canvas leaflet-layer');

        const pane = map.getPane('overlayPane');
        pane.appendChild(this._canvas);

        map.on('moveend', this._update, this);
        map.on('zoomend', this._update, this);

        this._update();
      },

      onRemove: function (map) {
        if (this._canvas && this._canvas.parentNode) {
          this._canvas.parentNode.removeChild(this._canvas);
        }
        map.off('moveend', this._update, this);
        map.off('zoomend', this._update, this);
      },

      setOpacity: function (opacity) {
        this._options.opacity = opacity;
        if (this._canvas) {
          this._canvas.style.opacity = opacity;
        }
      },

      refresh: function () {
        this._update();
      },

      _update: function () {
        if (!this._map || !this._geojson) return;

        const map = this._map;
        const size = map.getSize();
        const topLeft = map.containerPointToLayerPoint([0, 0]);

        L.DomUtil.setPosition(this._canvas, topLeft);
        this._canvas.width = size.x;
        this._canvas.height = size.y;
        this._canvas.style.opacity = this._options.opacity || 0.6;

        const ctx = this._canvas.getContext('2d');
        ctx.clearRect(0, 0, size.x, size.y);

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        // Collect cell data
        const cells = [];
        const features = this._geojson.features || [];

        for (const feature of features) {
          if (!feature.geometry || feature.geometry.type !== 'Polygon') continue;

          const coords = feature.geometry.coordinates[0];
          const value = feature.properties?.Hail_Size || 0;

          // Get center of cell
          let cx = 0, cy = 0;
          for (const coord of coords) {
            cx += coord[0];
            cy += coord[1];
          }
          cx /= coords.length;
          cy /= coords.length;

          const point = map.latLngToContainerPoint([cy, cx]);

          // Estimate cell size in pixels
          const p1 = map.latLngToContainerPoint([coords[0][1], coords[0][0]]);
          const p2 = map.latLngToContainerPoint([coords[2][1], coords[2][0]]);
          const cellWidth = Math.abs(p2.x - p1.x);
          const cellHeight = Math.abs(p2.y - p1.y);

          cells.push({
            x: point.x,
            y: point.y,
            width: cellWidth,
            height: cellHeight,
            value: value
          });
        }

        const smoothLevel = this._options.smoothness || 3;

        // Draw cells with radial gradients for smooth blending
        for (const cell of cells) {
          const color = getColorRGB(cell.value);
          const radius = Math.max(cell.width, cell.height) * (1 + smoothLevel * 0.3);

          const gradient = ctx.createRadialGradient(
            cell.x, cell.y, 0,
            cell.x, cell.y, radius
          );

          gradient.addColorStop(0, `rgba(${color.r},${color.g},${color.b},1)`);
          gradient.addColorStop(0.5, `rgba(${color.r},${color.g},${color.b},0.8)`);
          gradient.addColorStop(0.8, `rgba(${color.r},${color.g},${color.b},0.3)`);
          gradient.addColorStop(1, `rgba(${color.r},${color.g},${color.b},0)`);

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(cell.x, cell.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }

        // Additional blur for extra smoothing
        const blurRadius = smoothLevel * 2;
        if (blurRadius > 0) {
          ctx.filter = `blur(${blurRadius}px)`;
          ctx.drawImage(this._canvas, 0, 0);
          ctx.filter = 'none';
        }
      }
    });

    // ═══════════════════════════════════════════════════════════════════════════
    // STYLE FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    function clearFeatureStyleCache() {
      // Clear the WeakMap cache by reassigning
      if (footprintLayer) {
        footprintLayer.eachLayer(layer => {
          if (layer.feature && featureStyles.has(layer.feature)) {
            featureStyles.delete(layer.feature);
          }
        });
      }
    }

    function getFeatureStyle(feature) {
      if (!featureStyles.has(feature)) {
        featureStyles.set(feature, {
          fillColor: getColor(feature.properties?.Hail_Size || 0)
        });
      }
      const cached = featureStyles.get(feature);

      if (displayMode === 'continuous') {
        // CONTINUOUS MODE: Hide SVG layer entirely (canvas takes over)
        return {
          fillColor: 'transparent',
          fill: false,
          stroke: false,
          weight: 0,
          opacity: 0,
          fillOpacity: 0
        };
      } else {
        // GRID CELLS MODE
        if (showCellBordersEnabled) {
          return {
            fillColor: cached.fillColor,
            fill: true,
            stroke: true,
            weight: cellBorderWidth,
            opacity: 0.8,
            color: cellBorderColor,
            fillOpacity: currentOpacity
          };
        } else {
          return {
            fillColor: cached.fillColor,
            fill: true,
            stroke: false,
            weight: 0,
            opacity: 0,
            fillOpacity: currentOpacity
          };
        }
      }
    }

    function refreshLayerStyles() {
      // Handle different display modes
      if (displayMode === 'continuous') {
        // Hide SVG layer, show canvas
        if (footprintLayer) {
          footprintLayer.eachLayer(layer => {
            if (layer.feature) {
              layer.setStyle({ fillOpacity: 0, opacity: 0, stroke: false, fill: false });
            }
          });
        }

        // Create/update canvas layer
        if (!canvasSmoothLayer && geoJsonData) {
          canvasSmoothLayer = new CanvasContinuousLayer(geoJsonData, {
            opacity: currentOpacity,
            smoothness: smoothness
          });
        }
        if (canvasSmoothLayer && showFootprint?.checked) {
          if (!map.hasLayer(canvasSmoothLayer)) {
            canvasSmoothLayer.addTo(map);
          }
          canvasSmoothLayer.setOpacity(currentOpacity);
          canvasSmoothLayer._options.smoothness = smoothness;
          canvasSmoothLayer._update();
        }

      } else {
        // Grid Cells mode
        if (canvasSmoothLayer && map.hasLayer(canvasSmoothLayer)) {
          map.removeLayer(canvasSmoothLayer);
        }
        if (footprintLayer) {
          footprintLayer.eachLayer(layer => {
            if (layer.feature) {
              layer.setStyle(getFeatureStyle(layer.feature));
            }
          });
          if (!map.hasLayer(footprintLayer) && showFootprint?.checked) {
            footprintLayer.addTo(map);
          }
        }
      }
    }

    function refreshColorsAndStyles() {
      // Clear cached colors
      clearFeatureStyleCache();

      // Re-cache colors with new colormap
      if (footprintLayer) {
        footprintLayer.eachLayer(layer => {
          if (layer.feature) {
            featureStyles.set(layer.feature, {
              fillColor: getColor(layer.feature.properties?.Hail_Size || 0)
            });
          }
        });
      }

      // Update legend gradient
      updateLegendGradient();

      // Refresh the layers
      refreshLayerStyles();

      // If canvas layer exists, force redraw
      if (canvasSmoothLayer && map.hasLayer(canvasSmoothLayer)) {
        canvasSmoothLayer.refresh();
      }
    }

    function updateOutlineStyle() {
      if (!outlineLayer) return;
      outlineLayer.eachLayer(layer => {
        layer.setStyle({
          fill: false,
          weight: outlineWidth,
          color: outlineColor,
          opacity: 1
        });
      });
    }

    function updatePointsStyle() {
      if (!pointsLayer) return;
      pointsLayer.eachLayer(layer => {
        if (layer.setStyle) {
          layer.setStyle({
            radius: pointRadius,
            fillColor: pointColor,
            color: '#000'
          });
        }
      });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DISPLAY MODES
    // ═══════════════════════════════════════════════════════════════════════════
    function updateDisplayMode() {
      displayMode = displayModeSelect?.value || 'cells';
      updateDisplayModeOptions();
      refreshLayerStyles();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EVENT LISTENERS
    // ═══════════════════════════════════════════════════════════════════════════

    // Footprint toggle
    showFootprint?.addEventListener('change', function () {
      updateFootprintOptionsVisibility();

      if (this.checked) {
        if (displayMode === 'continuous' && canvasSmoothLayer) {
          canvasSmoothLayer.addTo(map);
        } else if (footprintLayer) {
          footprintLayer.addTo(map);
        }
        refreshLayerStyles();
      } else {
        if (footprintLayer && map.hasLayer(footprintLayer)) {
          map.removeLayer(footprintLayer);
        }
        if (canvasSmoothLayer && map.hasLayer(canvasSmoothLayer)) {
          map.removeLayer(canvasSmoothLayer);
        }
      }
    });

    // Display mode change
    displayModeSelect?.addEventListener('change', updateDisplayMode);

    // Smoothness slider (for continuous mode)
    document.addEventListener('input', (e) => {
      if (e.target.id === 'smoothnessLevel') {
        smoothness = safeNum(e.target.value, 1);
        if (displayMode === 'continuous' && canvasSmoothLayer) {
          canvasSmoothLayer._options.smoothness = smoothness;
          canvasSmoothLayer._update();
        }
      }
    });

    // Color map selector
    document.addEventListener('change', (e) => {
      if (e.target.id === 'colorMapSelect') {
        currentColorMap = e.target.value;
        updateColorMapPreview(currentColorMap);
        refreshColorsAndStyles();
      }
    });

    // Cell borders toggle
    showCellBorders?.addEventListener('change', function () {
      showCellBordersEnabled = this.checked;
      updateCellBorderOptionsVisibility();
      refreshLayerStyles();
    });

    // Cell border settings
    function updateCellSettings(e) {
      if (e.target.id === 'cellBorderWidth') cellBorderWidth = safeNum(e.target.value, 0.5);
      if (e.target.id === 'cellBorderColor') cellBorderColor = e.target.value;
      refreshLayerStyles();
    }

    document.addEventListener('input', (e) => {
      if (['cellBorderWidth', 'cellBorderColor'].includes(e.target.id)) updateCellSettings(e);
    });
    document.addEventListener('change', (e) => {
      if (e.target.id === 'cellBorderColor') updateCellSettings(e);
    });

    // Outline toggle
    showOutline?.addEventListener('change', function () {
      updateOutlineOptionsVisibility();
      if (outlineLayer) {
        if (this.checked) outlineLayer.addTo(map);
        else map.removeLayer(outlineLayer);
      }
    });

    // Outline settings
    function updateOutlineSettings(e) {
      if (e.target.id === 'outlineWidth') outlineWidth = safeNum(e.target.value, 2.5);
      if (e.target.id === 'outlineColor') outlineColor = e.target.value;
      updateOutlineStyle();
    }

    document.addEventListener('input', (e) => {
      if (['outlineWidth', 'outlineColor'].includes(e.target.id)) updateOutlineSettings(e);
    });
    document.addEventListener('change', (e) => {
      if (e.target.id === 'outlineColor') updateOutlineSettings(e);
    });

    // Points toggle
    showPoints?.addEventListener('change', function () {
      const pOpts = document.getElementById('pointOptions');
      if (pOpts) pOpts.style.display = 'block'; // Ensure visibility if needed, though mostly handled by flex layout

      if (!pointsLayer) return;
      if (this.checked) pointsLayer.addTo(map);
      else map.removeLayer(pointsLayer);
      el('togglePointsBtn')?.classList.toggle('hf-active', this.checked);
    });

    // Points settings
    function updatePointSettings(e) {
      if (e.target.id === 'pointRadius') pointRadius = safeNum(e.target.value, 6);
      if (e.target.id === 'pointColor') pointColor = e.target.value;
      updatePointsStyle();
    }

    document.addEventListener('input', (e) => {
      if (['pointRadius', 'pointColor'].includes(e.target.id)) updatePointSettings(e);
    });
    document.addEventListener('change', (e) => {
      if (e.target.id === 'pointColor') updatePointSettings(e);
    });

    // Opacity slider
    opacitySlider?.addEventListener('input', function () {
      currentOpacity = safeNum(this.value, 0.6);
      if (opacityValue) opacityValue.textContent = currentOpacity.toFixed(1);
      refreshLayerStyles();
    });

    // Basemap selector
    basemapSelect?.addEventListener('change', function () {
      currentBasemapId = this.value;
      if (currentBasemap) map.removeLayer(currentBasemap);
      currentBasemap = basemaps[currentBasemapId];
      if (currentBasemap) {
        currentBasemap.addTo(map);
        currentBasemap.bringToBack();
      }
    });

    // Quick action buttons
    el('resetViewBtn')?.addEventListener('click', () => {
      try { map.fitBounds(leafletBounds, { padding: [50, 50], maxZoom: 12 }); }
      catch (e) { map.setView(center, 8); }
    });

    el('toggleLayersBtn')?.addEventListener('click', () => {
      if (showFootprint) {
        showFootprint.checked = !showFootprint.checked;
        showFootprint.dispatchEvent(new Event('change'));
      }
    });

    el('togglePointsBtn')?.addEventListener('click', () => {
      if (showPoints) {
        showPoints.checked = !showPoints.checked;
        showPoints.dispatchEvent(new Event('change'));
      }
    });

    // Panel Toggle
    function togglePanel() {
      if (!controlPanel) return;
      controlPanel.classList.toggle('hf-collapsed');
      const collapsed = controlPanel.classList.contains('hf-collapsed');
      if (toggleIcon) toggleIcon.className = collapsed ? 'bi bi-chevron-left text-xs' : 'bi bi-chevron-right text-xs';
    }
    panelToggle?.addEventListener('click', (e) => { e.stopPropagation(); togglePanel(); });
    panelHeader?.addEventListener('click', togglePanel);

    // ═══════════════════════════════════════════════════════════════════════════
    // DATA LOADING
    // ═══════════════════════════════════════════════════════════════════════════
    async function loadLayers() {
      try {
        // Footprint (Polygons)
        const footprintRes = await fetch(`geojson/${jobId}`);
        console.log(footprintRes);
        if (footprintRes.ok) {
          const data = await footprintRes.json();
          geoJsonData = data;

          if (data.features?.length) {
            footprintLayer = L.geoJSON(data, {
              style: getFeatureStyle,
              smoothFactor: 0.5,
              onEachFeature: (feature, layer) => {
                getFeatureStyle(feature);

                layer.on({
                  mouseover: () => {
                    const val = feature.properties?.Hail_Size || 0;
                    const hoverInfo = el('hoverInfo');
                    if (hoverInfo) hoverInfo.innerHTML = `<strong>${Number(val).toFixed(2)} cm</strong>`;
                  },
                  mouseout: () => {
                    const hoverInfo = el('hoverInfo');
                    if (hoverInfo) hoverInfo.textContent = 'Hover over map';
                  },
                  click: (e) => {
                    const val = feature.properties?.Hail_Size || 0;
                    L.popup().setLatLng(e.latlng).setContent(`<b>Hail Size:</b> ${Number(val).toFixed(2)} cm`).openOn(map);
                  }
                });
              }
            });

            footprintLayer.addTo(map);
            refreshLayerStyles();

            // Update legend with initial colormap
            updateLegendGradient();
          }
        }

        // Outline
        const outlineRes = await fetch(`footprint_geojson/${jobId}`);
        if (outlineRes.ok) {
          const data = await outlineRes.json();
          if (data.features?.length) {
            outlineLayer = L.geoJSON(data, {
              style: {
                fill: false,
                weight: outlineWidth,
                color: outlineColor,
                opacity: 1
              }
            }).addTo(map);
          }
        }

        // Points
        const pointsRes = await fetch(`points_geojson/${jobId}`);
        if (pointsRes.ok) {
          const data = await pointsRes.json();
          if (data.features?.length) {
            pointsLayer = L.geoJSON(data, {
              pointToLayer: (_f, latlng) => L.circleMarker(latlng, {
                radius: pointRadius, fillColor: pointColor, color: '#000', weight: 1, opacity: 1, fillOpacity: 0.8
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

        // Initialize colormap selector after a delay
        setTimeout(() => {
          updateLegendGradient();
        }, 200);

      } catch (err) {
        console.error('[Viewer] Load error:', err);
        const loading = el('loading-overlay');
        if (loading) loading.style.display = 'none';
        showInlineError('Failed to load map data: ' + err.message);
      }
    }

    await loadLayers();

    // ═══════════════════════════════════════════════════════════════════════════
    // MAP EVENTS
    // ═══════════════════════════════════════════════════════════════════════════
    map.on('mousemove', (e) => {
      const c = el('coordsDisplay');
      if (c) c.textContent = `${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`;
    });
    L.control.scale({ metric: true, imperial: false }).addTo(map);

    // ═══════════════════════════════════════════════════════════════════════════
    // EXPORT
    // ═══════════════════════════════════════════════════════════════════════════
    function getCurrentBounds() {
      const b = map.getBounds();
      return [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    }

    function getCurrentSettings() {
      return {
        displayMode: displayModeSelect?.value || 'cells',
        showCellBorders: document.getElementById('showCellBorders')?.checked ?? showCellBordersEnabled,
        cellBorderWidth: safeNum(document.getElementById('cellBorderWidth')?.value, cellBorderWidth),
        cellBorderColor: document.getElementById('cellBorderColor')?.value || cellBorderColor,
        outlineWidth: safeNum(document.getElementById('outlineWidth')?.value, outlineWidth),
        outlineColor: document.getElementById('outlineColor')?.value || outlineColor,
        smoothness: smoothness,
        colorMap: currentColorMap,
        pointRadius: safeNum(document.getElementById('pointRadius')?.value, pointRadius),
        pointColor: document.getElementById('pointColor')?.value || pointColor
      };
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
        if (exportBtn) {
          exportBtn.disabled = true;
          exportBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin"></i> Generating…';
        }
        window.HailUI?.toast?.('Generating high-quality map…', { type: 'info', title: 'Export' });

        const settings = getCurrentSettings();

        const exportData = {
          bounds: getCurrentBounds(),
          basemap: currentBasemapId,
          show_footprint: !!showFootprint?.checked,
          show_outline: !!showOutline?.checked,
          show_points: !!showPoints?.checked,
          opacity: currentOpacity,
          display_mode: settings.displayMode,
          show_cell_borders: settings.showCellBorders,
          cell_border_width: settings.cellBorderWidth,
          cell_border_color: settings.cellBorderColor,
          outline_width: settings.outlineWidth,
          outline_color: settings.outlineColor,
          point_radius: settings.pointRadius,
          point_color: settings.pointColor,
          smoothness: settings.smoothness,
          color_map: settings.colorMap,
          width_px: 3200,
          height_px: 2000
        };

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000);

        const res = await fetch(`render_map/${jobId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(exportData),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!res.ok) {
          const errorText = await res.text();
          throw new Error('Export failed: ' + (errorText || res.statusText));
        }

        const blob = await res.blob();
        if (blob.size < 1000) throw new Error('Generated image is too small');

        const url = URL.createObjectURL(blob);
        const filename = `${eventName || 'hail_footprint'}_map.png`;
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 1000);

        window.HailUI?.toast?.('Map exported successfully!', { type: 'success', title: 'Export Complete' });

      } catch (e) {
        console.error('[Viewer] Export error:', e);
        showInlineError(e.message || 'Export failed.');
        window.HailUI?.toast?.(e.message || 'Export failed', { type: 'danger', title: 'Export Error' });
      } finally {
        isExporting = false;
        if (exportBtn) {
          exportBtn.disabled = false;
          exportBtn.innerHTML = originalContent || '<i class="bi bi-camera"></i> HQ export';
        }
      }
    }

    el('exportCurrentViewBtn')?.addEventListener('click', async () => {
      try { await exportServerRendered(); } catch (e) { showInlineError(e.message); }
    });

    console.log('[Viewer] ✓ Initialization complete');
  }

  window.HailViewer = { init };
})();