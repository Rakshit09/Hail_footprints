// Single-page progressive flow for upload -> configure -> process -> inline viewer

(function () {
  const root = document.getElementById('pageRoot');
  const MAX_UPLOAD_BYTES = Number(root?.dataset?.maxUploadBytes || 50 * 1024 * 1024);
  const ACCEPTED_EXT = (root?.dataset?.acceptedExt || 'csv,gpkg,geojson,shp')
    .split(',')
    .map(s => s.trim().toLowerCase())
    .filter(Boolean);

  const $ = (id) => document.getElementById(id);
  const ui = () => window.HailUI;

  let uploadedData = null;
  let socket = null;
  let uploadState = 'idle'; // idle | uploading | success | error

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return '';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    let v = bytes;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function setProgress(pct, label) {
    const bar = $('uploadProgressBar');
    const text = $('uploadProgressText');
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (text) text.textContent = label || '';
  }

  function setUploadState(state, meta) {
    // state: idle | dragging | uploading | success | error
    const dz = $('dropzone');
    const pill = $('filePill');
    const pillName = $('filePillName');
    const pillMeta = $('filePillMeta');
    const progressWrap = $('uploadProgressWrap');
    const errorBox = $('uploadError');
    const fileHelp = $('fileHelp');

    if (state === 'dragging') {
      dz?.classList.add('active');
      return;
    }
    dz?.classList.remove('active');
    uploadState = state;

    if (state === 'idle') {
      dz?.classList.remove('hidden');
      pill?.classList.add('hidden');
      progressWrap?.classList.add('hidden');
      errorBox?.classList.add('hidden');
      fileHelp?.classList.remove('hidden');
      setProgress(0, '');
      return;
    }

    if (state === 'uploading') {
      dz?.classList.remove('hidden');
      pill?.classList.add('hidden');
      progressWrap?.classList.remove('hidden');
      errorBox?.classList.add('hidden');
      fileHelp?.classList.add('hidden');
      setProgress(55, 'Uploading & parsing…');
    }

    if (state === 'success') {
      pill?.classList.remove('hidden');
      dz?.classList.add('hidden');
      progressWrap?.classList.add('hidden');
      errorBox?.classList.add('hidden');
      fileHelp?.classList.add('hidden');
      if (pillName) pillName.textContent = meta?.name || 'Uploaded file';
      if (pillMeta) pillMeta.textContent = meta?.meta || '';
      setProgress(100, 'Done');
    }

    if (state === 'error') {
      dz?.classList.remove('hidden');
      progressWrap?.classList.add('hidden');
      errorBox?.classList.remove('hidden');
      fileHelp?.classList.remove('hidden');
      errorBox && (errorBox.innerHTML = `
        <div class="flex items-start gap-3">
          <i class="bi bi-exclamation-triangle text-red-500"></i>
          <div>
            <div class="text-sm font-semibold text-red-800">Upload failed</div>
            <div class="text-sm text-red-700">${meta?.message || 'Please try again.'}</div>
          </div>
        </div>
      `);
      setProgress(0, '');
    }
  }

  function validateFile(file) {
    if (!file) return 'No file selected';
    if (file.size > MAX_UPLOAD_BYTES) return `File is too large (${formatBytes(file.size)}). Max allowed is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
    const name = file.name || '';
    const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : '';
    if (!ACCEPTED_EXT.includes(ext)) return `Unsupported file type ".${ext}". Allowed: ${ACCEPTED_EXT.join(', ').toUpperCase()}.`;
    return null;
  }

  function populateSelect(selectId, columns, selected) {
    const select = $(selectId);
    if (!select) return;
    select.innerHTML = `<option value="">Select…</option>` + columns.map(c => (
      `<option value="${escapeHtml(c)}"${c === selected ? ' selected' : ''}>${escapeHtml(c)}</option>`
    )).join('');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[m]));
  }

  function showPreview(columns, rows) {
    const thead = document.getElementById('previewThead');
    const tbody = document.getElementById('previewTbody');
    if (!thead || !tbody) return;
    
    thead.innerHTML = `<tr>` + columns.map(c => `<th class="px-4 py-3 text-left">${escapeHtml(c)}</th>`).join('') + `</tr>`;
    tbody.innerHTML = (rows || []).map(r => (
        `<tr class="hover:bg-gray-50">` +
        columns.map(c => `<td class="px-4 py-2.5 whitespace-nowrap">${escapeHtml(r?.[c] ?? '')}</td>`).join('') +
        `</tr>`
    )).join('');
    
    // Sync heights after populating
    setTimeout(syncPreviewHeight, 50);
}

  // Sync Data Preview height to match left column
  function syncPreviewHeight() {
    const leftColumn = document.getElementById('leftConfigColumn');
    const previewCard = document.getElementById('dataPreviewCard');
    const previewContainer = document.getElementById('previewTableContainer');
    
    if (leftColumn && previewCard && previewContainer) {
        // Get the left column height
        const leftHeight = leftColumn.offsetHeight;
        
        // Set the preview card to match
        previewCard.style.height = leftHeight + 'px';
        
        // Calculate header height and set container height
        const header = previewCard.querySelector('.config-group-header');
        const headerHeight = header ? header.offsetHeight : 45;
        previewContainer.style.height = (leftHeight - headerHeight) + 'px';
    }
  }

  // Call on config section show and window resize
  window.addEventListener('resize', syncPreviewHeight);
  document.getElementById('configSection').style.display = 'block';
  setTimeout(syncPreviewHeight, 100);

// Also call after preview is populated - add this after showPreview() is called
// Example: after your existing showPreview call, add:
// syncPreviewHeight();
  function revealConfigSection() {
    const section = $('configSection');
    if (section) {
      ui()?.reveal?.(section);
      setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 180);
    }
  }

  async function uploadFile(file) {
    const err = validateFile(file);
    if (err) {
      setUploadState('error', { message: err });
      ui()?.toast?.(err, { type: 'error', title: 'Invalid file' });
      return;
    }

    // Reset downstream UI for a fresh run
    const pb = $('processBtn');
    if (pb) pb.disabled = true;
    const resultsP = $('resultsPanel');
    if (resultsP) resultsP.style.display = 'none';
    const procP = $('processingPanel');
    if (procP) procP.style.display = 'none';
    ui()?.collapse?.($('viewerSection'));
    $('errorContainer') && ($('errorContainer').innerHTML = '');

    setUploadState('uploading');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Upload failed');

      uploadedData = data;
      $('uploadedFilename').value = data.filename;
      $('jobId').value = data.job_id;

      setUploadState('success', {
        name: file.name,
        meta: `${data.columns.length} columns · ~${data.row_count.toLocaleString()} rows`
      });

      // Populate selects with suggestions
      populateSelect('lonCol', data.columns, data.suggestions?.longitude);
      populateSelect('latCol', data.columns, data.suggestions?.latitude);
      populateSelect('hailCol', data.columns, data.suggestions?.hail_size);
      showPreview(data.columns, data.sample_data);

      // Enable process button + reveal config section
      $('processBtn')?.classList.remove('opacity-50');
      $('processBtn').disabled = false;
      revealConfigSection();

      ui()?.toast?.('File uploaded and parsed.', { type: 'success', title: 'Ready' });
    } catch (e) {
      console.error(e);
      setUploadState('error', { message: e.message });
      ui()?.toast?.(e.message, { type: 'error', title: 'Upload failed' });
    }
  }

  function initDropzone() {
    const dz = $('dropzone');
    const input = $('fileInput');
    const changeBtn = $('changeFileBtn');

    if (!dz || !input) return;

    dz.addEventListener('click', () => input.click());
    changeBtn?.addEventListener('click', () => input.click());

    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) uploadFile(file);
    });

    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      setUploadState('dragging');
    });
    dz.addEventListener('dragleave', () => dz.classList.remove('active'));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('active');
      const file = e.dataTransfer?.files?.[0];
      if (file) uploadFile(file);
    });
  }

  function validateParams(params) {
    const errors = [];
    if (!params.lon_col) errors.push('Longitude column is required');
    if (!params.lat_col) errors.push('Latitude column is required');
    if (!params.hail_col) errors.push('Hail size column is required');
    if (!params.event_name) errors.push('Event name is required');
    if (params.event_name && !/^[a-zA-Z0-9_-]+$/.test(params.event_name)) {
      errors.push('Event name can only contain letters, numbers, underscores, and hyphens');
    }
    return errors;
  }

  function showProcessingUI() {
    const p = $('processingPanel');
    if (p) p.style.display = 'block';
    const r = $('resultsPanel');
    if (r) r.style.display = 'none';
    ui()?.collapse?.($('viewerSection'));
    $('processingProgressLabel').textContent = '0%';
    $('processingProgressBar').style.width = '0%';
    $('statusMessage').textContent = 'Initializing…';
  }

  function updateProcessingUI(progress, message) {
    const pct = Math.max(0, Math.min(100, Number(progress) || 0));
    $('processingProgressBar').style.width = `${pct}%`;
    $('processingProgressLabel').textContent = `${pct}%`;
    $('statusMessage').textContent = message || '';
  }

  function showResults(jobId, result) {
    const panel = $('resultsPanel');
    if (panel) panel.style.display = 'block';

    // Downloads
    const pngLink = $('downloadPng');
    if (pngLink) {
      if (result?.footprint_png) {
        pngLink.href = `/outputs/${jobId}/${result.footprint_png}`;
        pngLink.classList.remove('hidden');
      } else {
        pngLink.classList.add('hidden');
      }
    }
    const geoLink = $('downloadGeoJson');
    if (geoLink) {
      if (result?.geojson) {
        geoLink.href = `/outputs/${jobId}/${result.geojson}`;
        geoLink.classList.remove('hidden');
      } else {
        geoLink.classList.add('hidden');
      }
    }

    // Summary
    const summary = $('resultsSummaryBody');
    if (summary) {
      const rows = [
        ['Points Processed', result?.n_points],
        ['Groups Found', result?.n_groups],
        ['Cell Size', result?.cell_size_m ? `${Number(result.cell_size_m).toFixed(1)} m` : '—'],
        ['Hail Size Range', (Number.isFinite(result?.hail_min) && Number.isFinite(result?.hail_max)) ? `${Number(result.hail_min).toFixed(2)} – ${Number(result.hail_max).toFixed(2)}` : '—'],
        ['Processing Time', result?.processing_seconds ? `${result.processing_seconds} seconds` : '—']
      ];

      summary.innerHTML = rows.map(([k, v]) => `
        <tr>
          <td>${escapeHtml(k)}</td>
          <td>${escapeHtml(v ?? '—')}</td>
        </tr>
      `).join('');
    }

    // New-tab viewer link (keeps the old route available)
    const newTab = $('openViewerNewTab');
    if (newTab) newTab.href = `/viewer/${jobId}`;
  }

  function revealViewer(jobId, eventName, result) {
    const section = $('viewerSection');
    ui()?.reveal?.(section);
    $('viewerTitle').textContent = eventName || 'Interactive Viewer';

    // Default basemap option
    const bs = $('basemapSelect');
    if (bs && !bs.value) bs.value = 'carto_light';

    // (Re)show loading overlay
    const loading = $('loading-overlay');
    if (loading) loading.style.display = 'flex';

    setTimeout(() => {
      try {
        window.HailViewer?.init?.({ jobId, eventName, result });
        section?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (e) {
        console.error(e);
        ui()?.toast?.(e.message, { type: 'error', title: 'Viewer failed to start' });
      }
    }, 120);
  }

  async function startProcessing() {
    const filename = $('uploadedFilename').value;
    const jobId = $('jobId').value;

    const params = {
      filename,
      job_id: jobId,
      lon_col: $('lonCol').value,
      lat_col: $('latCol').value,
      hail_col: $('hailCol').value,
      event_name: $('eventName').value,
      grouping_threshold_km: Number($('groupingThreshold').value),
      large_buffer_km: Number($('largeBuffer').value),
      small_buffer_km: Number($('smallBuffer').value)
    };

    const errors = validateParams(params);
    if (errors.length) {
      ui()?.toast?.(errors[0], { type: 'error', title: 'Missing info' });
      return;
    }

    ui()?.setBusy?.($('processBtn'), true);
    showProcessingUI();

    try {
      const res = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Failed to start processing');
      ui()?.toast?.('Processing started.', { type: 'info', title: 'Working' });
    } catch (e) {
      ui()?.setBusy?.($('processBtn'), false);
      ui()?.toast?.(e.message, { type: 'error', title: 'Processing failed' });
    }
  }

  function initSocket() {
    if (!window.io) return;
    socket = window.io();

    socket.on('progress', (data) => {
      updateProcessingUI(data.progress, data.message);
    });

    socket.on('completed', (data) => {
      const jobId = data.job_id;
      const result = data.result;
      ui()?.setBusy?.($('processBtn'), false);
      const p = $('processingPanel');
      if (p) p.style.display = 'none';
      showResults(jobId, result);
      revealViewer(jobId, $('eventName').value, result);
      ui()?.toast?.('Footprint generated successfully.', { type: 'success', title: 'Done' });
    });

    socket.on('error', (data) => {
      ui()?.setBusy?.($('processBtn'), false);
      ui()?.toast?.(data.error || 'Unknown error', { type: 'error', title: 'Processing error' });
    });
  }

  function init() {
    setUploadState('idle');
    initDropzone();
    initSocket();
    $('processBtn')?.addEventListener('click', startProcessing);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
