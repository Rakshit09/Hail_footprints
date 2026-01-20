// One page flow: upload -> configure -> process -> inline viewer

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
  let currentPoller = null;
  let uploadState = 'idle';

  // Job Poller 
  class JobPoller {
    constructor(jobId, options = {}) {
      this.jobId = jobId;
      this.pollInterval = options.pollInterval || 1000;
      this.maxRetries = options.maxRetries || 10;
      this.retryCount = 0;
      this.onProgress = options.onProgress || (() => { });
      this.onComplete = options.onComplete || (() => { });
      this.onError = options.onError || (() => { });
      this.polling = false;
      this.timeoutId = null;
    }

    start() {
      this.polling = true;
      this.retryCount = 0;
      this.poll();
    }

    stop() {
      this.polling = false;
      if (this.timeoutId) {
        clearTimeout(this.timeoutId);
        this.timeoutId = null;
      }
    }

    async poll() {
      if (!this.polling) return;

      try {
        const response = await fetch(`status/${this.jobId}`);

        const contentType = response.headers.get('content-type') || '';
        let data = null;

        // parse JSON 
        if (contentType.includes('application/json')) {
          try {
            data = await response.json();
          } catch (e) {
            console.warn('Could not parse JSON despite header');
          }
        }

        // Handle HTTP Errors
        if (!response.ok) {
          const isAlive = data && (data.status === 'processing' || data.status === 'queued');

          if (response.status === 404 && !isAlive) {
            this.onError({ message: 'Job not found (404)' });
            this.stop();
            return;
          }

          if (!isAlive && response.status !== 404) {
            throw new Error(`HTTP ${response.status}`);
          }
        }

        if (!data) throw new Error('No data received');

        this.retryCount = 0;

        // update progress
        this.onProgress({
          progress: data.progress,
          message: data.message,
          status: data.status
        });

        // handle different statuses
        switch (data.status) {
          case 'completed':
            this.onComplete({ job_id: this.jobId, result: data.result });
            this.stop();
            break;

          case 'error':
            this.onError({ error: data.error || 'Processing failed' });
            this.stop();
            break;

          case 'processing':
          case 'queued':
            this.scheduleNextPoll(this.pollInterval);
            break;

          default:
            console.warn('Unknown job status:', data.status);
            this.scheduleNextPoll(this.pollInterval);
        }

      } catch (error) {
        console.error('Polling error:', error);
        this.retryCount++;

        if (this.retryCount >= this.maxRetries) {
          this.onError({ error: `Connection lost after ${this.maxRetries} retries` });
          this.stop();
        } else {
          const delay = this.pollInterval * Math.pow(1.5, this.retryCount);
          this.scheduleNextPoll(delay);
        }
      }
    }

    scheduleNextPoll(delay) {
      if (this.polling) {
        this.timeoutId = setTimeout(() => this.poll(), delay);
      }
    }
  }

  // utility functions
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

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[m]));
  }

  // safe JSON parsing helper
  async function safeJsonParse(response) {
    const contentType = response.headers.get('content-type') || '';
    const text = await response.text();

    if (!contentType.includes('application/json')) {
      console.error('Response is not JSON:', contentType);
      console.error('Response body preview:', text.substring(0, 200));

      try {
        return JSON.parse(text);
      } catch (e) {
        throw new Error(`Server returned ${contentType} instead of JSON. Response: ${text.substring(0, 100)}`);
      }
    }

    try {
      return JSON.parse(text);
    } catch (e) {
      console.error('JSON parse error:', e);
      console.error('Response text:', text.substring(0, 500));
      throw new Error(`Invalid JSON response: ${e.message}`);
    }
  }

  // upload state management
  function setProgress(pct, label) {
    const bar = $('uploadProgressBar');
    const text = $('uploadProgressText');
    if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (text) text.textContent = label || '';
  }

  function setUploadState(state, meta) {
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
    if (file.size > MAX_UPLOAD_BYTES) {
      return `File is too large (${formatBytes(file.size)}). Max allowed is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
    }
    const name = file.name || '';
    const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : '';
    if (!ACCEPTED_EXT.includes(ext)) {
      return `Unsupported file type ".${ext}". Allowed: ${ACCEPTED_EXT.join(', ').toUpperCase()}.`;
    }
    return null;
  }

  // column selection & preview
  function populateSelect(selectId, columns, selected) {
    const select = $(selectId);
    if (!select) return;
    select.innerHTML = `<option value="">Select…</option>` + columns.map(c => (
      `<option value="${escapeHtml(c)}"${c === selected ? ' selected' : ''}>${escapeHtml(c)}</option>`
    )).join('');
  }

  function showPreview(columns, rows) {
    const thead = document.getElementById('previewThead');
    const tbody = document.getElementById('previewTbody');
    const hailColName = document.getElementById('hailCol')?.value;
    const qcColName = document.getElementById('qcCol')?.value;
    const missingBadge = document.getElementById('missingHailBadge');
    const missingCount = document.getElementById('missingHailCount');
    const qcBadge = document.getElementById('qcIssueBadge');
    const qcCount = document.getElementById('qcIssueCount');

    if (!thead || !tbody) return;

    // check if QC value is ok (1 or QC1)
    function isQcOk(val) {
      if (val === null || val === undefined || val === '') return true; // No QC = OK
      const strVal = String(val).trim().toLowerCase();
      return strVal === '1' || strVal === 'qc1' || strVal === 'q1';
    }

    // missing hail values
    let missingHailRows = 0;
    if (hailColName && rows) {
      missingHailRows = rows.filter(r =>
        r?.[hailColName] === null || r?.[hailColName] === undefined || r?.[hailColName] === ''
      ).length;
    }

    // QC issue rows
    let qcIssueRows = 0;
    if (qcColName && rows) {
      qcIssueRows = rows.filter(r => !isQcOk(r?.[qcColName])).length;
    }

    // update missing hail badge and count
    if (missingBadge && missingCount) {
      if (missingHailRows > 0) {
        missingCount.textContent = missingHailRows;
        missingBadge.classList.remove('hidden');
      } else {
        missingBadge.classList.add('hidden');
      }
    }

    // update QC issue badge and count
    if (qcBadge && qcCount) {
      if (qcIssueRows > 0) {
        qcCount.textContent = qcIssueRows;
        qcBadge.classList.remove('hidden');
      } else {
        qcBadge.classList.add('hidden');
      }
    }

    thead.innerHTML = `<tr>` + columns.map(c => `<th class="px-4 py-3 text-left">${escapeHtml(c)}</th>`).join('') + `</tr>`;
    tbody.innerHTML = (rows || []).map(r => {
      const isMissingHail = hailColName && (r?.[hailColName] === null || r?.[hailColName] === undefined || r?.[hailColName] === '');
      const isQcBad = qcColName && !isQcOk(r?.[qcColName]);

      let rowClass = 'hover:bg-gray-50';
      let rowStyle = '';

      if (isMissingHail && isQcBad) {
        rowStyle = 'style="background: linear-gradient(to bottom, rgb(254 226 226 / 0.85) 50%, rgb(254 243 199 / 0.85) 50%);"';
        rowClass = 'hover:opacity-90';
      } else if (isMissingHail) {
        rowClass = 'bg-red-50/70 hover:bg-red-100/70';
      } else if (isQcBad) {
        rowClass = 'bg-amber-50/70 hover:bg-amber-100/70';
      }

      return (
        `<tr class="${rowClass}" ${rowStyle}>` +
        columns.map(c => `<td class="px-4 py-2.5 whitespace-nowrap">${escapeHtml(r?.[c] ?? '')}</td>`).join('') +
        `</tr>`
      );
    }).join('');

    setTimeout(syncPreviewHeight, 50);
  }

  // refresh preview 
  document.getElementById('hailCol')?.addEventListener('change', () => {
    if (uploadedData) {
      showPreview(uploadedData.columns, uploadedData.sample_data);
    }
  });

  document.getElementById('qcCol')?.addEventListener('change', () => {
    if (uploadedData) {
      showPreview(uploadedData.columns, uploadedData.sample_data);
    }
  });

  function syncPreviewHeight() {
    const leftColumn = document.getElementById('leftConfigColumn');
    const previewCard = document.getElementById('dataPreviewCard');
    const previewContainer = document.getElementById('previewTableContainer');

    if (leftColumn && previewCard && previewContainer) {
      const leftHeight = leftColumn.offsetHeight;
      previewCard.style.height = leftHeight + 'px';
      const header = previewCard.querySelector('.config-group-header');
      const headerHeight = header ? header.offsetHeight : 45;
      previewContainer.style.height = (leftHeight - headerHeight) + 'px';
    }
  }

  window.addEventListener('resize', syncPreviewHeight);

  function revealConfigSection() {
    const section = $('configSection');
    if (section) {
      ui()?.reveal?.(section);
      setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 180);
    }
  }

  // file upload
  async function uploadFile(file) {
    const err = validateFile(file);
    if (err) {
      setUploadState('error', { message: err });
      ui()?.toast?.(err, { type: 'error', title: 'Invalid file' });
      return;
    }

    // reset downstream UI
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
      const res = await fetch('upload', { method: 'POST', body: formData });

      if (!res.ok) {
        const contentType = res.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
          const textBody = await res.text();
          throw new Error(`Server Error (${res.status}): ${textBody || res.statusText}`);
        }
      }

      const data = await safeJsonParse(res);

      if (!data.success) {
        throw new Error(data.error || 'Upload failed');
      }

      uploadedData = data;
      $('uploadedFilename').value = data.filename;
      $('jobId').value = data.job_id;

      setUploadState('success', {
        name: file.name,
        meta: `${data.columns.length} columns · ~${data.row_count.toLocaleString()} rows`
      });

      // populate selects with suggestions
      console.log('[Upload] Received suggestions:', data.suggestions);
      console.log('[Upload] Applying longitude suggestion:', data.suggestions?.longitude);
      console.log('[Upload] Applying latitude suggestion:', data.suggestions?.latitude);
      console.log('[Upload] Applying hail_size suggestion:', data.suggestions?.hail_size);
      console.log('[Upload] Applying qc_level suggestion:', data.suggestions?.qc_level);

      populateSelect('lonCol', data.columns, data.suggestions?.longitude);
      populateSelect('latCol', data.columns, data.suggestions?.latitude);
      populateSelect('hailCol', data.columns, data.suggestions?.hail_size);
      populateSelect('qcCol', data.columns, data.suggestions?.qc_level);
      showPreview(data.columns, data.sample_data);

      $('processBtn')?.classList.remove('opacity-50');
      $('processBtn').disabled = false;
      revealConfigSection();

      ui()?.toast?.('File uploaded and parsed.', { type: 'success', title: 'Ready' });

    } catch (e) {
      console.error('Upload error:', e);
      setUploadState('error', { message: e.message });
      ui()?.toast?.(e.message, { type: 'error', title: 'Upload failed' });
    }
  }

  // dropzone initialization
  function initDropzone() {
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('fileInput');
    const changeBtn = document.getElementById('changeFileBtn');

    if (!dz || !input) return;

    const toggleLoading = (show) => {
      const overlayId = 'dz-loading-overlay';
      const existing = document.getElementById(overlayId);

      if (show) {
        if (existing) return;
        dz.classList.add('relative');

        const overlay = document.createElement('div');
        overlay.id = overlayId;
        overlay.className = 'absolute inset-0 z-50 bg-white/75 flex items-center justify-center rounded-lg';
        overlay.innerHTML = `
          <div class="flex items-center gap-2 px-3 py-1.5 bg-white shadow-sm border border-gray-100 rounded-full">
             <svg class="animate-spin h-4 w-4 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
               <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
               <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
             </svg>
             <span class="text-xs font-medium text-gray-700">Opening...</span>
          </div>
        `;
        dz.appendChild(overlay);
        document.body.style.cursor = 'wait';
      } else {
        if (existing) existing.remove();
        dz.classList.remove('relative');
        document.body.style.cursor = '';
      }
    };

    const handleClick = () => {
      if (uploadState === 'uploading') return;
      toggleLoading(true);
      setTimeout(() => input.click(), 10);
    };

    dz.addEventListener('click', handleClick);
    if (changeBtn) changeBtn.addEventListener('click', handleClick);

    input.addEventListener('change', () => {
      document.body.style.cursor = '';
      toggleLoading(false);
      const file = input.files?.[0];
      if (file) uploadFile(file);
    });

    window.addEventListener('focus', () => {
      setTimeout(() => {
        if (uploadState === 'idle' || uploadState === 'error') {
          toggleLoading(false);
        }
      }, 300);
    }, { capture: false });

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

  // processing
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

    //  link
    const newTab = $('viewerOpenNewTab');
    if (newTab) newTab.href = `viewer/${jobId}`;

    //  geojson
    const geoLink = $('viewerDownloadGeoJson');
    if (geoLink) {
      if (result?.geojson) {
        geoLink.href = `outputs/${jobId}/${result.geojson}`;
        geoLink.classList.remove('hidden');
      } else {
        geoLink.classList.add('hidden');
      }
    }

    // geotiff
    const tiffLink = $('viewerDownloadGeoTiff');
    if (tiffLink) {
      if (result?.raster) {
        tiffLink.href = `outputs/${jobId}/${result.raster}`;
        tiffLink.classList.remove('hidden');
      } else {
        tiffLink.classList.add('hidden');
      }
    }

    // grid csv
    const gridCsvLink = $('viewerDownloadGridCsv');
    if (gridCsvLink) {
      if (result?.geojson) {
        gridCsvLink.href = `grid_csv/${jobId}`;
        gridCsvLink.classList.remove('hidden');
      } else {
        gridCsvLink.classList.add('hidden');
      }
    }

    // custom grid shapefile upload button
    const customGridBtn = $('viewerUploadGridShapefile');
    if (customGridBtn) {
      customGridBtn.onclick = () => showGridShapefileModal(jobId);
    }

    // summary
    const summary = $('resultsSummaryBody');
    if (summary) {
      const rows = [
        ['Points Processed', result?.n_points],
        ['Groups Found', result?.n_groups],
        ['Cell Size', result?.cell_size_m ? `${Number(result.cell_size_m).toFixed(1)} m` : '—'],
        ['Hail Size Range', (Number.isFinite(result?.hail_min) && Number.isFinite(result?.hail_max))
          ? `${Number(result.hail_min).toFixed(2)} – ${Number(result.hail_max).toFixed(2)}` : '—'],
        ['Processing Time', result?.processing_seconds ? `${result.processing_seconds} seconds` : '—']
      ];

      summary.innerHTML = rows.map(([k, v]) => `
        <tr>
          <td>${escapeHtml(k)}</td>
          <td>${escapeHtml(v ?? '—')}</td>
        </tr>
      `).join('');
    }
  }

  function revealViewer(jobId, eventName, result) {
    const section = $('viewerSection');
    ui()?.reveal?.(section);
    $('viewerTitle').textContent = eventName || 'Interactive Viewer';

    const bs = $('basemapSelect');
    if (bs && !bs.value) bs.value = 'carto_light';

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
      qc_col: $('qcCol').value || null,
      event_name: $('eventName').value,
      proxy_hail_size: $('proxyHailSize').value ? Number($('proxyHailSize').value) : null,
      grouping_threshold_km: Number($('groupingThreshold').value),
      large_buffer_km: Number($('largeBuffer').value),
      small_buffer_km: Number($('smallBuffer').value),
      idw_power: $('idwPower').value ? Number($('idwPower').value) : 2.0
    };

    const errors = validateParams(params);
    if (errors.length) {
      ui()?.toast?.(errors[0], { type: 'error', title: 'Missing info' });
      return;
    }

    // stop existing poller
    if (currentPoller) {
      currentPoller.stop();
      currentPoller = null;
    }

    ui()?.setBusy?.($('processBtn'), true);
    showProcessingUI();

    try {
      const res = await fetch('process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server Error (${res.status}): ${text.substring(0, 100)}`);
      }

      const data = await safeJsonParse(res);

      if (!data.success) {
        throw new Error(data.error || 'Failed to start processing');
      }

      ui()?.toast?.('Processing started.', { type: 'info', title: 'Working' });

      // start polling
      currentPoller = new JobPoller(data.job_id, {
        pollInterval: 1000,
        maxRetries: 30,
        onProgress: (info) => updateProcessingUI(info.progress, info.message),
        onComplete: (data) => {
          ui()?.setBusy?.($('processBtn'), false);
          const p = $('processingPanel');
          if (p) p.style.display = 'none';

          showResults(data.job_id, data.result);
          revealViewer(data.job_id, $('eventName').value, data.result);
          ui()?.toast?.('Footprint generated successfully.', { type: 'success', title: 'Done' });
        },
        onError: (data) => {
          ui()?.setBusy?.($('processBtn'), false);
          updateProcessingUI(0, 'Error: ' + (data.error || 'Unknown error'));
          ui()?.toast?.(data.error || 'Unknown error', { type: 'error', title: 'Processing error' });
        }
      });

      currentPoller.start();

    } catch (e) {
      console.error('Processing start error:', e);
      ui()?.setBusy?.($('processBtn'), false);
      updateProcessingUI(0, e.message);
      ui()?.toast?.(e.message, { type: 'error', title: 'Processing failed' });
    }
  }

  // upload custom grid shapefile
  function showGridShapefileModal(jobId) {
    const modalHtml = `
      <div id="gridShapefileModal" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div class="bg-white rounded-2xl shadow-2xl p-6 max-w-md w-full mx-4 relative">
          <!-- Processing Overlay -->
          <div id="gridProcessingOverlay" class="absolute inset-0 z-10 bg-white/95 rounded-2xl flex-col items-center justify-center hidden">
            <div class="w-12 h-12 rounded-full border-[3px] border-blue-200 border-t-brand-blue animate-spin"></div>
            <div class="mt-4 font-semibold text-brand-dark" id="gridProcessingTitle">Processing Grid</div>
            <div class="text-sm text-gray-500 mt-1" id="gridProcessingMessage">Uploading shapefile...</div>
            <div class="w-48 h-2 bg-gray-200 rounded-full mt-4 overflow-hidden">
              <div id="gridProgressBar" class="h-full bg-brand-blue rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <div class="text-xs text-gray-400 mt-2" id="gridProgressText">0%</div>
          </div>
          
          <h3 class="text-xl font-bold mb-2 flex items-center gap-2">
            <i class="bi bi-grid-3x3 text-brand-blue"></i> Upload Grid Shapefile
          </h3>
          <p class="text-gray-600 mb-4 text-sm">
            Upload a zipped shapefile (.zip) containing your grid to calculate maximum hail size per cell.
          </p>
          <div id="gridUploadZone" class="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-brand-blue hover:bg-blue-50/50 transition-all">
            <input type="file" id="gridShapefileInput" accept=".zip,.shp,.gpkg,.geojson" class="hidden">
            <i class="bi bi-cloud-upload text-4xl text-gray-400 mb-2"></i>
            <p class="text-gray-500">Click or drag to upload shapefile (zip)</p>
          </div>
          <div id="gridUploadStatus" class="mt-3 p-3 rounded-lg text-sm hidden"></div>
          <div class="flex justify-end gap-3 mt-4">
            <button id="gridModalCancel" class="px-4 py-2 rounded-lg border border-gray-300 hover:bg-gray-100 transition-colors">
              Cancel
            </button>
            <button id="gridModalProcess" class="px-4 py-2 rounded-lg bg-brand-blue text-white hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2" disabled>
              <span id="gridProcessBtnText">Process Grid</span>
            </button>
          </div>
        </div>
      </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    const modal = $('gridShapefileModal');
    const uploadZone = $('gridUploadZone');
    const fileInput = $('gridShapefileInput');
    const statusDiv = $('gridUploadStatus');
    const processBtn = $('gridModalProcess');
    const cancelBtn = $('gridModalCancel');
    
    let selectedFile = null;
    
    // upload click
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadZone.classList.add('border-brand-blue', 'bg-blue-50');
    });
    
    uploadZone.addEventListener('dragleave', () => {
      uploadZone.classList.remove('border-brand-blue', 'bg-blue-50');
    });
    
    uploadZone.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadZone.classList.remove('border-brand-blue', 'bg-blue-50');
      if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });
    
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        handleFileSelect(fileInput.files[0]);
      }
    });
    
    function handleFileSelect(file) {
      selectedFile = file;
      uploadZone.innerHTML = `
        <i class="bi bi-file-earmark-check text-4xl text-green-500 mb-2"></i>
        <p class="text-gray-700 font-medium">${escapeHtml(file.name)}</p>
        <p class="text-gray-400 text-xs">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
      `;
      processBtn.disabled = false;
    }
    
    // cancel button
    cancelBtn.addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
    
    // process button
    processBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      
      // overlay elements
      const overlay = $('gridProcessingOverlay');
      const progressBar = $('gridProgressBar');
      const progressText = $('gridProgressText');
      const processingTitle = $('gridProcessingTitle');
      const processingMessage = $('gridProcessingMessage');
      const processBtnText = $('gridProcessBtnText');
      
      // show processing overlay
      processBtn.disabled = true;
      overlay.classList.remove('hidden');
      overlay.classList.add('flex');
      
      // progress simulation function
      let currentProgress = 0;
      const updateProgress = (progress, message) => {
        currentProgress = progress;
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
        if (message) processingMessage.textContent = message;
      };
      
      // simulate progress stages
      updateProgress(10, 'Uploading shapefile...');
      
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      try {
        // simulate upload progress
        updateProgress(30, 'Processing shapefile...');
        
        // use relative URL
        const response = await fetch(`upload_grid_shapefile/${jobId}`, { method: 'POST', body: formData});
        
        updateProgress(60, 'Performing spatial join...');
        
        // check response status first
        if (!response.ok) {
          // try to get error message from response
          const contentType = response.headers.get('content-type') || '';
          let errorMsg = `Server error: ${response.status} ${response.statusText}`;
          
          if (contentType.includes('application/json')) {
            try {
              const errorData = await response.json();
              errorMsg = errorData.error || errorMsg;
            } catch (e) { /* ignore parse error */ }
          }
          
          throw new Error(errorMsg);
        }
        
        // parse JSON response
        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
          throw new Error('Server returned non-JSON response. Please try again.');
        }
        
        const result = await response.json();
        
        updateProgress(80, 'Generating CSV output...');
        
        if (result.success) {
          updateProgress(100, 'Complete!');
          processingTitle.textContent = 'Processing Complete';
          processingMessage.innerHTML = `<span class="text-green-600"><i class="bi bi-check-circle"></i> ${escapeHtml(result.message)}</span>`;
          
          // auto-download after short delay
          setTimeout(() => {
            window.location.href = result.download_url;
            modal.remove();
          }, 1500);
        } else {
          // hide overlay and show error  
          overlay.classList.add('hidden');
          overlay.classList.remove('flex');
          statusDiv.classList.remove('hidden', 'bg-blue-100', 'text-blue-700', 'bg-green-100', 'text-green-700');
          statusDiv.classList.add('bg-red-100', 'text-red-700');
          statusDiv.innerHTML = `<i class="bi bi-x-circle"></i> ${escapeHtml(result.error || 'Unknown error')}`;
          processBtn.disabled = false;
          processBtnText.textContent = 'Process Grid';
        }
      } catch (err) {
        // hide overlay and show error
        overlay.classList.add('hidden');
        overlay.classList.remove('flex');
        statusDiv.classList.remove('hidden', 'bg-blue-100', 'text-blue-700', 'bg-green-100', 'text-green-700');
        statusDiv.classList.add('bg-red-100', 'text-red-700');
        statusDiv.innerHTML = `<i class="bi bi-x-circle"></i> ${escapeHtml(err.message)}`;
        processBtn.disabled = false;
        processBtnText.textContent = 'Process Grid';
      }
    });
  }

  // make nodule available globally
  window.showGridShapefileModal = showGridShapefileModal;

  // initialize
  function init() {
    setUploadState('idle');
    initDropzone();
    $('processBtn')?.addEventListener('click', startProcessing);

    const configSection = document.getElementById('configSection');
    if (configSection) {
      configSection.style.display = 'block';
      setTimeout(syncPreviewHeight, 100);
    }
  }

  // cleanup on page unload 
  window.addEventListener('beforeunload', () => {
    if (currentPoller) {
      currentPoller.stop();
    }
  });

  document.addEventListener('DOMContentLoaded', init);
})();