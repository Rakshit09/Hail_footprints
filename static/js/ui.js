
(function () {
  function toast(message, opts) {
    const { type = 'info', title = '', duration = 4500 } = opts || {};
    const host = document.getElementById('toastHost') || document.body;

    const el = document.createElement('div');
    el.className = `toast ${type}`;

    const iconByType = {
      info: 'bi bi-info-circle',
      success: 'bi bi-check-circle',
      warning: 'bi bi-exclamation-triangle',
      danger: 'bi bi-x-circle',
      error: 'bi bi-x-circle'
    };

    el.innerHTML = `
      <div class="flex items-center gap-3">
        <div class="text-lg"><i class="${iconByType[type] || iconByType.info}"></i></div>
        <div class="flex-1">
          ${title ? `<div class="font-semibold">${title}</div>` : ''}
          <div class="text-sm">${message}</div>
        </div>
        <button class="ml-2 w-7 h-7 flex items-center justify-center rounded-lg hover:bg-white/20 transition" aria-label="Close">
          <i class="bi bi-x-lg text-sm"></i>
        </button>
      </div>
    `;

    const closeBtn = el.querySelector('button');
    const remove = () => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(100%)';
      setTimeout(() => el.remove(), 200);
    };
    closeBtn?.addEventListener('click', remove);

    host.appendChild(el);

    if (duration > 0) setTimeout(remove, duration);
  }

  function reveal(el) {
    if (!el) return;
    el.style.display = 'block';
    el.offsetHeight;
    el.classList.add('visible');
  }

  function collapse(el) {
    if (!el) return;
    el.classList.remove('visible');
    setTimeout(() => {
      el.style.display = 'none';
    }, 400);
  }

  function setBusy(el, isBusy) {
    if (!el) return;
    el.disabled = !!isBusy;
    el.classList.toggle('opacity-60', !!isBusy);
    el.classList.toggle('cursor-not-allowed', !!isBusy);
  }

  window.HailUI = { toast, reveal, collapse, setBusy };
})();
