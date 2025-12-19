// Utility functions for Hail Footprint App

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Debounce function for performance
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const mapped =
        type === 'error' ? 'danger' :
        type === 'danger' ? 'danger' :
        type === 'success' ? 'success' :
        type === 'warning' ? 'warning' :
        'info';

    if (window.HailUI?.toast) {
        window.HailUI.toast(message, { type: mapped });
        return;
    }

    // Ultra-fallback if ui.js hasn't loaded yet
    console.log(`[${mapped}] ${message}`);
}

/**
 * Validate form inputs
 */
function validateForm(formData) {
    const errors = [];
    
    if (!formData.lon_col) errors.push('Longitude column is required');
    if (!formData.lat_col) errors.push('Latitude column is required');
    if (!formData.hail_col) errors.push('Hail size column is required');
    if (!formData.event_name) errors.push('Event name is required');
    if (formData.event_name && !/^[a-zA-Z0-9_-]+$/.test(formData.event_name)) {
        errors.push('Event name can only contain letters, numbers, underscores, and hyphens');
    }
    
    return errors;
}

/**
 * Color interpolation for legends
 */
function interpolateColor(value, min, max, colorScale) {
    const normalized = (value - min) / (max - min);
    const index = Math.min(Math.floor(normalized * (colorScale.length - 1)), colorScale.length - 2);
    const t = (normalized * (colorScale.length - 1)) - index;
    
    // Simple linear interpolation between colors
    return colorScale[Math.round(index + t)];
}

// YlOrRd color scale
const YlOrRd = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'];

/**
 * Get color for hail value
 */
function getHailColor(value, min, max) {
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const index = Math.floor(normalized * (YlOrRd.length - 1));
    return YlOrRd[index];
}