/**
 * Plant Doctor - Main JavaScript
 * Global utilities and initialization
 */

'use strict';

// Plant Doctor namespace
const PlantDoctor = {
    // localStorage key for history
    STORAGE_KEY: 'plantDoctor_history',

    // API endpoints
    API: {
        DIAGNOSE: '/api/diagnose',
        HEALTH: '/api/health',
        DISEASE: '/api/disease',
        PREVENTION: '/api/prevention'
    },

    /**
     * Initialize application
     */
    init: function() {
        console.log('Plant Doctor initialized');
    },

    /**
     * Show user-friendly error message
     * @param {string} message - Error message to display
     */
    showError: function(message) {
        console.error('Error:', message);
        // Could be enhanced with toast notifications
    },

    /**
     * Format date for display
     * @param {string} isoDate - ISO 8601 date string
     * @returns {string} Formatted date
     */
    formatDate: function(isoDate) {
        const date = new Date(isoDate);
        return date.toLocaleDateString('fr-FR', {
            day: 'numeric',
            month: 'short',
            year: 'numeric'
        });
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    PlantDoctor.init();
});
