/**
 * Plant Doctor - History Manager
 * Manages diagnostic history in localStorage
 */

'use strict';

(function() {
    // DOM Elements
    let historyList;
    let historyEmpty;

    /**
     * Initialize history page
     */
    function init() {
        historyList = document.getElementById('history-list');
        historyEmpty = document.getElementById('history-empty');

        if (!historyList) return;

        loadHistory();
    }

    /**
     * Load history from localStorage
     */
    function loadHistory() {
        const history = getHistory();

        if (history.length === 0) {
            showEmptyState();
            return;
        }

        renderHistory(history);
    }

    /**
     * Get history from localStorage
     */
    function getHistory() {
        try {
            const data = localStorage.getItem(PlantDoctor.STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Error reading history:', e);
            return [];
        }
    }

    /**
     * Save diagnosis to history
     */
    function saveDiagnosis(diagnosis) {
        const history = getHistory();

        // Add new entry at beginning
        history.unshift({
            id: 'diag_' + Date.now(),
            date: new Date().toISOString(),
            thumbnail: diagnosis.thumbnail,
            diseaseName: diagnosis.diseaseName,
            diseaseId: diagnosis.diseaseId,
            confidence: diagnosis.confidence,
            isHealthy: diagnosis.isHealthy
        });

        // Keep max 50 entries
        if (history.length > 50) {
            history.pop();
        }

        try {
            localStorage.setItem(PlantDoctor.STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            console.error('Error saving history:', e);
        }
    }

    /**
     * Delete diagnosis from history
     */
    function deleteDiagnosis(id) {
        const history = getHistory();
        const filtered = history.filter(function(item) {
            return item.id !== id;
        });

        try {
            localStorage.setItem(PlantDoctor.STORAGE_KEY, JSON.stringify(filtered));
            loadHistory(); // Refresh display
        } catch (e) {
            console.error('Error deleting from history:', e);
        }
    }

    /**
     * Render history list
     */
    function renderHistory(history) {
        historyList.innerHTML = '';
        historyEmpty.style.display = 'none';
        historyList.style.display = 'block';

        history.forEach(function(item) {
            const element = createHistoryItem(item);
            historyList.appendChild(element);
        });
    }

    /**
     * Create history item element
     */
    function createHistoryItem(item) {
        const div = document.createElement('div');
        div.className = 'card mb-3 shadow-sm';
        div.innerHTML = `
            <div class="card-body d-flex align-items-center">
                <img src="${item.thumbnail || '/static/images/placeholders/plant.png'}"
                     alt="Diagnostic"
                     class="rounded me-3"
                     style="width: 60px; height: 60px; object-fit: cover;">
                <div class="flex-grow-1">
                    <h5 class="mb-1 h6">${escapeHtml(item.diseaseName)}</h5>
                    <small class="text-muted">
                        ${PlantDoctor.formatDate(item.date)}
                    </small>
                </div>
                <div class="text-end">
                    <span class="badge ${getConfidenceBadgeClass(item.confidence)}">
                        ${Math.round(item.confidence * 100)}%
                    </span>
                    <button class="btn btn-sm btn-outline-danger ms-2"
                            onclick="HistoryManager.delete('${item.id}')"
                            aria-label="Supprimer ce diagnostic">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
        return div;
    }

    /**
     * Get badge class based on confidence
     */
    function getConfidenceBadgeClass(confidence) {
        if (confidence >= 0.9) return 'bg-success';
        if (confidence >= 0.7) return 'bg-warning text-dark';
        return 'bg-danger';
    }

    /**
     * Show empty state
     */
    function showEmptyState() {
        historyList.style.display = 'none';
        historyEmpty.style.display = 'block';
    }

    /**
     * Escape HTML for security
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Expose public methods
    window.HistoryManager = {
        save: saveDiagnosis,
        delete: function(id) {
            if (confirm('Supprimer ce diagnostic ?')) {
                deleteDiagnosis(id);
            }
        },
        getAll: getHistory
    };

    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', init);

})();
