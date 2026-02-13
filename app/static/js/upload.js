/**
 * Plant Doctor - Upload Handler
 * Manages image upload, preview, and drag-and-drop
 */

'use strict';

(function() {
    // Constants
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
    const ALLOWED_TYPES = ['image/jpeg', 'image/png'];

    // DOM Elements
    let uploadZone;
    let fileInput;
    let cameraInput;
    let btnUpload;
    let btnCamera;
    let btnAnalyze;
    let btnChange;
    let btnRetry;
    let imagePreview;
    let fileInfo;
    let defaultState;
    let previewState;
    let loadingState;
    let errorState;
    let errorMessage;

    // Current file
    let currentFile = null;

    /**
     * Initialize upload handler
     */
    function init() {
        // Get DOM elements
        uploadZone = document.getElementById('upload-zone');
        fileInput = document.getElementById('file-input');
        cameraInput = document.getElementById('camera-input');
        btnUpload = document.getElementById('btn-upload');
        btnCamera = document.getElementById('btn-camera');
        btnAnalyze = document.getElementById('btn-analyze');
        btnChange = document.getElementById('btn-change');
        btnRetry = document.getElementById('btn-retry');
        imagePreview = document.getElementById('image-preview');
        fileInfo = document.getElementById('file-info');
        defaultState = document.getElementById('upload-default');
        previewState = document.getElementById('upload-preview');
        loadingState = document.getElementById('upload-loading');
        errorState = document.getElementById('upload-error');
        errorMessage = document.getElementById('error-message');

        if (!uploadZone) return;

        // Bind events
        bindEvents();
    }

    /**
     * Bind all event listeners
     */
    function bindEvents() {
        // Click on upload zone
        uploadZone.addEventListener('click', function(e) {
            if (e.target === uploadZone || e.target.closest('#upload-default')) {
                fileInput.click();
            }
        });

        // Keyboard accessibility
        uploadZone.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        // File input change
        fileInput.addEventListener('change', handleFileSelect);
        cameraInput.addEventListener('change', handleFileSelect);

        // Button clicks
        if (btnUpload) btnUpload.addEventListener('click', function(e) {
            e.stopPropagation();
            fileInput.click();
        });

        if (btnCamera) btnCamera.addEventListener('click', function(e) {
            e.stopPropagation();
            cameraInput.click();
        });

        if (btnChange) btnChange.addEventListener('click', function(e) {
            e.stopPropagation();
            resetToDefault();
            fileInput.click();
        });

        if (btnRetry) btnRetry.addEventListener('click', function(e) {
            e.stopPropagation();
            resetToDefault();
        });

        if (btnAnalyze) btnAnalyze.addEventListener('click', function(e) {
            e.stopPropagation();
            analyzeImage();
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', handleDragOver);
        uploadZone.addEventListener('dragleave', handleDragLeave);
        uploadZone.addEventListener('drop', handleDrop);
    }

    /**
     * Handle file selection
     */
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files && files.length > 0) {
            processFile(files[0]);
        }
    }

    /**
     * Handle drag over
     */
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('drag-over');
    }

    /**
     * Handle drag leave
     */
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('drag-over');
    }

    /**
     * Handle file drop
     */
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            processFile(files[0]);
        }
    }

    /**
     * Process and validate file
     */
    function processFile(file) {
        // Validate type
        if (!ALLOWED_TYPES.includes(file.type)) {
            showError('Format non accepte. Utilisez JPG ou PNG.');
            return;
        }

        // Validate size
        if (file.size > MAX_FILE_SIZE) {
            showError('Fichier trop volumineux (max 10 Mo).');
            return;
        }

        // Store file and show preview
        currentFile = file;
        showPreview(file);
    }

    /**
     * Show image preview
     */
    function showPreview(file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            // Show file info
            if (fileInfo) {
                fileInfo.textContent = file.name + ' (' + formatFileSize(file.size) + ')';
            }
            setState('preview');
        };

        reader.onerror = function() {
            showError('Impossible de lire le fichier.');
        };

        reader.readAsDataURL(file);
    }

    /**
     * Format file size for display
     */
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' o';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' Ko';
        return (bytes / (1024 * 1024)).toFixed(1) + ' Mo';
    }

    /**
     * Analyze the image
     */
    function analyzeImage() {
        if (!currentFile) {
            showError('Aucun fichier selectionne.');
            return;
        }

        setState('loading');

        // Prepare form data
        const formData = new FormData();
        formData.append('image', currentFile);

        // Send to server
        fetch(PlantDoctor.API.DIAGNOSE, {
            method: 'POST',
            body: formData
        })
        .then(function(response) {
            return response.json().then(function(data) {
                if (!response.ok) {
                    throw new Error(data.error || 'Erreur serveur');
                }
                return data;
            });
        })
        .then(function(data) {
            if (data.success) {
                // Store result and image in sessionStorage for result page
                const resultData = {
                    prediction: data.prediction,
                    diagnosis: data.diagnosis,
                    imageData: imagePreview.src,
                    timestamp: new Date().toISOString()
                };
                sessionStorage.setItem('plantDoctor_result', JSON.stringify(resultData));

                // Also save to history
                saveToHistory(resultData);

                // Redirect to result page
                window.location.href = '/result';
            } else {
                throw new Error(data.error || 'Erreur lors de l\'analyse');
            }
        })
        .catch(function(error) {
            console.error('Analysis error:', error);
            showError(error.message || 'Une erreur est survenue. Veuillez reessayer.');
        });
    }

    /**
     * Save diagnosis to history
     */
    function saveToHistory(resultData) {
        try {
            const history = JSON.parse(localStorage.getItem(PlantDoctor.STORAGE_KEY) || '[]');

            const historyEntry = {
                id: Date.now().toString(),
                date: resultData.timestamp,
                plant: resultData.diagnosis.disease_info?.plant || 'Inconnue',
                disease: resultData.diagnosis.disease_info?.name || 'Inconnue',
                isHealthy: resultData.diagnosis.disease_info?.is_healthy || false,
                confidence: resultData.prediction.confidence_percent,
                classId: resultData.prediction.class_id,
                className: resultData.prediction.class_name,
                imageData: resultData.imageData
            };

            history.unshift(historyEntry);

            // Keep only last 50 entries
            if (history.length > 50) {
                history.pop();
            }

            localStorage.setItem(PlantDoctor.STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            console.error('Error saving to history:', e);
        }
    }

    /**
     * Set UI state
     */
    function setState(state) {
        // Hide all states
        defaultState.style.display = 'none';
        previewState.style.display = 'none';
        loadingState.style.display = 'none';
        errorState.style.display = 'none';

        // Remove state classes
        uploadZone.classList.remove('is-loading', 'is-error');

        // Show requested state
        switch (state) {
            case 'default':
                defaultState.style.display = 'block';
                break;
            case 'preview':
                previewState.style.display = 'block';
                break;
            case 'loading':
                loadingState.style.display = 'block';
                uploadZone.classList.add('is-loading');
                break;
            case 'error':
                errorState.style.display = 'block';
                uploadZone.classList.add('is-error');
                break;
        }
    }

    /**
     * Reset to default state
     */
    function resetToDefault() {
        currentFile = null;
        fileInput.value = '';
        cameraInput.value = '';
        imagePreview.src = '';
        if (fileInfo) fileInfo.textContent = '';
        setState('default');
    }

    /**
     * Show error message
     */
    function showError(message) {
        errorMessage.textContent = message;
        setState('error');
    }

    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', init);

})();
