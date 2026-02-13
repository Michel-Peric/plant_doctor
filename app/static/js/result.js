/**
 * Plant Doctor - Result Page Handler
 * Displays diagnosis results from sessionStorage
 */

'use strict';

(function () {
    // DOM Elements
    let resultLoading;
    let noResult;
    let resultContent;
    let resultImage;
    let statusBadge;
    let plantName;
    let diseaseName;
    let confidenceBar;
    let confidenceValue;
    let confidenceWarning;
    let diseaseDetails;
    let diseaseDescription;
    let symptomsList;
    let causesList;
    let actionsList;
    let bioTreatments;
    let chemicalTreatments;
    let followUpDays;
    let preventionSection;
    let preventionTips;
    let btnExplain;
    let heatmapOverlay;
    let heatmapControls;
    let heatmapToggle;

    /**
     * Initialize result page
     */
    function init() {
        // Get DOM elements
        resultLoading = document.getElementById('result-loading');
        noResult = document.getElementById('no-result');
        resultContent = document.getElementById('result-content');
        resultImage = document.getElementById('result-image');
        heatmapOverlay = document.getElementById('heatmap-overlay');
        heatmapControls = document.getElementById('heatmap-controls');
        heatmapToggle = document.getElementById('heatmap-toggle');

        statusBadge = document.getElementById('status-badge');
        plantName = document.getElementById('plant-name');
        diseaseName = document.getElementById('disease-name');
        confidenceBar = document.getElementById('confidence-bar');
        confidenceValue = document.getElementById('confidence-value');
        confidenceWarning = document.getElementById('confidence-warning');
        diseaseDetails = document.getElementById('disease-details');
        diseaseDescription = document.getElementById('disease-description');
        symptomsList = document.getElementById('symptoms-list');
        causesList = document.getElementById('causes-list');
        actionsList = document.getElementById('actions-list');
        bioTreatments = document.getElementById('bio-treatments');
        chemicalTreatments = document.getElementById('chemical-treatments');
        followUpDays = document.getElementById('follow-up-days');
        preventionSection = document.getElementById('prevention-section');
        preventionTips = document.getElementById('prevention-tips');
        btnExplain = document.getElementById('btn-explain');


        // Event Listeners
        if (btnExplain) {
            btnExplain.addEventListener('click', requestExplanation);
        }

        if (heatmapToggle) {
            heatmapToggle.addEventListener('change', function () {
                if (this.checked) {
                    heatmapOverlay.style.display = 'block';
                } else {
                    heatmapOverlay.style.display = 'none';
                }
            });
        }

        // Load result from sessionStorage
        loadResult();
    }

    /**
     * Load result from sessionStorage
     */
    function loadResult() {
        try {
            // Check for server-injected data first (History Details)
            if (window.serverDiagnosisData) {
                console.log("Loading data from server history");
                displayResult(window.serverDiagnosisData);
                return;
            }

            const resultData = sessionStorage.getItem('plantDoctor_result');

            if (!resultData) {
                showNoResult();
                return;
            }

            const data = JSON.parse(resultData);
            displayResult(data);

        } catch (e) {
            console.error('Error loading result:', e);
            showNoResult();
        }
    }

    /**
     * Show no result state
     */
    function showNoResult() {
        resultLoading.style.display = 'none';
        noResult.style.display = 'block';
        resultContent.style.display = 'none';
    }

    /**
     * Display the diagnosis result
     */
    /**
     * Display the diagnosis result
     */
    function displayResult(data) {
        const prediction = data.prediction;
        const diagnosis = data.diagnosis;
        const diseaseInfo = diagnosis.disease_info;

        // Hide loading, show content
        resultLoading.style.display = 'none';
        noResult.style.display = 'none';
        resultContent.style.display = 'block';

        // Display image
        if (data.imageData) {
            resultImage.src = data.imageData;
        }

        // Display status badge
        if (diseaseInfo.is_healthy) {
            statusBadge.innerHTML = '<span class="badge bg-success fs-6"><i class="bi bi-check-circle me-1"></i>Plante saine</span>';
        } else {
            statusBadge.innerHTML = '<span class="badge bg-danger fs-6"><i class="bi bi-exclamation-triangle me-1"></i>Maladie détectée</span>';
        }

        // Display plant and disease names
        plantName.textContent = diseaseInfo.plant || diagnosis.plant || 'Inconnue';
        diseaseName.textContent = diseaseInfo.disease || diagnosis.disease_name || 'Inconnue';

        // Reset Heatmap UI
        if (heatmapControls) {
            heatmapControls.style.display = 'none';
        }
        if (heatmapOverlay) {
            heatmapOverlay.style.display = 'none';
            heatmapOverlay.src = "";
        }

        // Show Explain button if disease detected
        if (btnExplain) {
            if (!diseaseInfo.is_healthy) {
                btnExplain.style.display = 'inline-block';
            } else {
                btnExplain.style.display = 'none';
            }
        }

        // Display confidence
        const confidencePercent = prediction.confidence_percent || 0;
        confidenceBar.style.width = confidencePercent + '%';
        confidenceValue.textContent = confidencePercent + '%';

        // Confidence bar color
        if (confidencePercent >= 90) {
            confidenceBar.className = 'progress-bar bg-success';
        } else if (confidencePercent >= 70) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-danger';
        }

        // Show warning if low confidence
        if (!diagnosis.is_confident) {
            const uncertaintyAlert = document.getElementById('uncertainty-alert');
            const alertConfidenceValue = document.getElementById('alert-confidence-value');

            if (uncertaintyAlert && alertConfidenceValue) {
                alertConfidenceValue.textContent = confidencePercent + '%';
                uncertaintyAlert.style.display = 'block';
            }
            if (confidenceWarning) confidenceWarning.style.display = 'none';
        } else {
            const uncertaintyAlert = document.getElementById('uncertainty-alert');
            if (uncertaintyAlert) uncertaintyAlert.style.display = 'none';
        }

        // --- NEW: Populate Tabbed Interface ---
        const resultDetails = document.getElementById('result-details');
        const healthyContent = document.getElementById('healthy-content');
        const sickPreventionContent = document.getElementById('sick-prevention-content');
        const healthyTipsList = document.getElementById('healthy-tips-list');
        const preventionTips = document.getElementById('prevention-tips');

        // Show the details card
        if (resultDetails) resultDetails.style.display = 'block';

        // TAB 1: DETAILS
        if (diseaseDescription) {
            diseaseDescription.textContent = diseaseInfo.description || 'Aucune description disponible.';
        }

        if (causesList) {
            if (diseaseInfo.causes && diseaseInfo.causes.length > 0) {
                causesList.innerHTML = diseaseInfo.causes
                    .map(function (c) { return '<li class="list-group-item bg-transparent border-0 px-0 py-1"><i class="bi bi-caret-right-fill text-secondary me-2"></i>' + escapeHtml(c) + '</li>'; })
                    .join('');
            } else {
                causesList.innerHTML = '<li class="list-group-item bg-transparent border-0 px-0 py-1 text-muted">Causes non spécifiées</li>';
            }
        }

        if (diseaseInfo.is_healthy) {
            // HEALTHY LOGIC
            // Hide Tabs related to disease (Symptoms, Treatments) could be done, 
            // but simpler: Empty them or specific message.
            if (document.getElementById('symptoms-list')) document.getElementById('symptoms-list').innerHTML = '<li class="list-group-item border-0 text-success">Aucun symptôme, plante saine.</li>';
            if (document.getElementById('bio-treatments')) document.getElementById('bio-treatments').innerHTML = '<p class="text-success">Aucun traitement nécessaire.</p>';
            if (document.getElementById('chemical-treatments')) document.getElementById('chemical-treatments').innerHTML = '';
            if (document.getElementById('actions-list')) document.getElementById('actions-list').parentElement.style.display = 'none'; // Hide urgent alert

            // Tab 4 (Prevention) - Show Healthy Content
            if (healthyContent) healthyContent.style.display = 'block';
            if (sickPreventionContent) sickPreventionContent.style.display = 'none';

            // Populate Healthy Tips
            var tips = diseaseInfo.healthy_tips || diseaseInfo.prevention || [];
            if (healthyTipsList) {
                if (tips.length > 0) {
                    healthyTipsList.innerHTML = tips
                        .map(function (tip) { return '<li class="list-group-item bg-transparent border-0 px-0"><i class="bi bi-check2-circle text-success me-2"></i>' + escapeHtml(tip) + '</li>'; })
                        .join('');
                } else {
                    healthyTipsList.innerHTML = '<li class="list-group-item bg-transparent border-0 px-0">Continuez les bonnes pratiques d\'entretien.</li>';
                }
            }
            // Activate Prevention Tab for healthy plants by default? Or keep Details. 
            // Let's keep Details active by default.

        } else {
            // SICK LOGIC
            // Tab 2: Symptoms
            if (symptomsList) {
                if (diseaseInfo.symptoms && diseaseInfo.symptoms.length > 0) {
                    symptomsList.innerHTML = diseaseInfo.symptoms
                        .map(function (s) { return '<li class="list-group-item bg-transparent border-0 px-0 py-1"><i class="bi bi-dot text-warning me-2" style="font-size: 1.5rem; vertical-align: middle;"></i>' + escapeHtml(s) + '</li>'; })
                        .join('');
                } else {
                    symptomsList.innerHTML = '<li class="list-group-item bg-transparent border-0 px-0 text-muted">Aucun symptôme spécifique</li>';
                }
            }

            // Tab 3: Treatments
            // Urgent Actions
            if (actionsList) {
                if (diseaseInfo.immediate_actions && diseaseInfo.immediate_actions.length > 0) {
                    actionsList.parentElement.style.display = 'block';
                    actionsList.innerHTML = diseaseInfo.immediate_actions
                        .map(function (a) { return '<li class="mb-1">' + escapeHtml(a) + '</li>'; })
                        .join('');
                } else {
                    actionsList.parentElement.style.display = 'none';
                }
            }

            // Bio Treatments
            if (bioTreatments) {
                if (diseaseInfo.treatments && diseaseInfo.treatments.bio && diseaseInfo.treatments.bio.length > 0) {
                    bioTreatments.innerHTML = diseaseInfo.treatments.bio
                        .map(function (t) { return formatTreatment(t, 'treatment-bio'); })
                        .join('');
                } else {
                    bioTreatments.innerHTML = '<p class="text-muted small fst-italic">Aucun traitement biologique spécifique.</p>';
                }
            }

            // Chemical Treatments
            if (chemicalTreatments) {
                if (diseaseInfo.treatments && diseaseInfo.treatments.chemical && diseaseInfo.treatments.chemical.length > 0) {
                    chemicalTreatments.innerHTML = diseaseInfo.treatments.chemical
                        .map(function (t) { return formatTreatment(t, 'treatment-chemical'); })
                        .join('');
                } else {
                    chemicalTreatments.innerHTML = '<p class="text-muted small fst-italic">Aucun traitement chimique recommandé.</p>';
                }
            }

            // Tab 4: Prevention (Sick context)
            if (healthyContent) healthyContent.style.display = 'none';
            if (sickPreventionContent) sickPreventionContent.style.display = 'block';

            if (preventionTips) {
                if (diseaseInfo.prevention && diseaseInfo.prevention.length > 0) {
                    preventionTips.innerHTML = diseaseInfo.prevention
                        .map(function (p) { return '<li class="list-group-item bg-transparent border-0 px-0 py-1"><i class="bi bi-shield-check text-success me-2"></i>' + escapeHtml(p) + '</li>'; })
                        .join('');
                } else {
                    preventionTips.innerHTML = '<li class="list-group-item bg-transparent border-0 px-0 text-muted">Aucune prévention spécifique.</li>';
                }
            }
            if (followUpDays) followUpDays.textContent = diseaseInfo.follow_up_days || 14;
        }
    }

    /**
     * Format a treatment for display
     * Handles both string format and object format for backwards compatibility
     */
    function formatTreatment(treatment, cssClass) {
        // Handle string format
        if (typeof treatment === 'string') {
            return '<div class="card mb-2 treatment-card ' + cssClass + '">' +
                '<div class="card-body py-2 px-3">' +
                '<p class="card-text mb-0 fs-6">' +
                '<i class="bi bi-check-circle-fill me-2 opacity-75"></i>' + escapeHtml(treatment) + '</p>' +
                '</div></div>';
        }

        // Handle object format (legacy)
        return '<div class="card mb-2 treatment-card ' + cssClass + '">' +
            '<div class="card-body py-2 px-3">' +
            '<h6 class="card-title mb-1 fw-bold">' + escapeHtml(treatment.name) + '</h6>' +
            '<p class="card-text small mb-1">' + escapeHtml(treatment.description) + '</p>' +
            (treatment.dosage && treatment.dosage !== 'N/A' ?
                '<small class="text-muted"><i class="bi bi-droplet me-1"></i>Dosage: ' + escapeHtml(treatment.dosage) + '</small>' : '') +
            '</div></div>';
    }

    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        if (!text) return '';
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initialize on DOM ready
    document.addEventListener('DOMContentLoaded', init);

    /**
     * Request Grad-CAM explanation
     */
    async function requestExplanation() {
        // Disable button loading state
        const originalText = btnExplain.innerHTML;
        btnExplain.disabled = true;
        btnExplain.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyse...';

        try {
            // Get original image data logic would be here
            // Since API requires file upload, we need the File object orBlob
            // Re-using the logic from upload page is complex here as we only have base64 in session
            // Workaround: Convert base64 back to blob or cache file in backend in future

            // For this implementation, we assume we need to re-send the image (Story limitation)
            // But we don't have the file object here, only base64.
            // Let's convert base64 to blob
            const resultData = JSON.parse(sessionStorage.getItem('plantDoctor_result'));
            if (!resultData || !resultData.imageData) throw new Error('No image data found');

            const blob = await (await fetch(resultData.imageData)).blob();
            const formData = new FormData();
            formData.append('image', blob, 'image.png');

            // Add class name if available to force explanation for the predicted disease
            if (resultData.prediction && resultData.prediction.class_name) {
                formData.append('class_name', resultData.prediction.class_name);
            }

            const response = await fetch('/api/explain', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Show heatmap overlay
                heatmapOverlay.src = data.heatmap;
                heatmapOverlay.style.display = 'block';

                // Show toggle controls
                if (heatmapControls) {
                    heatmapControls.style.display = 'flex';
                }

                // Reset toggle to checked
                if (heatmapToggle) {
                    heatmapToggle.checked = true;
                }

                // Hide Explain button as the toggle now controls it
                btnExplain.style.display = 'none';
            } else {
                alert('Erreur: ' + (data.error || 'Impossible de generer l\'explication'));
            }

        } catch (e) {
            console.error('Error requesting explanation:', e);
            alert('Erreur lors de la generation de l\'explication');
        } finally {
            btnExplain.disabled = false;
            btnExplain.innerHTML = originalText;
        }
    }

})();
