// ==================== STATE MANAGEMENT ====================
const AppState = {
    currentFile: null,
    currentAnalysis: null,
    analysisHistory: JSON.parse(localStorage.getItem('fairframe_history')) || [],
    settings: JSON.parse(localStorage.getItem('fairframe_settings')) || {
        theme: 'dark',
        autoDelete: '24',
        defaultMode: 'standard',
        notifications: true
    },
    apiEndpoint: 'https://fair-frame-2k31.onrender.com/'
};

// ==================== DOM ELEMENTS ====================
const elements = {
    // File upload
    fileInput: document.getElementById('fileInput'),
    uploadZone: document.getElementById('uploadZone'),
    browseBtn: document.getElementById('browseBtn'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    clearFileBtn: document.getElementById('clearFileBtn'),
    filePreview: document.getElementById('filePreview'),
    previewContent: document.getElementById('previewContent'),
    
    // Pages
    uploadPage: document.getElementById('uploadPage'),
    resultsPage: document.getElementById('resultsPage'),
    historyPage: document.getElementById('historyPage'),
    settingsPage: document.getElementById('settingsPage'),
    contentArea: document.getElementById('contentArea'),
    
    // Navigation
    navUpload: document.getElementById('navUpload'),
    navHistory: document.getElementById('navHistory'),
    navExport: document.getElementById('navExport'),
    navSettings: document.getElementById('navSettings'),
    navHelp: document.getElementById('navHelp'),
    navAbout: document.getElementById('navAbout'),
    
    // Header
    pageTitle: document.getElementById('pageTitle'),
    pageSubtitle: document.getElementById('pageSubtitle'),
    themeToggle: document.getElementById('themeToggle'),
    notificationsBtn: document.getElementById('notificationsBtn'),
    fullscreenBtn: document.getElementById('fullscreenBtn'),
    newAnalysisBtn: document.getElementById('newAnalysisBtn'),
    
    // Results
    resultsContainer: document.getElementById('resultsContainer'),
    resultsSubtitle: document.getElementById('resultsSubtitle'),
    backToUploadBtn: document.getElementById('backToUploadBtn'),
    exportReportBtn: document.getElementById('exportReportBtn'),
    
    // History
    historyContainer: document.getElementById('historyContainer'),
    historyCount: document.getElementById('historyCount'),
    startFirstAnalysisBtn: document.getElementById('startFirstAnalysisBtn'),
    
    // Settings
    displayName: document.getElementById('displayName'),
    defaultMode: document.getElementById('defaultMode'),
    autoDelete: document.getElementById('autoDelete'),
    
    // Modals
    loadingModal: document.getElementById('loadingModal'),
    loadingMessage: document.getElementById('loadingMessage'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    estimatedTime: document.getElementById('estimatedTime'),
    
    // Toasts
    toastContainer: document.getElementById('toastContainer')
};

// ==================== INITIALIZATION ====================
function init() {
    console.log('🚀 FairFrame AI Initializing...');
    
    // Load saved settings
    loadSettings();
    updateHistoryCount();
    
    // Setup event listeners
    setupEventListeners();
    
    // Check API connection
    checkApiConnection();
    
    console.log('✅ FairFrame AI Ready');
    showToast('FairFrame AI ready to analyze', 'info');
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // File upload
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
    elements.clearFileBtn.addEventListener('click', clearCurrentFile);
    elements.analyzeBtn.addEventListener('click', startAnalysis);
    
    // Drag and drop
    elements.uploadZone.addEventListener('dragover', handleDragOver);
    elements.uploadZone.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone.addEventListener('drop', handleDrop);
    
    // Navigation
    elements.navUpload.addEventListener('click', (e) => {
        e.preventDefault();
        showPage('upload');
    });
    
    elements.navHistory.addEventListener('click', (e) => {
        e.preventDefault();
        showPage('history');
    });
    
    elements.navSettings.addEventListener('click', (e) => {
        e.preventDefault();
        showPage('settings');
    });
    
    elements.navHelp.addEventListener('click', (e) => {
        e.preventDefault();
        showToast('Help documentation coming soon!', 'info');
    });
    
    elements.navAbout.addEventListener('click', (e) => {
        e.preventDefault();
        showToast('FairFrame AI v2.1 - Ethical Media Analysis', 'info');
    });
    
    // Header buttons
    elements.themeToggle.addEventListener('click', toggleTheme);
    elements.notificationsBtn.addEventListener('click', () => {
        showToast('No new notifications', 'info');
    });
    
    elements.fullscreenBtn.addEventListener('click', toggleFullscreen);
    elements.newAnalysisBtn.addEventListener('click', () => showPage('upload'));
    
    // Results page
    elements.backToUploadBtn.addEventListener('click', () => showPage('upload'));
    elements.exportReportBtn.addEventListener('click', exportReport);
    
    // History page
    elements.startFirstAnalysisBtn.addEventListener('click', () => showPage('upload'));
    
    // Settings
    elements.displayName.addEventListener('change', saveSettings);
    elements.defaultMode.addEventListener('change', saveSettings);
    elements.autoDelete.addEventListener('change', saveSettings);
    
    // Theme options
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.addEventListener('click', function() {
            const theme = this.dataset.theme;
            setTheme(theme);
        });
    });
    
    // Upgrade button
    document.getElementById('upgradeBtn')?.addEventListener('click', () => {
        showToast('Premium features coming soon!', 'info');
    });
    
    // Cancel button
    document.getElementById('cancelBtn')?.addEventListener('click', clearCurrentFile);
}

// ==================== FILE HANDLING ====================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (validateFile(file)) {
        AppState.currentFile = file;
        showFilePreview(file);
        elements.analyzeBtn.disabled = false;
        showToast(`Selected: ${file.name}`, 'info');
    }
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadZone.classList.add('dragover');
}

function handleDragLeave() {
    elements.uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
        AppState.currentFile = file;
        showFilePreview(file);
        elements.analyzeBtn.disabled = false;
        showToast(`Dropped: ${file.name}`, 'info');
    }
}

function validateFile(file) {
    const validTypes = [
        'image/jpeg', 'image/jpg', 'image/png',
        'video/mp4', 'video/quicktime', 'video/x-msvideo',
        'audio/mpeg', 'audio/wav'
    ];
    
    // Reduce max size for Render free tier
    const maxSize = 50 * 1024 * 1024; // 50MB (reduced from 500MB)
    
    if (!validTypes.includes(file.type)) {
        showToast(`Invalid file type: ${file.type}`, 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showToast(`File too large (max 50MB). Render free tier limitation.`, 'error');
        return false;
    }
    
    return true;
}

function showFilePreview(file) {
    const fileType = file.type.split('/')[0];
    const fileSize = formatFileSize(file.size);
    
    let icon = 'fa-file';
    if (fileType === 'image') icon = 'fa-image';
    else if (fileType === 'video') icon = 'fa-video';
    else if (fileType === 'audio') icon = 'fa-file-audio';
    
    elements.previewContent.innerHTML = `
        <div class="file-icon">
            <i class="fas ${icon}"></i>
        </div>
        <div class="file-details">
            <h5>${file.name}</h5>
            <p>${fileType.toUpperCase()} • ${fileSize}</p>
            <p>Uploaded: ${new Date().toLocaleDateString()}</p>
        </div>
    `;
    
    elements.filePreview.style.display = 'block';
    elements.uploadZone.style.display = 'none';
}

function clearCurrentFile() {
    AppState.currentFile = null;
    elements.fileInput.value = '';
    elements.filePreview.style.display = 'none';
    elements.uploadZone.style.display = 'block';
    elements.analyzeBtn.disabled = true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ==================== ANALYSIS ====================
async function startAnalysis() {
    if (!AppState.currentFile) {
        showToast('Please select a file first', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', AppState.currentFile);
        
        // Show special message for Render free tier
        showToast('Note: First request may take up to 50 seconds (Render free tier)', 'info');
        
        // Add timeout for Render free tier
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout
        
        const response = await fetch(AppState.apiEndpoint, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const results = await response.json();
        
        if (results.success) {
            AppState.currentAnalysis = results;
            saveToHistory(results);
            hideLoading();
            showResults(results);
            showPage('results');
            showToast('Analysis completed successfully!', 'success');
        } else {
            throw new Error(results.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        hideLoading();
        
        // Special handling for Render timeout
        if (error.name === 'AbortError') {
            showToast('Request timed out. Render free tier may be spinning up. Try again in 30 seconds.', 'error');
        } else {
            showToast(`Analysis failed: ${error.message}`, 'error');
        }
    }
}

function showLoading() {
    elements.loadingMessage.textContent = 'Initializing FairFrame AI...';
    elements.progressBar.style.width = '0%';
    elements.progressText.textContent = '0%';
    elements.loadingModal.style.display = 'flex';
    
    // Simulate progress
    const steps = [
        { percent: 15, message: 'Validating file format...' },
        { percent: 30, message: 'Extracting media content...' },
        { percent: 50, message: 'Analyzing with FairFrame AI...' },
        { percent: 75, message: 'Processing results...' },
        { percent: 90, message: 'Generating insights...' }
    ];
    
    let currentStep = 0;
    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            const step = steps[currentStep];
            elements.progressBar.style.width = step.percent + '%';
            elements.progressText.textContent = step.percent + '%';
            elements.loadingMessage.textContent = step.message;
            currentStep++;
        } else {
            clearInterval(interval);
            elements.progressBar.style.width = '100%';
            elements.progressText.textContent = '100%';
            elements.loadingMessage.textContent = 'Finalizing analysis...';
        }
    }, 500);
    
    // Update estimated time
    const fileSize = AppState.currentFile?.size || 0;
    const estimatedSeconds = Math.max(5, Math.min(30, fileSize / (5 * 1024 * 1024)));
    elements.estimatedTime.textContent = `Estimated: ${Math.round(estimatedSeconds)} seconds`;
}

function hideLoading() {
    elements.loadingModal.style.display = 'none';
}

// ==================== RESULTS DISPLAY ====================
function showResults(data) {
    const analysis = data.analysis;
    const details = analysis.details || {};
    
    // Update page title
    elements.pageTitle.textContent = 'Analysis Results';
    elements.resultsSubtitle.textContent = `Analysis of ${data.filename}`;
    
    // Get colors based on scores
    const getBiasColor = (score) => {
        if (score < 20) return '#10b981';
        if (score < 50) return '#f59e0b';
        return '#ef4444';
    };
    
    const getInterpretation = (score) => {
        if (score < 20) return 'Minimal bias detected';
        if (score < 50) return 'Moderate bias detected';
        return 'Significant bias detected';
    };
    
    const biasColor = getBiasColor(analysis.bias_score);
    
    // Build results HTML
    elements.resultsContainer.innerHTML = `
        <div class="results-grid">
            <!-- Main Results -->
            <div class="main-results">
                <!-- Bias Score Card -->
                <div class="bias-score-card">
                    <div class="bias-score" style="color: ${biasColor}">
                        ${analysis.bias_score.toFixed(1)}%
                    </div>
                    <div class="bias-label">Overall Bias Score</div>
                    <div class="meter-container">
                        <div class="meter">
                            <div class="meter-fill" style="width: ${analysis.bias_score}%; background: ${biasColor}"></div>
                        </div>
                        <div class="meter-labels">
                            <span>Low Bias</span>
                            <span>High Bias</span>
                        </div>
                    </div>
                    <p style="color: var(--text-secondary); margin-top: 20px; font-size: 16px;">
                        ${getInterpretation(analysis.bias_score)}
                    </p>
                </div>
                
                <!-- Executive Summary -->
                <div class="analysis-section">
                    <h4><i class="fas fa-file-alt"></i> Executive Summary</h4>
                    <p>${analysis.summary}</p>
                    
                    <div class="details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Sentiment</div>
                            <div class="detail-value">${analysis.sentiment}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Confidence</div>
                            <div class="detail-value" style="color: ${analysis.confidence > 85 ? '#10b981' : analysis.confidence > 70 ? '#f59e0b' : '#ef4444'}">
                                ${analysis.confidence}%
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Credibility</div>
                            <div class="detail-value" style="color: ${details.credibility_score > 80 ? '#10b981' : details.credibility_score > 60 ? '#f59e0b' : '#ef4444'}">
                                ${details.credibility_score}%
                            </div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Analysis Model</div>
                            <div class="detail-value">${details.analysis_model}</div>
                        </div>
                    </div>
                </div>
                
                <!-- Recommendations -->
                <div class="analysis-section">
                    <h4><i class="fas fa-lightbulb"></i> Recommendations</h4>
                    <div class="recommendations-list">
                        ${analysis.recommendations.map(rec => `
                            <div class="recommendation-item">
                                <i class="fas fa-check-circle"></i>
                                <span>${rec}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
            
            <!-- Side Results -->
            <div class="side-results">
                <!-- Stats -->
                <div class="analysis-section">
                    <h4><i class="fas fa-chart-pie"></i> Statistics</h4>
                    <div class="stats-grid-small">
                        <div class="stat-card-small">
                            <div class="stat-value-small" style="color: #10b981">
                                ${analysis.neutral_content.toFixed(1)}%
                            </div>
                            <div class="stat-label-small">Neutral Content</div>
                        </div>
                        <div class="stat-card-small">
                            <div class="stat-value-small" style="color: #f59e0b">
                                ${analysis.potential_bias.toFixed(1)}%
                            </div>
                            <div class="stat-label-small">Potential Bias</div>
                        </div>
                        <div class="stat-card-small">
                            <div class="stat-value-small" style="color: ${getBiasColor(analysis.strong_bias)}">
                                ${analysis.strong_bias.toFixed(1)}%
                            </div>
                            <div class="stat-label-small">Strong Bias</div>
                        </div>
                    </div>
                </div>
                
                <!-- File Details -->
                <div class="analysis-section">
                    <h4><i class="fas fa-info-circle"></i> File Details</h4>
                    <div class="details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Filename</div>
                            <div class="detail-value">${data.filename}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Type</div>
                            <div class="detail-value">${data.file_type}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Size</div>
                            <div class="detail-value">${formatFileSize(data.file_size)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Analysis ID</div>
                            <div class="detail-value">${data.analysis_id}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Processed</div>
                            <div class="detail-value">${new Date(data.timestamp).toLocaleString()}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Processing Time</div>
                            <div class="detail-value">${data.processing_time}s</div>
                        </div>
                    </div>
                </div>
                
                <!-- Actions -->
                <div class="analysis-section">
                    <h4><i class="fas fa-cogs"></i> Actions</h4>
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <button class="btn-secondary" onclick="reanalyzeFile()" style="width: 100%;">
                            <i class="fas fa-redo"></i> Re-analyze File
                        </button>
                        <button class="btn-secondary" onclick="shareAnalysis()" style="width: 100%;">
                            <i class="fas fa-share-alt"></i> Share Results
                        </button>
                        <button class="btn-primary" onclick="exportReport()" style="width: 100%;">
                            <i class="fas fa-download"></i> Export Full Report
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ==================== PAGE MANAGEMENT ====================
function showPage(page) {
    // Hide all pages
    elements.uploadPage.classList.remove('active');
    elements.resultsPage.style.display = 'none';
    elements.historyPage.style.display = 'none';
    elements.settingsPage.style.display = 'none';
    
    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected page
    switch(page) {
        case 'upload':
            elements.uploadPage.classList.add('active');
            elements.navUpload.classList.add('active');
            elements.pageTitle.textContent = 'Media Analysis';
            elements.pageSubtitle.textContent = 'Upload and analyze media files for bias detection';
            break;
            
        case 'results':
            elements.resultsPage.style.display = 'block';
            elements.pageTitle.textContent = 'Analysis Results';
            elements.pageSubtitle.textContent = 'Detailed analysis and insights';
            break;
            
        case 'history':
            elements.historyPage.style.display = 'block';
            elements.navHistory.classList.add('active');
            elements.pageTitle.textContent = 'Analysis History';
            elements.pageSubtitle.textContent = 'Review previous analyses';
            loadHistory();
            break;
            
        case 'settings':
            elements.settingsPage.style.display = 'block';
            elements.navSettings.classList.add('active');
            elements.pageTitle.textContent = 'Settings';
            elements.pageSubtitle.textContent = 'Configure FairFrame preferences';
            break;
    }
}

// ==================== HISTORY MANAGEMENT ====================
function saveToHistory(analysis) {
    // Keep only last 50 analyses
    AppState.analysisHistory.unshift({
        id: analysis.analysis_id,
        filename: analysis.filename,
        timestamp: analysis.timestamp,
        bias_score: analysis.analysis.bias_score,
        sentiment: analysis.analysis.sentiment
    });
    
    if (AppState.analysisHistory.length > 50) {
        AppState.analysisHistory = AppState.analysisHistory.slice(0, 50);
    }
    
    localStorage.setItem('fairframe_history', JSON.stringify(AppState.analysisHistory));
    updateHistoryCount();
}

function loadHistory() {
    if (AppState.analysisHistory.length === 0) {
        elements.historyContainer.innerHTML = `
            <div class="empty-history">
                <i class="fas fa-history"></i>
                <h4>No analysis history yet</h4>
                <p>Analyze your first file to see it here</p>
                <button class="btn-primary" onclick="showPage('upload')">
                    <i class="fas fa-play-circle"></i> Start First Analysis
                </button>
            </div>
        `;
        return;
    }
    
    const historyHTML = AppState.analysisHistory.map(item => `
        <div class="history-item" data-id="${item.id}">
            <div class="history-icon">
                <i class="fas fa-file"></i>
            </div>
            <div class="history-info">
                <h5>${item.filename}</h5>
                <p>${new Date(item.timestamp).toLocaleString()}</p>
            </div>
            <div class="history-stats">
                <span class="bias-badge" style="background: ${item.bias_score < 30 ? '#10b981' : item.bias_score < 70 ? '#f59e0b' : '#ef4444'}">
                    ${item.bias_score}% bias
                </span>
                <span class="sentiment-badge">${item.sentiment}</span>
            </div>
            <div class="history-actions">
                <button class="btn-icon" onclick="viewHistoryItem('${item.id}')" title="View">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon" onclick="deleteHistoryItem('${item.id}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
    
    elements.historyContainer.innerHTML = `
        <div class="history-header">
            <h4>Recent Analyses (${AppState.analysisHistory.length})</h4>
            <button class="btn-secondary" onclick="clearHistory()">
                <i class="fas fa-trash"></i> Clear All
            </button>
        </div>
        <div class="history-list">
            ${historyHTML}
        </div>
    `;
}

function updateHistoryCount() {
    elements.historyCount.textContent = AppState.analysisHistory.length;
}

function viewHistoryItem(id) {
    showToast('Viewing historical analysis', 'info');
    // In a real app, you would load the full analysis data here
}

function deleteHistoryItem(id) {
    AppState.analysisHistory = AppState.analysisHistory.filter(item => item.id !== id);
    localStorage.setItem('fairframe_history', JSON.stringify(AppState.analysisHistory));
    loadHistory();
    updateHistoryCount();
    showToast('Analysis removed from history', 'success');
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all analysis history?')) {
        AppState.analysisHistory = [];
        localStorage.removeItem('fairframe_history');
        loadHistory();
        updateHistoryCount();
        showToast('History cleared', 'success');
    }
}

// ==================== SETTINGS ====================
function loadSettings() {
    // Apply theme
    setTheme(AppState.settings.theme);
    
    // Load values
    if (elements.displayName) {
        elements.displayName.value = AppState.settings.displayName || 'Guest User';
    }
    
    if (elements.defaultMode) {
        elements.defaultMode.value = AppState.settings.defaultMode || 'standard';
    }
    
    if (elements.autoDelete) {
        elements.autoDelete.value = AppState.settings.autoDelete || '24';
    }
    
    // Update theme buttons
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === AppState.settings.theme);
    });
}

function saveSettings() {
    AppState.settings = {
        ...AppState.settings,
        displayName: elements.displayName?.value || 'Guest User',
        defaultMode: elements.defaultMode?.value || 'standard',
        autoDelete: elements.autoDelete?.value || '24'
    };
    
    localStorage.setItem('fairframe_settings', JSON.stringify(AppState.settings));
    showToast('Settings saved', 'success');
}

function setTheme(theme) {
    AppState.settings.theme = theme;
    localStorage.setItem('fairframe_settings', JSON.stringify(AppState.settings));
    
    // Update theme button
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === theme);
    });
    
    // Update theme toggle icon
    const icon = elements.themeToggle.querySelector('i');
    if (theme === 'dark' || theme === 'auto') {
        icon.className = 'fas fa-moon';
    } else {
        icon.className = 'fas fa-sun';
    }
    
    showToast(`Theme set to ${theme}`, 'info');
}

function toggleTheme() {
    const current = AppState.settings.theme;
    const next = current === 'dark' ? 'light' : 'dark';
    setTheme(next);
}

// ==================== EXPORT & SHARE ====================
function exportReport() {
    if (!AppState.currentAnalysis) {
        showToast('No analysis to export', 'error');
        return;
    }
    
    try {
        // Create a downloadable JSON file
        const dataStr = JSON.stringify(AppState.currentAnalysis, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `fairframe-analysis-${AppState.currentAnalysis.analysis_id}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        showToast('Report exported as JSON', 'success');
        
    } catch (error) {
        console.error('Export error:', error);
        showToast('Failed to export report', 'error');
    }
}

function shareAnalysis() {
    if (!AppState.currentAnalysis) {
        showToast('No analysis to share', 'error');
        return;
    }
    
    if (navigator.share) {
        navigator.share({
            title: `FairFrame Analysis: ${AppState.currentAnalysis.filename}`,
            text: `Bias score: ${AppState.currentAnalysis.analysis.bias_score}% - ${AppState.currentAnalysis.analysis.summary}`,
            url: window.location.href
        })
        .then(() => showToast('Analysis shared successfully', 'success'))
        .catch(error => showToast('Share cancelled', 'info'));
    } else {
        showToast('Web Share API not supported in this browser', 'error');
    }
}

// ==================== UTILITIES ====================
function reanalyzeFile() {
    if (AppState.currentFile) {
        startAnalysis();
    } else {
        showToast('No file to re-analyze', 'error');
    }
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            showToast(`Error attempting to enable fullscreen: ${err.message}`, 'error');
        });
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

async function checkApiConnection() {
    try {
        // Check health endpoint instead of root
        const response = await fetch(AppState.apiEndpoint.replace('/api/analyze', '/health'));
        if (response.ok) {
            console.log('✅ API connected');
            document.getElementById('apiStatus').textContent = 'API Connected';
            document.getElementById('apiStatus').style.color = '#10b981';
        }
    } catch (error) {
        console.log('⚠️ API not connected');
        document.getElementById('apiStatus').textContent = 'API Disconnected';
        document.getElementById('apiStatus').style.color = '#ef4444';
        showToast('Backend API not connected. First request may take 50 seconds.', 'warning');
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Remove toast after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 5000);
}

// ==================== START APPLICATION ====================
// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);

// Expose some functions to global scope for HTML onclick handlers
window.reanalyzeFile = reanalyzeFile;
window.exportReport = exportReport;
window.shareAnalysis = shareAnalysis;
window.showPage = showPage;