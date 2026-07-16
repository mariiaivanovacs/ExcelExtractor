// Django version - Table Extraction OCR JavaScript

class OCRApp {
    constructor() {
        this.sessionId = null;
        this.ws = null;
        this.currentRotation = 0;
        this.displayRotation = 0; // For smooth animation
        this.uploadedFile = null;

        this.initElements();
        this.attachEventListeners();
    }

    initElements() {
        // Sections
        this.uploadSection = document.getElementById('upload-section');
        this.previewSection = document.getElementById('preview-section');
        this.processingSection = document.getElementById('processing-section');
        this.resultsSection = document.getElementById('results-section');
        this.errorSection = document.getElementById('error-section');

        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.browseBtn = document.getElementById('browse-btn');

        // Preview elements
        this.previewImage = document.getElementById('preview-image');
        this.rotateLeftBtn = document.getElementById('rotate-left-btn');
        this.rotateRightBtn = document.getElementById('rotate-right-btn');
        this.rotationAngle = document.getElementById('rotation-angle');
        this.startExtractionBtn = document.getElementById('start-extraction-btn');

        // Processing elements
        this.progressFill = document.getElementById('progress-fill');
        this.progressPercentage = document.getElementById('progress-percentage');
        this.statusMessages = document.getElementById('status-messages');

        // Results elements
        this.csvPreview = document.getElementById('csv-preview');
        this.resultImage = document.getElementById('result-image');
        this.downloadCsvBtn = document.getElementById('download-csv-btn');
        this.downloadImageBtn = document.getElementById('download-image-btn');
        this.newUploadBtn = document.getElementById('new-upload-btn');

        // Error elements
        this.errorMessage = document.getElementById('error-message');
        this.retryBtn = document.getElementById('retry-btn');
    }

    attachEventListeners() {
        // Upload interactions
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Preview/Rotation buttons
        this.rotateLeftBtn.addEventListener('click', () => this.rotateImage(-90));
        this.rotateRightBtn.addEventListener('click', () => this.rotateImage(90));
        this.startExtractionBtn.addEventListener('click', () => this.startExtraction());

        // Buttons
        this.newUploadBtn.addEventListener('click', () => this.resetApp());
        this.retryBtn.addEventListener('click', () => this.resetApp());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }

    async handleFileSelect(file) {
        if (!file) return;
        console.log("File selected:", file.name);

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (JPG, PNG, BMP, or TIFF)');
            return;
        }

        // Store file and show preview
        this.uploadedFile = file;
        this.showImagePreview(file);
    }

    showImagePreview(file) {
        // Read file and display preview
        const reader = new FileReader();

        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.currentRotation = 0;
            this.displayRotation = 0;
            this.updateRotationDisplay();
            this.showSection('preview');
        };

        reader.readAsDataURL(file);
    }

    rotateImage(degrees) {
        // Update logical rotation (0, 90, 180, 270)
        this.currentRotation = (this.currentRotation + degrees) % 360;
        if (this.currentRotation < 0) {
            this.currentRotation += 360;
        }

        // Update display rotation for smooth animation
        this.displayRotation += degrees;
        this.updateRotationDisplay();
    }

    updateRotationDisplay() {
        // Add CSS transition for smooth animation
        this.previewImage.style.transition = 'transform 0.3s ease-in-out';

        // Use displayRotation for continuous animation (can go beyond 360°)
        this.previewImage.style.transform = `rotate(${this.displayRotation}deg)`;

        // Show logical rotation in UI (always 0-270)
        this.rotationAngle.textContent = `${this.currentRotation}°`;
    }

    async startExtraction() {
        if (!this.uploadedFile) {
            alert('No file selected');
            return;
        }

        // Upload file with rotation info
        await this.uploadFile(this.uploadedFile);
    }

    async uploadFile(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('rotation', this.currentRotation);

            console.log("Uploading file with rotation:", this.currentRotation);

            const response = await fetch('/upload/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            this.sessionId = data.session_id;

            // Show processing section
            this.showSection('processing');

            // Connect to WebSocket
            this.connectWebSocket();

        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Failed to upload file. Please try again.');
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/process/${this.sessionId}/`;

        console.log('Connecting to:', wsUrl);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            // Start processing
            this.ws.send(JSON.stringify({
                action: 'start_processing'
            }));
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error. Please try again.');
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'progress':
                this.updateProgress(data.percentage, data.message);
                break;
            case 'complete':
                this.handleComplete(data.session_id);
                break;
            case 'error':
                this.showError(data.message);
                break;
        }
    }

    updateProgress(percentage, message) {
        // Update progress bar
        this.progressFill.style.width = `${percentage}%`;
        this.progressPercentage.textContent = `${percentage}%`;

        // Add status message
        const messageElement = document.createElement('p');
        messageElement.className = 'status-message';
        messageElement.textContent = `[${percentage}%] ${message}`;
        this.statusMessages.appendChild(messageElement);

        // Scroll to bottom
        this.statusMessages.scrollTop = this.statusMessages.scrollHeight;
    }

    async handleComplete(sessionId) {
        // Load previews
        await this.loadCsvPreview(sessionId);
        this.loadImagePreview(sessionId);

        // Setup download buttons
        this.downloadCsvBtn.onclick = () => window.open(`/download/csv/${sessionId}/`, '_blank');
        this.downloadImageBtn.onclick = () => window.open(`/download/image/${sessionId}/`, '_blank');

        // Show results
        this.showSection('results');

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
        }
    }

    async loadCsvPreview(sessionId) {
        try {
            const response = await fetch(`/preview/csv/${sessionId}/`);
            if (!response.ok) throw new Error('Failed to load CSV preview');

            const data = await response.json();
            this.displayCsvTable(data.data);
        } catch (error) {
            console.error('CSV preview error:', error);
            this.csvPreview.innerHTML = '<p class="csv-loading">Failed to load CSV preview</p>';
        }
    }

    displayCsvTable(data) {
        if (!data || data.length === 0) {
            this.csvPreview.innerHTML = '<p class="csv-loading">No data available</p>';
            return;
        }

        let html = '<table class="csv-table">';

        // Header row
        html += '<thead><tr>';
        for (let i = 0; i < data[0].length; i++) {
            html += `<th>${data[0][i]}</th>`;
        }
        html += '</tr></thead>';

        // Data rows
        html += '<tbody>';
        for (let i = 1; i < data.length; i++) {
            html += '<tr>';
            for (let j = 0; j < data[i].length; j++) {
                html += `<td>${data[i][j] || ''}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';

        this.csvPreview.innerHTML = html;
    }

    loadImagePreview(sessionId) {
        this.resultImage.src = `/preview/image/${sessionId}/`;
        this.resultImage.alt = 'Table Visualization';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.showSection('error');

        if (this.ws) {
            this.ws.close();
        }
    }

    showSection(section) {
        // Hide all sections
        this.uploadSection.style.display = 'none';
        this.previewSection.style.display = 'none';
        this.processingSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';

        // Show requested section
        switch (section) {
            case 'upload':
                this.uploadSection.style.display = 'block';
                break;
            case 'preview':
                this.previewSection.style.display = 'block';
                break;
            case 'processing':
                this.processingSection.style.display = 'block';
                break;
            case 'results':
                this.resultsSection.style.display = 'block';
                break;
            case 'error':
                this.errorSection.style.display = 'block';
                break;
        }
    }

    resetApp() {
        // Close WebSocket if open
        if (this.ws) {
            this.ws.close();
        }

        // Reset state
        this.sessionId = null;
        this.uploadedFile = null;
        this.currentRotation = 0;
        this.displayRotation = 0;

        // Reset UI
        this.progressFill.style.width = '0%';
        this.progressPercentage.textContent = '0%';
        this.statusMessages.innerHTML = '<p class="status-message">Initializing...</p>';
        this.fileInput.value = '';
        this.previewImage.src = '';
        this.previewImage.style.transition = 'none'; // Remove transition for instant reset
        this.previewImage.style.transform = 'rotate(0deg)';
        this.rotationAngle.textContent = '0°';

        // Show upload section
        this.showSection('upload');
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new OCRApp();
});
