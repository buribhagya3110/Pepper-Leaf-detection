document.addEventListener("DOMContentLoaded", () => {
    // ── Element refs ───────────────────────────────────
    const fileInput       = document.getElementById("file-input");
    const uploadPreview   = document.getElementById("upload-preview");
    const startCameraBtn  = document.getElementById("start-camera");
    const captureBtn      = document.getElementById("capture-photo");
    const cameraVideo     = document.getElementById("camera-video");
    const cameraCanvas    = document.getElementById("camera-canvas");
    const capturedInput   = document.getElementById("captured-image");
    const cameraPreview   = document.getElementById("camera-preview");
    const cameraStatus    = document.getElementById("camera-status");
    const cameraOverlay   = document.getElementById("camera-idle-overlay");
    const form            = document.getElementById("prediction-form");

    // Tab elements
    const tabUpload   = document.getElementById("tab-upload");
    const tabLive     = document.getElementById("tab-live");
    const panelUpload = document.getElementById("panel-upload");
    const panelLive   = document.getElementById("panel-live");

    let activeStream = null;

    // ── Camera helpers ─────────────────────────────────
    const stopCamera = () => {
        if (activeStream) {
            activeStream.getTracks().forEach(t => t.stop());
            activeStream = null;
        }
        if (cameraVideo) cameraVideo.srcObject = null;
    };

    const showLive = () => {
        cameraVideo  && cameraVideo.classList.remove("preview-hidden");
        cameraPreview && cameraPreview.classList.add("preview-hidden");
        cameraCanvas  && cameraCanvas.classList.add("preview-hidden");
        cameraOverlay && cameraOverlay.classList.add("hidden");
    };

    const showSnapshot = (src) => {
        cameraVideo   && cameraVideo.classList.add("preview-hidden");
        cameraCanvas  && cameraCanvas.classList.add("preview-hidden");
        cameraOverlay && cameraOverlay.classList.add("hidden");
        if (cameraPreview) {
            cameraPreview.src = src;
            cameraPreview.classList.remove("preview-hidden");
        }
    };

    // ── Tab switching ──────────────────────────────────
    const switchTab = (tab) => {
        if (tab === "upload") {
            tabUpload.classList.add("active");
            tabLive  .classList.remove("active");
            panelUpload.classList.remove("preview-hidden");
            panelLive  .classList.add("preview-hidden");
            stopCamera();
            // Clear captured image when switching to upload
            if (capturedInput) capturedInput.value = "";
        } else {
            tabLive  .classList.add("active");
            tabUpload.classList.remove("active");
            panelLive  .classList.remove("preview-hidden");
            panelUpload.classList.add("preview-hidden");
            // Clear file input when switching to camera
            if (fileInput) fileInput.value = "";
            if (uploadPreview) uploadPreview.classList.add("preview-hidden");
        }
    };

    if (tabUpload) tabUpload.addEventListener("click", () => switchTab("upload"));
    if (tabLive)   tabLive  .addEventListener("click", () => switchTab("live"));

    // ── File upload preview ────────────────────────────
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files && fileInput.files[0];
            if (capturedInput) capturedInput.value = "";
            if (!file || !uploadPreview) return;

            const reader = new FileReader();
            reader.onload = () => {
                uploadPreview.src = String(reader.result);
                uploadPreview.classList.remove("preview-hidden");
                // Hide dropzone inner content when preview is shown
                const inner = document.querySelector(".dropzone-inner");
                if (inner) inner.style.display = "none";
            };
            reader.readAsDataURL(file);
        });
    }

    // ── Start camera ───────────────────────────────────
    if (startCameraBtn && cameraVideo && cameraStatus) {
        startCameraBtn.addEventListener("click", async () => {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                cameraStatus.textContent = "Camera not supported.";
                return;
            }
            try {
                activeStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" },
                    audio: false
                });
                cameraVideo.srcObject = activeStream;
                showLive();
                cameraStatus.textContent = "Ready — center the leaf";
            } catch {
                cameraStatus.textContent = "Permission blocked.";
            }
        });
    }

    // ── Capture photo ──────────────────────────────────
    if (captureBtn && cameraVideo && cameraCanvas && capturedInput && cameraStatus) {
        captureBtn.addEventListener("click", () => {
            if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
                cameraStatus.textContent = "Start camera first.";
                return;
            }
            const ctx = cameraCanvas.getContext("2d");

            // Resize to max 800px on the longest side before encoding
            const MAX_PX = 800;
            const rawW = cameraVideo.videoWidth;
            const rawH = cameraVideo.videoHeight;
            const scale = Math.min(1, MAX_PX / Math.max(rawW, rawH));
            cameraCanvas.width  = Math.round(rawW * scale);
            cameraCanvas.height = Math.round(rawH * scale);
            ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

            // Use JPEG at 85% quality — keeps size ~80–150 KB vs 3–8 MB PNG
            const snapshot = cameraCanvas.toDataURL("image/jpeg", 0.85);
            capturedInput.value = snapshot;
            showSnapshot(snapshot);

            if (fileInput) fileInput.value = "";
            cameraStatus.textContent = "Captured — submit to analyze";
            stopCamera();
        });
    }

    // ── Form submit: stop camera ───────────────────────
    if (form) form.addEventListener("submit", stopCamera);
    window.addEventListener("beforeunload", stopCamera);
});
