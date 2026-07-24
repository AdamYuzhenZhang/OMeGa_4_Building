// Browser-local JPEG export for the currently visible editor canvas.
function syncPreviewExportControl() {
  if (!exportPreviewButton) return;
  exportPreviewButton.disabled = Boolean(state.previewExportBusy);
  exportPreviewButton.textContent = state.previewExportBusy ? "Saving..." : "Save JPG";
}

function imageIsReady(image) {
  return Boolean(image && image.complete && image.naturalWidth > 0 && image.naturalHeight > 0);
}

function waitForImageReady(image, timeoutMs = 4500) {
  if (!image || imageIsReady(image)) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (ready) => {
      if (settled) return;
      settled = true;
      image.removeEventListener("load", onLoad);
      image.removeEventListener("error", onError);
      clearTimeout(timer);
      resolve(ready);
    };
    const onLoad = () => finish(true);
    const onError = () => finish(false);
    const timer = setTimeout(() => finish(imageIsReady(image)), timeoutMs);
    image.addEventListener("load", onLoad, { once: true });
    image.addEventListener("error", onError, { once: true });
  });
}

function currentPreviewImages() {
  const images = [];
  if (state.showFrameImage && state.selectedImage) images.push(state.selectedImage);
  if (state.proposalLayerOverlayImage && state.proposalLayerOverlayImage.size) {
    for (const layer of proposalLayerDrawOrder(visibleProposalLayers())) {
      const image = state.proposalLayerOverlayImage.get(layer);
      if (image) images.push(image);
    }
  } else if (state.proposalOverlayImage) {
    images.push(state.proposalOverlayImage);
  }
  if (state.showRegions && state.regionOverlayImage) images.push(state.regionOverlayImage);
  if (state.rgbdDebugOverlayImage) images.push(state.rgbdDebugOverlayImage);
  if (state.proposalSelectionOverlayImage) images.push(state.proposalSelectionOverlayImage);
  return images;
}

async function ensurePreviewExportAssets() {
  if (!state.selectedFrame) return;
  if (state.showFrameImage && !imageIsReady(state.selectedImage)) {
    loadSelectedFrameImage(state.selectedFrame);
  }
  await loadFrameProposalLayers(state.selectedFrame.id);
  await Promise.all(currentPreviewImages().map((image) => waitForImageReady(image)));
}

function previewExportLayerSuffix() {
  const layers = visibleProposalLayers();
  const parts = [];
  if (layers.length) parts.push(layers.join("-"));
  if (state.showRegions) parts.push("regions");
  if (state.proposalSelectionOverlayImage) parts.push("selection");
  if (!parts.length) return "rgb";
  return parts.join("_");
}

function safeFilenamePart(value, fallback) {
  const text = String(value || fallback || "")
    .replace(/\.[^.]+$/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return text || fallback || "preview";
}

function previewExportFilename() {
  const project = state.project || {};
  const baseline = safeFilenamePart(project.baselineName, "omega");
  const mode = safeFilenamePart(state.frameBackgroundMode || "rgb", "rgb");
  const layerText = safeFilenamePart(previewExportLayerSuffix(), "rgb");
  if (!state.selectedFrame) {
    return `${baseline}_3d_${mode}_${layerText}.jpg`;
  }
  const frameName = safeFilenamePart(state.selectedFrame.imageName, `frame_${state.selectedFrame.id}`);
  const frameId = String(state.selectedFrame.id).padStart(4, "0");
  return `${baseline}_frame_${frameId}_${frameName}_${mode}_${layerText}.jpg`;
}

function previewExportCropRect() {
  if (!state.selectedFrame || !state.camera) {
    return {
      x: 0,
      y: 0,
      width: canvas.width,
      height: canvas.height,
    };
  }
  const layout = frameLayout(canvas.width, canvas.height, state.camera);
  const x0 = clamp(Math.floor(layout.x0), 0, Math.max(0, canvas.width - 1));
  const y0 = clamp(Math.floor(layout.y0), 0, Math.max(0, canvas.height - 1));
  const x1 = clamp(Math.ceil(layout.x0 + layout.width), x0 + 1, canvas.width);
  const y1 = clamp(Math.ceil(layout.y0 + layout.height), y0 + 1, canvas.height);
  return {
    x: x0,
    y: y0,
    width: x1 - x0,
    height: y1 - y0,
  };
}

function croppedPreviewCanvas() {
  const crop = previewExportCropRect();
  const output = document.createElement("canvas");
  output.width = Math.max(1, crop.width);
  output.height = Math.max(1, crop.height);
  const outputCtx = output.getContext("2d", { alpha: false });
  outputCtx.drawImage(
    canvas,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    output.width,
    output.height,
  );
  return output;
}

function canvasToJpegBlob(sourceCanvas, quality = 0.92) {
  return new Promise((resolve, reject) => {
    sourceCanvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode preview JPEG"));
    }, "image/jpeg", quality);
  });
}

async function exportCurrentPreviewJpeg() {
  if (state.previewExportBusy) return;
  state.previewExportBusy = true;
  syncPreviewExportControl();
  const previousStatus = selectionStatusEl ? selectionStatusEl.textContent : "";
  try {
    setSelectionStatus("Preparing preview export");
    await ensurePreviewExportAssets();
    render();
    await new Promise((resolve) => requestAnimationFrame(resolve));
    const exportCanvas = croppedPreviewCanvas();
    const blob = await canvasToJpegBlob(exportCanvas);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = previewExportFilename();
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30000);
    setSelectionStatus(`Downloaded ${link.download}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message || "Preview export failed");
  } finally {
    state.previewExportBusy = false;
    syncPreviewExportControl();
    if (previousStatus && selectionStatusEl && selectionStatusEl.textContent === "Preparing preview export") {
      setSelectionStatus(previousStatus);
    }
  }
}
