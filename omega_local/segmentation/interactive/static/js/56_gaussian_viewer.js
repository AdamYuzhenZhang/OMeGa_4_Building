// Camera-aligned SuperSplat bridge. The editor owns all camera interaction.
const GAUSSIAN_BRIDGE_SOURCE = "omega-splat-embed";
const GAUSSIAN_CAMERA_MESSAGE = "omega-splat-camera";
const GAUSSIAN_VISIBILITY_MESSAGE = "omega-splat-visibility";
const GAUSSIAN_COLOR_MODE_MESSAGE = "omega-splat-color-mode";
const GAUSSIAN_VIEWER_STATUS_URL = "/api/3d-segmentation/viewer/status";
const GAUSSIAN_VIEWER_SETTINGS_URL = "/api/3d-segmentation/viewer/settings.json";
let gaussianViewportCamera = null;
let gaussianViewportLoadSequence = 0;
let gaussianViewportSessionId = "";

function gaussianArtifact(runId, variantId) {
  const run = savedSegmentation3dRun(runId);
  const variant = segmentation3dGaussianArtifacts(run).find(
    (row) => String(row.variantId) === String(variantId)
  );
  return run && variant ? { run, variant } : null;
}

function gaussianArtifactContentUrl(runId, variantId, contentVersion = "") {
  const base = `/api/3d-segmentation/runs/${encodeURIComponent(runId)}` +
    `/gaussians/${encodeURIComponent(variantId)}.ply`;
  return datasetUrl(contentVersion ? `${base}?v=${encodeURIComponent(contentVersion)}` : base);
}

function gaussianSceneManifestUrl(runId, variantId) {
  return datasetUrl(
    `/api/3d-segmentation/runs/${encodeURIComponent(runId)}` +
    `/gaussian-scenes/${encodeURIComponent(variantId)}.json`
  );
}

function gaussianSceneKey(scene) {
  const versions = (scene.parts || [])
    .map((part) => `${part.variantId}:${part.contentVersion || ""}`)
    .join("|");
  return `${scene.runId}:${scene.sceneId}:${versions}`;
}

async function requireGaussianViewer() {
  const status = await loadJson(GAUSSIAN_VIEWER_STATUS_URL);
  if (!status.available) {
    throw new Error(status.message || "Gaussian viewer is not available.");
  }
}

function gaussianViewportVisible() {
  return Boolean(
    state.gaussianViewportVisible &&
    gaussianViewportFrame &&
    !gaussianViewportFrame.hidden
  );
}

function gaussianPartSceneActive() {
  const scene = state.gaussianViewportScene;
  return Boolean(
    gaussianViewportVisible() &&
    scene &&
    Array.isArray(scene.parts) &&
    (
      scene.parts.length > 1 ||
      (Array.isArray(scene.colorModes) && scene.colorModes.length > 1)
    )
  );
}

function hidePointRenderersForGaussian() {
  state.showPointCloud = false;
  if (showPointCloudInput) showPointCloudInput.checked = false;
  for (const layer of Object.values(state.evidencePointClouds)) {
    layer.visible = false;
  }
  syncPointCloudSourceControls();
  syncSegmentation3dResultList();
}

function initializeGaussianPartState(scene, preserve = false) {
  const previous = preserve ? state.gaussianViewportPartVisibility : {};
  state.gaussianViewportPartVisibility = {};
  state.gaussianViewportPartProgress = {};
  for (const part of scene.parts || []) {
    const partId = String(part.partId || part.variantId);
    state.gaussianViewportPartVisibility[partId] = Object.hasOwn(previous, partId)
      ? Boolean(previous[partId])
      : true;
    state.gaussianViewportPartProgress[partId] = 0;
  }
}

async function activateGaussianViewport(runId, variantId) {
  if (!gaussianViewportFrame) throw new Error("The in-place Gaussian viewport is unavailable.");
  if (!gaussianArtifact(runId, variantId)) {
    throw new Error(`Gaussian artifact ${runId}/${variantId} is not registered.`);
  }
  await requireGaussianViewer();
  const scene = await loadJson(gaussianSceneManifestUrl(runId, variantId));
  if (!Array.isArray(scene.parts) || !scene.parts.length) {
    throw new Error(`Gaussian scene ${runId}/${variantId} contains no renderable parts.`);
  }

  hidePointRenderersForGaussian();
  const key = gaussianSceneKey(scene);
  const preserve = state.gaussianViewportLoadedKey === key;
  const colorModeIds = new Set(
    (Array.isArray(scene.colorModes) ? scene.colorModes : [])
      .map((row) => String(row.modeId || ""))
  );
  if (!preserve || !colorModeIds.has(state.gaussianViewportColorMode)) {
    state.gaussianViewportColorMode = String(scene.defaultColorMode || "rgb");
  }
  state.gaussianViewportRunId = String(runId);
  state.gaussianViewportVariantId = String(variantId);
  state.gaussianViewportScene = scene;
  initializeGaussianPartState(scene, preserve);
  state.gaussianViewportVisible = true;
  state.gaussianViewportError = "";
  gaussianViewportFrame.hidden = false;
  gaussianViewportFrame.setAttribute("aria-hidden", "false");

  if (!preserve) {
    gaussianViewportLoadSequence += 1;
    gaussianViewportSessionId = `${runId}:${variantId}:${gaussianViewportLoadSequence}`;
    state.gaussianViewportLoadedKey = key;
    state.gaussianViewportReady = false;
    state.gaussianViewportColorSwitching = false;
    state.gaussianViewportColorProgress = 100;
    state.gaussianViewportProgress = 0;
    gaussianViewportFrame.setAttribute("aria-busy", "true");
    gaussianViewportFrame.src = "/static/splat_embed.html" +
      `?scene=${encodeURIComponent(gaussianSceneManifestUrl(runId, variantId))}` +
      `&settings=${encodeURIComponent(datasetUrl(GAUSSIAN_VIEWER_SETTINGS_URL))}` +
      `&datasetId=${encodeURIComponent(activeDatasetId())}` +
      `&session=${encodeURIComponent(gaussianViewportSessionId)}`;
  } else {
    sendGaussianViewportCamera();
    sendAllGaussianPartVisibility();
    sendGaussianColorMode();
  }
  syncReconstructionResultLists();
  renderIdPanel();
  render();
}

function deactivateGaussianViewport({ redraw = true } = {}) {
  const wasVisible = gaussianViewportVisible();
  state.gaussianViewportVisible = false;
  state.gaussianViewportRunId = "";
  state.gaussianViewportVariantId = "";
  state.gaussianViewportScene = null;
  if (gaussianViewportFrame) {
    gaussianViewportFrame.hidden = true;
    gaussianViewportFrame.setAttribute("aria-hidden", "true");
  }
  syncReconstructionResultLists();
  renderIdPanel();
  if (redraw && wasVisible) render();
}

function gaussianViewportCameraPayload(camera, layout) {
  const bounds = state.bounds || { center: [0, 0, 0], radius: 10 };
  const radius = Math.max(Number(bounds.radius) || 10, 0.1);
  const centerDepth = dot(sub(bounds.center, camera.eye), camera.forward);
  const width = Math.max(canvas.width, 1);
  const height = Math.max(canvas.height, 1);
  const scale = Math.max(Number(layout.scale) || 1, 1e-6);
  return {
    eye: camera.eye.slice(),
    forward: camera.forward.slice(),
    down: camera.down.slice(),
    intrinsics: {
      // Express the calibrated image camera in full-canvas coordinates. This
      // keeps splats aligned with the letterboxed RGB image while allowing
      // geometry outside the captured image bounds to remain visible.
      fx: camera.fx * scale,
      fy: camera.fy * scale,
      cx: layout.x0 + camera.cx * scale,
      cy: layout.y0 + camera.cy * scale,
      width,
      height,
    },
    viewport: {
      x: 0,
      y: 0,
      width: 1,
      height: 1,
    },
    near: 0.01,
    far: Math.max(100, centerDepth + radius * 4, radius * 20),
  };
}

function sendGaussianViewportCamera() {
  if (!gaussianViewportVisible() || !gaussianViewportCamera || !gaussianViewportSessionId) return;
  const target = gaussianViewportFrame.contentWindow;
  if (!target) return;
  target.postMessage(
    {
      type: GAUSSIAN_CAMERA_MESSAGE,
      sessionId: gaussianViewportSessionId,
      camera: gaussianViewportCamera,
    },
    location.origin,
  );
}

function sendGaussianPartVisibility(partId, visible) {
  if (!gaussianViewportVisible() || !gaussianViewportSessionId) return;
  const target = gaussianViewportFrame.contentWindow;
  if (!target) return;
  target.postMessage(
    {
      type: GAUSSIAN_VISIBILITY_MESSAGE,
      sessionId: gaussianViewportSessionId,
      partId: String(partId),
      visible: Boolean(visible),
    },
    location.origin,
  );
}

function sendAllGaussianPartVisibility() {
  for (const [partId, visible] of Object.entries(state.gaussianViewportPartVisibility)) {
    sendGaussianPartVisibility(partId, visible);
  }
}
function sendGaussianColorMode() {
  if (!gaussianViewportVisible() || !gaussianViewportSessionId) return;
  const target = gaussianViewportFrame.contentWindow;
  if (!target) return;
  target.postMessage(
    {
      type: GAUSSIAN_COLOR_MODE_MESSAGE,
      sessionId: gaussianViewportSessionId,
      mode: state.gaussianViewportColorMode,
    },
    location.origin,
  );
}

function setGaussianColorMode(modeId) {
  const scene = state.gaussianViewportScene;
  const allowed = new Set(
    (Array.isArray(scene?.colorModes) ? scene.colorModes : [])
      .map((row) => String(row.modeId || ""))
  );
  const mode = String(modeId || "");
  if (!allowed.has(mode) || state.gaussianViewportColorMode === mode) return;
  state.gaussianViewportColorMode = mode;
  state.gaussianViewportColorSwitching = true;
  state.gaussianViewportColorProgress = 0;
  sendGaussianColorMode();
  renderIdPanel();
}


function setGaussianPartVisibility(partId, visible) {
  const key = String(partId);
  if (!Object.hasOwn(state.gaussianViewportPartVisibility, key)) return;
  state.gaussianViewportPartVisibility[key] = Boolean(visible);
  sendGaussianPartVisibility(key, visible);
  renderIdPanel();
}

function setAllGaussianPartsVisible(visible) {
  for (const partId of Object.keys(state.gaussianViewportPartVisibility)) {
    state.gaussianViewportPartVisibility[partId] = Boolean(visible);
  }
  sendAllGaussianPartVisibility();
  renderIdPanel();
}

function syncGaussianViewportCamera(camera, layout) {
  gaussianViewportCamera = gaussianViewportCameraPayload(camera, layout);
  if (gaussianViewportFrame) {
    gaussianViewportFrame.style.clipPath = "none";
  }
  sendGaussianViewportCamera();
}

function handleGaussianViewportMessage(event) {
  if (
    !gaussianViewportFrame ||
    event.source !== gaussianViewportFrame.contentWindow ||
    event.origin !== location.origin
  ) {
    return;
  }
  const message = event.data || {};
  if (
    message.source !== GAUSSIAN_BRIDGE_SOURCE ||
    message.sessionId !== gaussianViewportSessionId
  ) {
    return;
  }
  if (message.type === "bridge-starting") {
    state.gaussianViewportProgress = 1;
    syncReconstructionResultLists();
    renderIdPanel();
  } else if (message.type === "bridge-ready") {
    sendGaussianViewportCamera();
    sendAllGaussianPartVisibility();
    sendGaussianColorMode();
  } else if (message.type === "progress") {
    state.gaussianViewportProgress = Number(message.progress) || 0;
    syncReconstructionResultLists();
  } else if (message.type === "part-progress") {
    const partId = String(message.partId || "");
    if (partId) state.gaussianViewportPartProgress[partId] = Number(message.progress) || 0;
    state.gaussianViewportProgress = Number(message.sceneProgress) || 0;
    syncReconstructionResultLists();
    renderIdPanel();
  } else if (message.type === "part-ready") {
    const partId = String(message.partId || "");
    if (partId) state.gaussianViewportPartProgress[partId] = 100;
    renderIdPanel();
  } else if (message.type === "color-progress") {
    state.gaussianViewportColorSwitching = true;
    state.gaussianViewportColorProgress = Number(message.progress) || 0;
    renderIdPanel();
  } else if (message.type === "color-ready") {
    state.gaussianViewportColorSwitching = false;
    state.gaussianViewportColorProgress = 100;
    renderIdPanel();
  } else if (message.type === "ready") {
    state.gaussianViewportProgress = 100;
    state.gaussianViewportReady = true;
    state.gaussianViewportError = "";
    gaussianViewportFrame.setAttribute("aria-busy", "false");
    syncReconstructionResultLists();
    renderIdPanel();
    sendGaussianViewportCamera();
    sendAllGaussianPartVisibility();
    render();
    sendGaussianColorMode();
  } else if (message.type === "error") {
    state.gaussianViewportReady = false;
    state.gaussianViewportColorSwitching = false;
    state.gaussianViewportColorProgress = 100;
    state.gaussianViewportError = String(message.message || "Gaussian rendering failed.");
    state.gaussianViewportLoadedKey = "";
    gaussianViewportFrame.setAttribute("aria-busy", "false");
    syncReconstructionResultLists();
    renderIdPanel();
    setSelectionStatus(state.gaussianViewportError);
  }
}

window.addEventListener("message", handleGaussianViewportMessage);
