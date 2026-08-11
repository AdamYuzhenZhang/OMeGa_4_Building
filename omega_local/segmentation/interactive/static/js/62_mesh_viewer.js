// Camera-aligned PlayCanvas bridge for multipart object meshes.
const MESH_BRIDGE_SOURCE = "omega-mesh-embed";
const MESH_CAMERA_MESSAGE = "omega-mesh-camera";
const MESH_VISIBILITY_MESSAGE = "omega-mesh-visibility";
const MESH_COLOR_MODE_MESSAGE = "omega-mesh-color-mode";
let meshViewportCamera = null;
let meshViewportLoadSequence = 0;
let meshViewportSessionId = "";

function segmentation3dMeshArtifacts(run) {
  return run && Array.isArray(run.meshArtifacts) ? run.meshArtifacts : [];
}

function meshArtifact(runId, variantId) {
  const run = savedSegmentation3dRun(runId);
  const variant = segmentation3dMeshArtifacts(run).find(
    (row) => String(row.variantId) === String(variantId)
  );
  return run && variant ? { run, variant } : null;
}

function meshSceneManifestUrl(runId, variantId) {
  return datasetUrl(
    `/api/3d-segmentation/runs/${encodeURIComponent(runId)}` +
    `/mesh-scenes/${encodeURIComponent(variantId)}.json`
  );
}

function meshSceneKey(scene) {
  const versions = (scene.parts || [])
    .map((part) => `${part.partId}:${part.contentVersion || ""}`)
    .join("|");
  return `${scene.runId}:${scene.sceneId}:${versions}`;
}

function meshViewportVisible() {
  return Boolean(
    state.meshViewportVisible &&
    meshViewportFrame &&
    !meshViewportFrame.hidden
  );
}

function meshPartSceneActive() {
  return Boolean(
    meshViewportVisible() &&
    state.meshViewportScene &&
    Array.isArray(state.meshViewportScene.parts) &&
    state.meshViewportScene.parts.length
  );
}

function initializeMeshPartState(scene, preserve = false) {
  const previous = preserve ? state.meshViewportPartVisibility : {};
  state.meshViewportPartVisibility = {};
  state.meshViewportPartProgress = {};
  for (const part of scene.parts || []) {
    const partId = String(part.partId);
    state.meshViewportPartVisibility[partId] = Object.hasOwn(previous, partId)
      ? Boolean(previous[partId])
      : true;
    state.meshViewportPartProgress[partId] = 0;
  }
}

async function activateMeshViewport(runId, variantId) {
  if (!meshViewportFrame) throw new Error("The in-place mesh viewport is unavailable.");
  if (!meshArtifact(runId, variantId)) {
    throw new Error(`Mesh artifact ${runId}/${variantId} is not registered.`);
  }
  const scene = await loadJson(meshSceneManifestUrl(runId, variantId));
  if (!Array.isArray(scene.parts) || !scene.parts.length) {
    throw new Error(`Mesh scene ${runId}/${variantId} contains no renderable parts.`);
  }

  deactivateGaussianViewport({ redraw: false });
  hidePointRenderersForGaussian();
  const key = meshSceneKey(scene);
  const preserve = state.meshViewportLoadedKey === key;
  const modeIds = new Set(
    (scene.colorModes || []).map((row) => String(row.modeId || ""))
  );
  if (!preserve || !modeIds.has(state.meshViewportColorMode)) {
    state.meshViewportColorMode = String(scene.defaultColorMode || "rgb");
  }
  state.meshViewportRunId = String(runId);
  state.meshViewportVariantId = String(variantId);
  state.meshViewportScene = scene;
  initializeMeshPartState(scene, preserve);
  state.meshViewportVisible = true;
  state.meshViewportError = "";
  meshViewportFrame.hidden = false;
  meshViewportFrame.setAttribute("aria-hidden", "false");

  if (!preserve) {
    meshViewportLoadSequence += 1;
    meshViewportSessionId = `${runId}:${variantId}:${meshViewportLoadSequence}`;
    state.meshViewportLoadedKey = key;
    state.meshViewportReady = false;
    state.meshViewportProgress = 0;
    meshViewportFrame.setAttribute("aria-busy", "true");
    meshViewportFrame.src = "/static/mesh_embed.html" +
      `?scene=${encodeURIComponent(meshSceneManifestUrl(runId, variantId))}` +
      `&datasetId=${encodeURIComponent(activeDatasetId())}` +
      `&session=${encodeURIComponent(meshViewportSessionId)}`;
  } else {
    sendMeshViewportCamera();
    sendAllMeshPartVisibility();
    sendMeshColorMode();
  }
  syncReconstructionResultLists();
  renderIdPanel();
  render();
}

function deactivateMeshViewport({ redraw = true } = {}) {
  const wasVisible = meshViewportVisible();
  state.meshViewportVisible = false;
  state.meshViewportRunId = "";
  state.meshViewportVariantId = "";
  state.meshViewportScene = null;
  if (meshViewportFrame) {
    meshViewportFrame.hidden = true;
    meshViewportFrame.setAttribute("aria-hidden", "true");
  }
  syncReconstructionResultLists();
  renderIdPanel();
  if (redraw && wasVisible) render();
}

function sendMeshViewportCamera() {
  if (!meshViewportVisible() || !meshViewportCamera || !meshViewportSessionId) return;
  meshViewportFrame.contentWindow?.postMessage(
    {
      type: MESH_CAMERA_MESSAGE,
      sessionId: meshViewportSessionId,
      camera: meshViewportCamera,
    },
    location.origin,
  );
}

function sendMeshPartVisibility(partId, visible) {
  if (!meshViewportVisible() || !meshViewportSessionId) return;
  meshViewportFrame.contentWindow?.postMessage(
    {
      type: MESH_VISIBILITY_MESSAGE,
      sessionId: meshViewportSessionId,
      partId: String(partId),
      visible: Boolean(visible),
    },
    location.origin,
  );
}

function sendAllMeshPartVisibility() {
  for (const [partId, visible] of Object.entries(state.meshViewportPartVisibility)) {
    sendMeshPartVisibility(partId, visible);
  }
}

function sendMeshColorMode() {
  if (!meshViewportVisible() || !meshViewportSessionId) return;
  meshViewportFrame.contentWindow?.postMessage(
    {
      type: MESH_COLOR_MODE_MESSAGE,
      sessionId: meshViewportSessionId,
      mode: state.meshViewportColorMode,
    },
    location.origin,
  );
}

function setMeshColorMode(modeId) {
  const allowed = new Set(
    (state.meshViewportScene?.colorModes || []).map((row) => String(row.modeId || ""))
  );
  const mode = String(modeId || "");
  if (!allowed.has(mode) || state.meshViewportColorMode === mode) return;
  state.meshViewportColorMode = mode;
  sendMeshColorMode();
  renderIdPanel();
}

function setMeshPartVisibility(partId, visible) {
  const key = String(partId);
  if (!Object.hasOwn(state.meshViewportPartVisibility, key)) return;
  state.meshViewportPartVisibility[key] = Boolean(visible);
  sendMeshPartVisibility(key, visible);
  renderIdPanel();
}

function setAllMeshPartsVisible(visible) {
  for (const partId of Object.keys(state.meshViewportPartVisibility)) {
    state.meshViewportPartVisibility[partId] = Boolean(visible);
  }
  sendAllMeshPartVisibility();
  renderIdPanel();
}

function syncMeshViewportCamera(camera, layout) {
  meshViewportCamera = gaussianViewportCameraPayload(camera, layout);
  sendMeshViewportCamera();
}

function handleMeshViewportMessage(event) {
  if (
    !meshViewportFrame ||
    event.source !== meshViewportFrame.contentWindow ||
    event.origin !== location.origin
  ) return;
  const message = event.data || {};
  if (
    message.source !== MESH_BRIDGE_SOURCE ||
    message.sessionId !== meshViewportSessionId
  ) return;
  if (message.type === "bridge-starting") {
    state.meshViewportProgress = 1;
  } else if (message.type === "bridge-ready") {
    sendMeshViewportCamera();
    sendAllMeshPartVisibility();
    sendMeshColorMode();
  } else if (message.type === "part-progress") {
    const partId = String(message.partId || "");
    if (partId) state.meshViewportPartProgress[partId] = Number(message.progress) || 0;
    state.meshViewportProgress = Number(message.sceneProgress) || 0;
  } else if (message.type === "part-ready") {
    const partId = String(message.partId || "");
    if (partId) state.meshViewportPartProgress[partId] = 100;
  } else if (message.type === "ready") {
    state.meshViewportProgress = 100;
    state.meshViewportReady = true;
    state.meshViewportError = "";
    meshViewportFrame.setAttribute("aria-busy", "false");
    sendMeshViewportCamera();
    sendAllMeshPartVisibility();
    sendMeshColorMode();
    render();
  } else if (message.type === "error") {
    state.meshViewportReady = false;
    state.meshViewportError = String(message.message || "Mesh rendering failed.");
    state.meshViewportLoadedKey = "";
    meshViewportFrame.setAttribute("aria-busy", "false");
    setSelectionStatus(state.meshViewportError);
  }
  syncReconstructionResultLists();
  renderIdPanel();
}

window.addEventListener("message", handleMeshViewportMessage);
