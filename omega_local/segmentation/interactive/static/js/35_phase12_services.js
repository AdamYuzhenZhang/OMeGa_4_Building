// Phase 1/2 service calls: keyframe detection and SAM2 proposal generation/loading.
async function refreshProposalStatus() {
  try {
    const [proposalStatus, layerStatus] = await Promise.all([
      loadJson("/api/proposals/sam2/status"),
      loadJson("/api/proposals/layers/status"),
    ]);
    state.proposalStatus = proposalStatus;
    state.proposalLayerStatus = layerStatus;
  } catch (error) {
    console.error(error);
    state.proposalStatus = {
      ready: false,
      running: false,
      failed: true,
      message: error.message,
    };
    state.proposalLayerStatus = null;
  }
  initializeProposalLayerState();
  syncProposalControls();
  if (state.proposalStatus && state.proposalStatus.running) {
    startProposalPolling();
  }
  return state.proposalStatus;
}

async function refreshKeyframeStatus() {
  try {
    state.keyframeStatus = await loadJson("/api/keyframes/status");
  } catch (error) {
    console.error(error);
    state.keyframeStatus = {
      ready: false,
      running: false,
      failed: true,
      message: error.message,
    };
  }
  syncKeyframeControls();
  return state.keyframeStatus;
}

async function runKeyframeDetection() {
  if (state.keyframeBusy) return;
  state.keyframeBusy = true;
  syncKeyframeControls();
  setSelectionStatus("Detecting keyframes from proposals, boundaries, and poses");
  try {
    state.keyframeStatus = await postJson("/api/keyframes/detect", {});
    const count = Number(state.keyframeStatus.keyframeCount) || 0;
    setSelectionStatus(`Detected ${count.toLocaleString()} suggested keyframes`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
    state.keyframeStatus = {
      ready: false,
      failed: true,
      message: error.message,
    };
  } finally {
    state.keyframeBusy = false;
    syncKeyframeControls();
  }
}

function startProposalPolling() {
  if (state.proposalPollTimer) return;
  state.proposalPollTimer = window.setInterval(async () => {
    await refreshProposalStatus();
    if (!state.proposalStatus || !state.proposalStatus.running) {
      window.clearInterval(state.proposalPollTimer);
      state.proposalPollTimer = null;
      if (proposalReady() && state.selectedFrame) {
        loadActiveProposalOverlay(state.selectedFrame.id);
        loadProposalFrameInfo(state.selectedFrame.id);
      }
    }
  }, 1500);
}

async function startProposalRun(overwrite = false) {
  if (state.proposalStatus && state.proposalStatus.running) return;
  try {
    state.proposalStatus = await postJson("/api/proposals/sam2/run", { overwrite });
    if (overwrite) {
      state.proposalOverlayImage = null;
      state.proposalOverlayImages.clear();
      state.proposalLayerOverlayImages.clear();
    }
    syncProposalControls();
    startProposalPolling();
    setSelectionStatus(state.proposalStatus.message || (overwrite ? "Regenerating SAM2 proposals" : "Started SAM2 proposals"));
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
    await refreshProposalStatus();
  }
}

async function runAllProposals() {
  startProposalRun(false);
}

function loadExistingProposals() {
  if (!proposalReady()) return;
  syncProposalControls();
  if (state.selectedFrame) {
    loadFrameProposalLayers(state.selectedFrame.id).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  } else {
    render();
  }
  setSelectionStatus("Loaded existing SAM2 proposal overlays");
}

function regenerateProposals() {
  if (!proposalReady()) return;
  const ok = window.confirm("Regenerate SAM2 proposals and overwrite the current interactive proposal run?");
  if (!ok) return;
  startProposalRun(true);
}

async function loadActiveProposalOverlay(frameId) {
  const layers = visibleProposalLayers();
  if (!layers.length) {
    state.proposalOverlayImage = null;
    state.proposalLayerOverlayImage = new Map();
    render();
    return;
  }
  const images = new Map();
  for (const layer of layers) {
    const key = `${frameId}:${layer}:${proposalLayerStamp(layer)}`;
    let image = state.proposalLayerOverlayImages.get(key);
    if (!image) {
      try {
        image = await loadImage(proposalLayerOverlayUrl(frameId, layer));
        state.proposalLayerOverlayImages.set(key, image);
      } catch (error) {
        console.error(error);
        continue;
      }
    }
    images.set(layer, image);
  }
  if (state.selectedFrame && state.selectedFrame.id === frameId) {
    state.proposalLayerOverlayImage = images;
    state.proposalOverlayImage = images.get(layers[0]) || null;
    render();
  }
}

async function loadProposalFrameInfo(frameId) {
  if (!visibleProposalLayers().length) {
    state.proposalFrameInfo = null;
    return;
  }
  try {
    state.proposalFrameInfo = await loadJson(`/api/proposals/sam2/frame/${frameId}/summary`);
  } catch (error) {
    console.error(error);
    state.proposalFrameInfo = null;
  }
  syncMaskEditControls();
}
