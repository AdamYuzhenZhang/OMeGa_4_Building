// Phase 3 geometry-aware selection: StableNormal flood fill and RGB-D cue graph selection.
const RGBD_CUE_PROMPT_CONFIG = {
  label: "RGB-D Cue",
  promptKey: "rgbdCuePrompts",
  timerKey: "rgbdCueRefreshTimer",
  serialKey: "rgbdCueRefreshSerial",
  delayMs: 260,
  refresh: refreshRgbdCueSelection,
};

function geometryGrowLabel(mode) {
  return mode === "rgbd-cue-select" ? "RGB-D Cue" : "Normal Grow";
}

function geometryGrowParameterLabel(mode) {
  if (mode === "rgbd-cue-select") return `lambda ${state.cueSmoothness.toFixed(2)}, ${state.cueSuperpixels} sp`;
  return `angle ${Math.round(state.normalGrowAngleDeg)} deg`;
}

function resetRgbdCuePrompts() {
  resetPromptStore(RGBD_CUE_PROMPT_CONFIG);
  resetRgbdCueDebug();
}

function activeFrameRgbdCuePrompts() {
  return activeFramePrompts(RGBD_CUE_PROMPT_CONFIG.promptKey);
}

function rgbdCueEvidenceReady() {
  return depthEvidenceCount() > 0 && normalEvidenceCount() > 0;
}

function rgbdCueMissingEvidenceMessage() {
  const missing = [];
  if (depthEvidenceCount() <= 0) missing.push("Depth Anything");
  if (normalEvidenceCount() <= 0) missing.push("StableNormal");
  return missing.length
    ? `Generate ${missing.join(" and ")} evidence before using RGB-D Cue`
    : "";
}

function resetRgbdCueDebug() {
  state.rgbdDebugPanels = [];
  state.rgbdDebugActiveId = "";
  state.rgbdDebugOverlayImage = null;
  state.rgbdDebugFrameId = null;
  state.rgbdDebugSummary = null;
  syncRgbdCueDebugControls();
  render();
}

function syncRgbdCueDebugControls() {
  const prompts = activeFrameRgbdCuePrompts();
  const canRun = Boolean(
    state.selectedFrame &&
    state.exactFrameView &&
    rgbdCueEvidenceReady() &&
    hasPositivePrompt(prompts)
  );
  if (runRgbdCueDebugButton) {
    runRgbdCueDebugButton.disabled = state.rgbdDebugBusy || !canRun;
    runRgbdCueDebugButton.title = canRun
      ? "Generate RGB-D cue debug overlays for the current frame and clicks"
      : "Add RGB-D foreground/background clicks on a frame with StableNormal and depth evidence first";
  }
  if (rgbdCueDebugSelect) {
    const active = state.rgbdDebugActiveId || "";
    rgbdCueDebugSelect.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "No debug overlay";
    rgbdCueDebugSelect.appendChild(empty);
    for (const panel of state.rgbdDebugPanels) {
      const option = document.createElement("option");
      option.value = panel.id;
      option.textContent = panel.label;
      rgbdCueDebugSelect.appendChild(option);
    }
    rgbdCueDebugSelect.value = active;
    rgbdCueDebugSelect.disabled = state.rgbdDebugBusy || state.rgbdDebugPanels.length === 0;
  }
}

async function setRgbdCueDebugOverlay(panelId) {
  state.rgbdDebugActiveId = panelId || "";
  if (!state.rgbdDebugActiveId) {
    state.rgbdDebugOverlayImage = null;
    syncRgbdCueDebugControls();
    render();
    return;
  }
  const panel = state.rgbdDebugPanels.find((item) => item.id === state.rgbdDebugActiveId);
  if (!panel || !panel.overlayPng) {
    state.rgbdDebugOverlayImage = null;
    syncRgbdCueDebugControls();
    render();
    return;
  }
  state.rgbdDebugOverlayImage = await loadImage(`data:image/png;base64,${panel.overlayPng}`);
  syncRgbdCueDebugControls();
  render();
}

async function runRgbdCueDebug() {
  if (state.rgbdDebugBusy || !state.selectedFrame || !state.exactFrameView) return;
  const prompts = activeFrameRgbdCuePrompts();
  const seed = latestPositivePrompt(prompts);
  if (!seed || !hasPositivePrompt(prompts)) {
    setSelectionStatus("Add RGB-D foreground cues before generating debug overlays");
    return;
  }
  if (!rgbdCueEvidenceReady()) {
    setSelectionStatus(rgbdCueMissingEvidenceMessage().replace("using RGB-D Cue", "RGB-D debug"));
    return;
  }
  const frameId = state.selectedFrame.id;
  state.rgbdDebugBusy = true;
  setSelectionBusy("RGB-D Debug", "Generating overlays", "rgbd-cue-select");
  syncRgbdCueDebugControls();
  setSelectionStatus(`RGB-D Cue Debug running (${geometryGrowParameterLabel("rgbd-cue-select")})`);
  try {
    const result = await postJson(`/api/view-evidence/frame/${frameId}/rgbd-cue-debug`, {
      mode: "rgbd-cue-select",
      strength: "normal",
      cueSmoothness: state.cueSmoothness,
      nSegments: state.cueSuperpixels,
      prompts,
      x: seed.sourceX,
      y: seed.sourceY,
    });
    if (!state.selectedFrame || Number(state.selectedFrame.id) !== Number(frameId)) return;
    state.rgbdDebugPanels = Array.isArray(result.panels) ? result.panels : [];
    state.rgbdDebugSummary = result.summary || null;
    state.rgbdDebugFrameId = frameId;
    const preferred = state.rgbdDebugPanels.some((panel) => panel.id === state.rgbdDebugActiveId)
      ? state.rgbdDebugActiveId
      : (state.rgbdDebugPanels[0] ? state.rgbdDebugPanels[0].id : "");
    await setRgbdCueDebugOverlay(preferred);
    setSelectionStatus(`RGB-D Cue Debug ready | ${state.rgbdDebugPanels.length} overlays`);
  } catch (error) {
    console.error(error);
    resetRgbdCueDebug();
    setSelectionStatus(error.message);
  } finally {
    state.rgbdDebugBusy = false;
    clearSelectionBusy("RGB-D Debug");
    syncRgbdCueDebugControls();
  }
}

function scheduleRgbdCueRefresh(reason = "parameters") {
  if (!rgbdCueEvidenceReady()) return;
  schedulePromptRefresh(RGBD_CUE_PROMPT_CONFIG, reason);
}

async function refreshRgbdCueSelection(reason = "parameters") {
  if (state.growBusy || !state.selectedFrame || !state.exactFrameView) {
    scheduleRgbdCueRefresh(reason);
    return;
  }
  const prompts = activeFrameRgbdCuePrompts();
  const seed = latestPositivePrompt(prompts);
  if (!seed || !hasPositivePrompt(prompts)) return;
  if (!rgbdCueEvidenceReady()) {
    setSelectionStatus(rgbdCueMissingEvidenceMessage());
    return;
  }

  const frameId = state.selectedFrame.id;
  const serial = state.rgbdCueRefreshSerial + 1;
  state.rgbdCueRefreshSerial = serial;
  state.growBusy = true;
  setSelectionBusy("RGB-D Cue", `Solving ${prompts.length} cue${prompts.length === 1 ? "" : "s"}`, "rgbd-cue-select");
  syncMaskEditControls();
  setSelectionStatus(`RGB-D Cue refreshing (${geometryGrowParameterLabel("rgbd-cue-select")})`);
  try {
    const result = await postJson(`/api/view-evidence/frame/${frameId}/grow-selection`, {
      mode: "rgbd-cue-select",
      strength: "normal",
      cueSmoothness: state.cueSmoothness,
      nSegments: state.cueSuperpixels,
      prompts,
      x: seed.sourceX,
      y: seed.sourceY,
    });
    if (serial !== state.rgbdCueRefreshSerial || !state.selectedFrame || Number(state.selectedFrame.id) !== Number(frameId)) return;
    if (!result.maskPng || Number(result.areaPixels) <= 0) {
      setSelectionStatus("RGB-D Cue refresh found no region");
      return;
    }
    await applyMaskPngSelection(result.maskPng, { replace: true });
    const area = Number(result.areaPixels) || 0;
    setSelectionStatus(`RGB-D Cue refreshed | ${area.toLocaleString()} px | ${geometryGrowParameterLabel("rgbd-cue-select")}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.growBusy = false;
    clearSelectionBusy("RGB-D Cue");
    syncMaskEditControls();
    render();
  }
}

async function growEvidenceSelectionAt(event, mode) {
  if (state.growBusy) return;
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before using geometry grow");
    return;
  }
  const isNormalGrow = mode === "normal-grow";
  const isRgbdCue = mode === "rgbd-cue-select";
  if (isNormalGrow && normalEvidenceCount() <= 0) {
    setSelectionStatus("Generate StableNormal evidence before using Normal Grow");
    return;
  }
  if (isRgbdCue && !rgbdCueEvidenceReady()) {
    setSelectionStatus(rgbdCueMissingEvidenceMessage());
    return;
  }
  const point = canvasPointFromEvent(event);
  const source = canvasPointToSource(point);
  if (!source) {
    setSelectionStatus(`${geometryGrowLabel(mode)} seed is outside the frame image`);
    return;
  }
  const operation = proposalSelectionOperationFromEvent(event);
  let rgbdPrompts = null;
  if (isRgbdCue) {
    rgbdPrompts = operation === "replace" ? [] : activeFrameRgbdCuePrompts();
    if (operation === "subtract" && !hasPositivePrompt(rgbdPrompts)) {
      setSelectionStatus("Add a positive RGB-D cue before adding background cues");
      return;
    }
    rgbdPrompts.push({
      frameId: state.selectedFrame.id,
      sourceX: source.x,
      sourceY: source.y,
      label: promptLabelFromOperation(operation),
    });
  }

  state.growBusy = true;
  if (isRgbdCue) {
    const count = rgbdPrompts ? rgbdPrompts.length : 1;
    setSelectionBusy("RGB-D Cue", `Solving ${count} cue${count === 1 ? "" : "s"}`, "rgbd-cue-select");
  }
  syncMaskEditControls();
  setSelectionStatus(`${geometryGrowLabel(mode)} running (${geometryGrowParameterLabel(mode)})`);
  try {
    const result = await postJson(`/api/view-evidence/frame/${state.selectedFrame.id}/grow-selection`, {
      mode,
      strength: "normal",
      normalAngleDeg: state.normalGrowAngleDeg,
      cueSmoothness: state.cueSmoothness,
      nSegments: state.cueSuperpixels,
      prompts: rgbdPrompts,
      x: source.x,
      y: source.y,
    });
    if (!result.maskPng || Number(result.areaPixels) <= 0) {
      setSelectionStatus(`${geometryGrowLabel(mode)} found no connected region`);
      return;
    }
    pushUndoSnapshot();
    if (isRgbdCue) {
      state.rgbdCuePrompts = cloneRgbdCuePrompts(rgbdPrompts);
      state.rgbdCueRefreshSerial += 1;
      resetRgbdCueDebug();
    }
    await applyMaskPngSelection(result.maskPng, {
      replace: operation === "replace" || isRgbdCue,
      operation: isRgbdCue ? "add" : operation,
    });
    const verb = isRgbdCue ? "updated" : operation === "subtract" ? "removed from" : operation === "add" ? "added to" : "selected";
    const area = Number(result.areaPixels) || 0;
    const promptText = isRgbdCue ? ` | ${rgbdPrompts.length} fg/bg click${rgbdPrompts.length === 1 ? "" : "s"}` : "";
    setSelectionStatus(`${geometryGrowLabel(mode)} ${verb} selection | ${area.toLocaleString()} px | ${geometryGrowParameterLabel(mode)}${promptText}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.growBusy = false;
    if (isRgbdCue) clearSelectionBusy("RGB-D Cue");
    syncMaskEditControls();
    render();
  }
}
