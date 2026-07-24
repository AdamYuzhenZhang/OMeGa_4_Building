// Phase 3 SAM2 live selection: prompt clicks become one temporary selected-pixel region.
const SAM2_PROMPT_CONFIG = {
  label: "SAM2",
  promptKey: "samPrompts",
  timerKey: "samRefreshTimer",
  serialKey: "samRefreshSerial",
  delayMs: 180,
  refresh: refreshSam2Selection,
};

function resetSamState() {
  resetPromptStore(SAM2_PROMPT_CONFIG);
}

function activeFrameSamPrompts() {
  return activeFramePrompts(SAM2_PROMPT_CONFIG.promptKey);
}

function scheduleSam2Refresh(reason = "prompt") {
  schedulePromptRefresh(SAM2_PROMPT_CONFIG, reason);
}

function addSamPrompt(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before adding SAM2 prompts");
    return;
  }
  const point = canvasPointFromEvent(event);
  const source = canvasPointToSource(point);
  if (!source) {
    setSelectionStatus("SAM2 prompt is outside the frame image");
    return;
  }
  const label = promptLabelFromOperation(samPromptOperation(event));
  pushUndoSnapshot();
  appendPromptToStore(SAM2_PROMPT_CONFIG, source, label);
  syncSamControls();
  syncSelectionOperationControls();
  render();
  scheduleSam2Refresh(`${promptLabelText(label)} prompt`);
}

async function refreshSam2Selection(reason = "prompt", options = {}) {
  if (!state.selectedFrame || !state.exactFrameView) return;
  if (state.samBusy) {
    scheduleSam2Refresh(reason);
    return;
  }
  const frameId = state.selectedFrame.id;
  const prompts = activeFrameSamPrompts();
  if (!prompts.length) {
    setSelectionStatus("SAM2 prompts belong to another frame");
    return;
  }
  if (!hasPositivePrompt(prompts)) {
    setSelectionStatus("SAM2 needs at least one positive prompt");
    return;
  }
  const serial = state.samRefreshSerial + 1;
  state.samRefreshSerial = serial;
  state.samBusy = true;
  syncSamControls();
  setSelectionBusy("SAM2", `Running ${prompts.length} prompt${prompts.length === 1 ? "" : "s"}`, "sam2");
  setSelectionStatus(`SAM2 refreshing (${prompts.length} prompt${prompts.length === 1 ? "" : "s"})`);
  try {
    const result = await postJson("/api/sam2/predict", {
      frameId,
      prompts,
      multimask: true,
      maskThreshold: -state.sam2MaskSize,
    });
    if (serial !== state.samRefreshSerial || !state.selectedFrame || Number(state.selectedFrame.id) !== Number(frameId)) return;
    const counts = promptCounts(prompts);
    if (options.pushUndo) pushUndoSnapshot();
    await applyMaskPngSelection(result.maskPng, { replace: true });
    syncSamControls();
    syncSelectionOperationControls();
    const area = Number(result.area) || 0;
    const score = Number(result.score);
    const scoreText = Number.isFinite(score) ? ` | score ${score.toFixed(3)}` : "";
    setSelectionStatus(`SAM2 live selection | ${counts.positive} positive, ${counts.negative} negative | ${area.toLocaleString()} px | size ${state.sam2MaskSize.toFixed(2)}${scoreText}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.samBusy = false;
    clearSelectionBusy("SAM2");
    syncSamControls();
    syncSelectionOperationControls();
    render();
  }
}
