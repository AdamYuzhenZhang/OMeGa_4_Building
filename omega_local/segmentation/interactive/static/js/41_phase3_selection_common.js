// Phase 3 shared helpers: frame-coordinate mapping, selected-pixel masks, and click prompts.
function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const pi = polygon[i];
    const pj = polygon[j];
    const denom = Math.abs(pj.y - pi.y) < 1e-9 ? 1e-9 : pj.y - pi.y;
    const intersects =
      (pi.y > point.y) !== (pj.y > point.y) &&
      point.x < ((pj.x - pi.x) * (point.y - pi.y)) / denom + pi.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });
}

function canvasPointToSource(point, options = {}) {
  if (!state.selectedFrame || !state.exactFrameView) return null;
  const camera = state.camera;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const view = currentBaseFrameView();
  const display = canvasToDisplayPoint(point.x, point.y, camera, layout, view);
  const source = displayToSource(state.selectedFrame, display.u, display.v);
  if (options.clamp) {
    return {
      x: clamp(source.x, 0, Math.max(0, state.selectedFrame.width - 1)),
      y: clamp(source.y, 0, Math.max(0, state.selectedFrame.height - 1)),
    };
  }
  if (
    source.x < 0 ||
    source.x >= state.selectedFrame.width ||
    source.y < 0 ||
    source.y >= state.selectedFrame.height
  ) {
    return null;
  }
  return source;
}

function sourceToPixel(frame, source) {
  if (!frame || !source) return null;
  return {
    x: clamp(Math.floor(source.x), 0, Math.max(0, frame.width - 1)),
    y: clamp(Math.floor(source.y), 0, Math.max(0, frame.height - 1)),
  };
}

function canvasPolygonToSource(points, options = {}) {
  const polygon = [];
  for (const point of points) {
    const source = canvasPointToSource(point, options);
    if (!source) continue;
    const previous = polygon[polygon.length - 1];
    if (previous && Math.hypot(source.x - previous.x, source.y - previous.y) < 0.5) continue;
    polygon.push({ x: source.x, y: source.y });
  }
  if (polygon.length > 1) {
    const first = polygon[0];
    const last = polygon[polygon.length - 1];
    if (Math.hypot(first.x - last.x, first.y - last.y) < 0.5) polygon.pop();
  }
  return polygon;
}

function resetFramePixelSelectionForReplace() {
  state.selectedProposalIds.clear();
  state.activeProposalId = 0;
  state.selectionOps = [];
  state.proposalSelectionOverlayImage = null;
  state.selectionPreviewArea = 0;
  state.selectionPreviewCoverage = 0;
  state.selectionPreviewProtectedArea = 0;
}

async function applyMaskPngSelection(maskPng, options = {}) {
  if (!maskPng) return false;
  const operation = options.operation === "subtract" ? "subtract" : "add";
  if (options.replace) {
    resetFramePixelSelectionForReplace();
  }
  appendSelectionSource({ maskPng }, operation);
  state.proposalSelectionOverlayStamp += 1;
  await refreshSelectionPreview();
  syncMaskEditControls();
  return true;
}

function clearPromptRefresh(timerKey, serialKey) {
  if (state[timerKey]) {
    clearTimeout(state[timerKey]);
    state[timerKey] = null;
  }
  state[serialKey] += 1;
}

function resetPromptStore(config) {
  clearPromptRefresh(config.timerKey, config.serialKey);
  state[config.promptKey] = [];
}

function activeFramePrompts(promptKey) {
  if (!state.selectedFrame) return [];
  return cloneSamPrompts(state[promptKey]).filter((prompt) => (
    Number(prompt.frameId) === Number(state.selectedFrame.id)
  ));
}

function hasPositivePrompt(prompts) {
  return prompts.some((prompt) => Number(prompt.label) > 0);
}

function promptCounts(prompts) {
  const positive = prompts.filter((prompt) => Number(prompt.label) > 0).length;
  return {
    positive,
    negative: prompts.length - positive,
    total: prompts.length,
  };
}

function latestPositivePrompt(prompts) {
  for (let index = prompts.length - 1; index >= 0; index -= 1) {
    if (Number(prompts[index].label) > 0) return prompts[index];
  }
  return prompts.length ? prompts[prompts.length - 1] : null;
}

function promptLabelFromOperation(operation) {
  return operation === "subtract" ? 0 : 1;
}

function promptLabelText(label) {
  return Number(label) > 0 ? "positive" : "negative";
}

function appendPromptToStore(config, source, label, options = {}) {
  if (options.replace) {
    state[config.promptKey] = [];
  }
  state[config.promptKey].push({
    frameId: state.selectedFrame.id,
    sourceX: source.x,
    sourceY: source.y,
    label: Number(label) > 0 ? 1 : 0,
  });
  state[config.serialKey] += 1;
  return activeFramePrompts(config.promptKey);
}

function schedulePromptRefresh(config, reason = "prompt") {
  if (state[config.timerKey]) {
    clearTimeout(state[config.timerKey]);
    state[config.timerKey] = null;
  }
  if (!state.selectedFrame || !state.exactFrameView) return;
  const prompts = activeFramePrompts(config.promptKey);
  if (!hasPositivePrompt(prompts)) {
    setSelectionStatus(`Add at least one positive ${config.label} prompt`);
    return;
  }
  setSelectionStatus(`${config.label} waiting for ${reason} update...`);
  state[config.timerKey] = window.setTimeout(() => {
    state[config.timerKey] = null;
    config.refresh(reason);
  }, config.delayMs);
}
