// Phase 3 tools: point click/lasso, proposal pick/lasso, SAM2 prompts, and proposal updates.
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

function nearestPointAt(canvasX, canvasY) {
  const camera = renderCamera();
  if (!camera) return null;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const threshold = 12 * (window.devicePixelRatio || 1);
  const threshold2 = threshold * threshold;
  let best = null;
  const positions = state.positions;
  for (let i = 0, j = 0; i < state.labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const x = layout.x0 + projected.u * layout.scale;
    const y = layout.y0 + projected.v * layout.scale;
    const dx = x - canvasX;
    const dy = y - canvasY;
    const dist2 = dx * dx + dy * dy;
    if (dist2 > threshold2) continue;
    if (!best || dist2 < best.dist2) {
      best = { index: i, dist2, projected };
    }
  }
  return best;
}

function selectByClickedId(event) {
  const point = canvasPointFromEvent(event);
  const nearest = nearestPointAt(point.x, point.y);
  if (!nearest) {
    setSelectionStatus("No point under cursor");
    return;
  }
  const label = state.labels[nearest.index];
  if (label <= 0) {
    pushUndoSnapshot();
    applySelection([nearest.index], selectionOperationFromEvent(event));
    setSelectionStatus(`Selected raw point | ${state.selectedPointIndices.size.toLocaleString()} selected`);
    render();
    return;
  }
  const indices = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    if (state.labels[i] === label) indices.push(i);
  }
  pushUndoSnapshot();
  applySelection(indices, selectionOperationFromEvent(event));
  setSelectionStatus(`Selected label ${label}: ${state.selectedPointIndices.size.toLocaleString()} points`);
  render();
}

function projectedIndicesInsidePolygon(polygon) {
  const camera = renderCamera();
  if (!camera) return [];
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const result = [];
  const positions = state.positions;
  for (let i = 0, j = 0; i < state.labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const point = {
      x: layout.x0 + projected.u * layout.scale,
      y: layout.y0 + projected.v * layout.scale,
    };
    if (pointInPolygon(point, polygon)) result.push(i);
  }
  return result;
}

function beginLasso(event) {
  const point = canvasPointFromEvent(event);
  state.lasso = {
    kind: "points",
    points: [point],
    operation: selectionOperationFromEvent(event),
  };
  canvas.classList.add("dragging");
}

function updateLasso(event) {
  if (!state.lasso) return;
  const point = canvasPointFromEvent(event);
  const points = state.lasso.points;
  const previous = points[points.length - 1];
  const minStep = 2 * (window.devicePixelRatio || 1);
  if (Math.hypot(point.x - previous.x, point.y - previous.y) >= minStep) {
    points.push(point);
    render();
  }
}

function finishLasso() {
  if (!state.lasso) return;
  const lasso = state.lasso;
  state.lasso = null;
  canvas.classList.remove("dragging");
  if (lasso.kind === "mask") {
    finishMaskLasso(lasso);
    return;
  }
  if (lasso.points.length >= 3) {
    const indices = projectedIndicesInsidePolygon(lasso.points);
    pushUndoSnapshot();
    applySelection(indices, lasso.operation);
    setSelectionStatus(`Lasso selected ${indices.length.toLocaleString()} visible points`);
  }
  render();
}

function clearSamState() {
  if (!state.samPrompts.length) return;
  pushUndoSnapshot();
  resetSamState();
  syncSamControls();
  syncSelectionOperationControls();
  render();
}

function resetSamState() {
  state.samPrompts = [];
}

function addSamPrompt(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before adding SAM2 prompts");
    return;
  }
  const point = canvasPointFromEvent(event);
  const camera = state.camera;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const view = currentBaseFrameView();
  const display = canvasToDisplayPoint(point.x, point.y, camera, layout, view);
  const source = displayToSource(state.selectedFrame, display.u, display.v);
  const inside =
    source.x >= 0 &&
    source.x < state.selectedFrame.width &&
    source.y >= 0 &&
    source.y < state.selectedFrame.height;
  if (!inside) {
    setSelectionStatus("SAM2 prompt is outside the frame image");
    return;
  }
  const operation = samPromptOperation(event);
  const label = operation === "subtract" ? 0 : 1;
  pushUndoSnapshot();
  state.samPrompts.push({
    frameId: state.selectedFrame.id,
    sourceX: source.x,
    sourceY: source.y,
    label,
  });
  syncSamControls();
  syncSelectionOperationControls();
  setSelectionStatus(`${state.samPrompts.length} SAM2 prompt${state.samPrompts.length === 1 ? "" : "s"} (${label ? "positive" : "negative"} added)`);
  render();
}

function canvasPointToSource(point) {
  if (!state.selectedFrame || !state.exactFrameView) return null;
  const camera = state.camera;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const view = currentBaseFrameView();
  const display = canvasToDisplayPoint(point.x, point.y, camera, layout, view);
  const source = displayToSource(state.selectedFrame, display.u, display.v);
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

function canvasPolygonToSource(points) {
  const polygon = [];
  for (const point of points) {
    const source = canvasPointToSource(point);
    if (source) polygon.push({ x: source.x, y: source.y });
  }
  return polygon;
}

async function pickProposalAt(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before picking proposal masks");
    return;
  }
  if (!proposalReady()) {
    setSelectionStatus("Generate or load SAM2 proposals first");
    return;
  }
  const point = canvasPointFromEvent(event);
  const source = canvasPointToSource(point);
  if (!source) {
    setSelectionStatus("Proposal pick is outside the frame image");
    return;
  }
  try {
    const result = await postJson(`/api/proposals/sam2/frame/${state.selectedFrame.id}/pick`, source);
    if (!result.labelId) {
      setSelectionStatus("No proposal under cursor");
      return;
    }
    applyProposalSelection([result.labelId], proposalSelectionOperationFromEvent(event));
    setSelectionStatus(`Proposal ${result.labelId} selected | ${state.selectedProposalIds.size.toLocaleString()} total`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  }
}

function beginMaskLasso(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before drawing a lasso region");
    return;
  }
  const point = canvasPointFromEvent(event);
  state.lasso = {
    kind: "mask",
    points: [point],
    operation: proposalSelectionOperationFromEvent(event),
  };
  canvas.classList.add("dragging");
}

function finishMaskLasso(lasso) {
  if (!state.selectedFrame || lasso.points.length < 3) return;
  const polygon = canvasPolygonToSource(lasso.points);
  if (polygon.length < 3) {
    setSelectionStatus("Lasso must stay inside the active frame");
    return;
  }
  const operation = lasso.operation || "replace";
  pushUndoSnapshot();
  if (operation === "replace") {
    state.selectedProposalIds.clear();
    state.activeProposalId = 0;
    state.selectionOps = [];
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
  }
  appendSelectionSource({ polygons: [{ operation: "add", polygon }] }, operation);
  state.proposalSelectionOverlayStamp += 1;
  syncMaskEditControls();
  const verb = operation === "subtract" ? "removed from" : operation === "add" ? "added to" : "selected";
  setSelectionStatus(`Lasso pixel region ${verb} selection`);
  refreshSelectionPreview();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });
}

async function runSam2() {
  if (state.samBusy || !state.selectedFrame || !state.samPrompts.length) return;
  const frameId = state.selectedFrame.id;
  const prompts = state.samPrompts.filter((prompt) => prompt.frameId === frameId);
  if (!prompts.length) {
    setSelectionStatus("SAM2 prompts belong to another frame");
    return;
  }
  state.samBusy = true;
  syncSamControls();
  setSelectionStatus("Running SAM2");
  try {
    const result = await postJson("/api/sam2/predict", {
      frameId,
      prompts,
      multimask: true,
    });
    const positiveCount = prompts.filter((prompt) => Number(prompt.label) > 0).length;
    const negativeCount = prompts.length - positiveCount;
    pushUndoSnapshot();
    state.selectedProposalIds.clear();
    state.activeProposalId = 0;
    state.selectionOps = [];
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
    appendSelectionSource({ maskPng: result.maskPng }, "add");
    state.proposalSelectionOverlayStamp += 1;
    resetSamState();
    refreshSelectionPreview();
    syncMaskEditControls();
    syncSamControls();
    syncSelectionOperationControls();
    setSelectionStatus(`SAM2 temporary selection | ${positiveCount} positive, ${negativeCount} negative | score ${result.score.toFixed(3)}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.samBusy = false;
    syncSamControls();
    syncSelectionOperationControls();
    render();
  }
}

async function queueProposalUpdate() {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before updating a proposal");
    return;
  }
  if (!hasPixelSelection()) {
    setSelectionStatus("Pick proposals, draw a lasso region, or add a mask first");
    return;
  }
  if (state.proposalUpdateBusy) return;

  const selectionOps = cloneSelectionOps(state.selectionOps);
  const preferredTargetLabelId = Number(state.activeProposalId) || Number([...state.selectedProposalIds][0]) || 0;
  const payload = {
    frameId: state.selectedFrame.id,
    preferredTargetLabelId,
    selectionOps,
  };
  state.proposalUpdateBusy = true;
  syncMaskEditControls();
  setSelectionStatus("Saving proposal update");
  try {
    const saveResult = await postJson("/api/proposals/sam2/save-edits", { edits: [payload] });
    const frameResult = Array.isArray(saveResult.frames)
      ? saveResult.frames.find((item) => Number(item.frameId) === Number(state.selectedFrame.id))
      : null;
    const editRecord = frameResult && Array.isArray(frameResult.editRecords) ? frameResult.editRecords[0] : null;
    const targetLabelId = Number(editRecord && editRecord.targetLabelId) || preferredTargetLabelId || 0;
    const createdNewTarget = Boolean(editRecord && editRecord.createdNewTarget);
    const areaPixels = Number(editRecord && editRecord.areaPixels) || 0;

    state.activeProposalId = targetLabelId;
    state.selectedProposalIds = targetLabelId > 0 ? new Set([targetLabelId]) : new Set();
    state.selectionOps = targetLabelId > 0 ? [{ operation: "add", proposalIds: [targetLabelId], polygons: [] }] : [];
    state.selectionPreviewArea = areaPixels;
    state.selectionPreviewCoverage = 0;
    state.proposalSelectionOverlayImage = null;
    state.proposalOverlayImages.clear();
    await refreshProposalStatus();
    if (state.selectedFrame) {
      await loadFrameProposalLayers(state.selectedFrame.id);
    }
    const idText = targetLabelId > 0 ? `proposal ${targetLabelId}` : "proposal";
    const modeText = createdNewTarget ? "created" : "updated";
    const areaText = areaPixels > 0 ? ` | ${areaPixels.toLocaleString()} px` : "";
    setSelectionStatus(`${idText} ${modeText} and saved${areaText}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.proposalUpdateBusy = false;
    syncMaskEditControls();
    syncSaveControls();
    render();
  }
}
