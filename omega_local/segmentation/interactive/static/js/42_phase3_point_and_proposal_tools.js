// Phase 3 direct selection tools: 3D point selection, proposal picking, and lasso pixels.
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

async function pickProposalAt(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before picking proposal masks");
    return;
  }
  if (!proposalLayerPriorityOrder().some((layer) => proposalLayerReady(layer))) {
    setSelectionStatus("Generate/load proposals or create propagated region proposals first");
    return;
  }
  const point = canvasPointFromEvent(event);
  const source = canvasPointToSource(point);
  if (!source) {
    setSelectionStatus("Proposal pick is outside the frame image");
    return;
  }
  try {
    const layers = pickProposalLayers();
    if (!layers.length) {
      setSelectionStatus("Turn on at least one ready proposal layer before picking");
      return;
    }
    const result = await postJson(`/api/proposals/sam2/frame/${state.selectedFrame.id}/pick`, {
      ...sourceToPixel(state.selectedFrame, source),
      layers,
    });
    if (!result.labelId) {
      setSelectionStatus("No proposal under cursor");
      return;
    }
    const layer = result.layer || "sam2";
    const details = result.details || { labelId: result.labelId, layer };
    applyProposalSelection([{ labelId: result.labelId, layer }], proposalSelectionOperationFromEvent(event));
    setSelectionStatus(`${proposalDisplayName({ ...details, labelId: result.labelId, layer })} selected | ${state.selectedProposalIds.size.toLocaleString()} total`);
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
  const polygon = canvasPolygonToSource(lasso.points, { clamp: true });
  if (polygon.length < 3) {
    setSelectionStatus("Lasso needs at least three points over the active frame");
    return;
  }
  const operation = lasso.operation || "replace";
  pushUndoSnapshot();
  if (operation === "replace") {
    resetFramePixelSelectionForReplace();
  }
  appendSelectionSource({ polygons: [{ operation: "add", polygon }] }, operation);
  state.proposalSelectionOverlayStamp += 1;
  syncMaskEditControls();
  const verb = operation === "subtract" ? "removed from" : operation === "add" ? "added to" : "selected";
  setSelectionStatus(`Lasso pixel region ${verb} selection`);
  refreshSelectionPreview();
}
