// Event wiring and initial project load.
function onPointerDown(event) {
  if (event.button !== 0 && event.button !== 2) return;
  updatePointerPosition(event);
  event.preventDefault();
  cancelTransition();
  if (event.button === 0 && state.tool === "click-id") {
    selectByClickedId(event);
    return;
  }
  if (event.button === 0 && state.tool === "sam2") {
    addSamPrompt(event);
    return;
  }
  if (event.button === 0 && state.tool === "mask-pick") {
    pickProposalAt(event);
    return;
  }
  if (event.button === 0 && state.tool === "mask-lasso") {
    beginMaskLasso(event);
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is best-effort across browsers.
    }
    return;
  }
  if (event.button === 0 && state.tool === "lasso") {
    beginLasso(event);
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is best-effort across browsers.
    }
    return;
  }
  if (event.button === 2) {
    exitExactFrameViewFor3D();
  }
  state.dragging = true;
  state.dragButton = event.button;
  state.lastPointer = [event.clientX, event.clientY];
  canvas.classList.add("dragging");
  updateCursorBadge();
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
}

function onPointerMove(event) {
  updatePointerPosition(event);
  if (state.lasso) {
    event.preventDefault();
    updateLasso(event);
    return;
  }
  if (!state.dragging) return;
  event.preventDefault();
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  if (state.dragButton === 2) {
    rotateCamera(dx, dy);
  } else if (!panFrameView(dx, dy)) {
    panCamera(dx, dy);
  }
  render();
}

function onPointerEnter(event) {
  updatePointerPosition(event);
}

function onPointerLeave() {
  state.pointerInsideViewer = false;
  updateCursorBadge();
}

function hideCursorBadge() {
  state.pointerInsideViewer = false;
  updateCursorBadge();
}

function onPointerUp(event) {
  if (state.lasso) {
    event.preventDefault();
    finishLasso();
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is best-effort across browsers.
    }
    return;
  }
  if (!state.dragging) return;
  state.dragging = false;
  state.dragButton = null;
  canvas.classList.remove("dragging");
  updateCursorBadge();
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
}

function onWheel(event) {
  event.preventDefault();
  cancelTransition();
  if (!zoomFrameView(event, event.deltaY)) {
    zoomCamera(event.deltaY);
  }
  render();
}

function markActiveFrame() {
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    tile.classList.toggle("active", Number(tile.dataset.frameId) === (state.selectedFrame ? state.selectedFrame.id : -1));
    tile.classList.toggle("portrait", state.rotateFrames);
  }
}

function buildFilmstrip() {
  filmstrip.textContent = "";
  for (const frame of state.frames) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "frame-tile";
    tile.dataset.frameId = String(frame.id);
    tile.dataset.baseTitle = `${frame.id}: ${frame.imageName}`;
    tile.title = `${frame.id}: ${frame.imageName}`;

    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = frame.imageUrl;
    image.alt = frame.imageName;

    const proposalOverlay = document.createElement("img");
    proposalOverlay.className = "proposal-thumb";
    proposalOverlay.loading = "lazy";
    proposalOverlay.alt = "";

    const label = document.createElement("div");
    label.className = "frame-label";
    label.textContent = `Frame ${frame.id}`;

    const keyframeMarker = document.createElement("span");
    keyframeMarker.className = "keyframe-marker";
    keyframeMarker.hidden = true;

    const keyframeScore = document.createElement("span");
    keyframeScore.className = "keyframe-score";

    tile.append(image, proposalOverlay, keyframeScore, keyframeMarker, label);
    tile.addEventListener("click", () => focusFrame(frame));
    filmstrip.appendChild(tile);
  }
  markActiveFrame();
  updateKeyframeThumbnails();
  updateProposalThumbnails();
}

async function initialize() {
  setStatus("Loading project");
  const [project, frames, points, labelSummary, proposalStatus, keyframeStatus, viewEvidenceStatus] = await Promise.all([
    loadJson("/api/project"),
    loadJson("/api/frames"),
    loadJson("/api/points"),
    loadJson("/api/labels/summary"),
    loadJson("/api/proposals/sam2/status"),
    loadJson("/api/keyframes/status"),
    loadJson("/api/view-evidence/status"),
  ]);

  state.project = project;
  state.frames = frames;
  state.positions = Float32Array.from(points.positions);
  state.labels = Int32Array.from(points.labels);
  state.sourceIndices = Array.isArray(points.indices) ? points.indices : [];
  state.labelSummary = labelSummary;
  state.proposalStatus = proposalStatus;
  state.keyframeStatus = keyframeStatus;
  state.viewEvidenceStatus = viewEvidenceStatus;
  state.bounds = points.bounds;
  state.pointSize = Number(pointSizeInput.value);
  state.showPointCloud = showPointCloudInput ? showPointCloudInput.checked : true;
  state.showFrameImage = showFrameImageInput.checked;
  state.frameBackgroundMode = frameBackgroundModeInput ? frameBackgroundModeInput.value : "rgb";
  state.rotateFrames = rotateFramesInput.checked;
  state.showProposals = showProposalsInput ? showProposalsInput.checked && proposalReady() : false;

  const labelText = project.labelCount > 0 ? `${project.labelCount.toLocaleString()} ids` : "raw unsegmented";
  setStatus(`${project.baselineName} | ${labelText} | ${project.servedPointCount.toLocaleString()} points | ${project.frameCount} frames`);
  if (pointCloudStatusEl) {
    const source = project.labelSource ? `${project.labelSource}` : "raw";
    pointCloudStatusEl.textContent = `${source} | ${project.servedPointCount.toLocaleString()} / ${project.pointCount.toLocaleString()} pts`;
    pointCloudStatusEl.title = project.pointsPath || "";
  }
  syncSelectionControls();
  syncToolButtons();
  syncMaskEditControls();
  syncKeyframeControls();
  syncProposalControls();
  syncViewEvidenceControls();
  if (state.proposalStatus && state.proposalStatus.running) startProposalPolling();
  if (state.viewEvidenceStatus && state.viewEvidenceStatus.running) startViewEvidencePolling();
  buildFilmstrip();
  resetView();
}

for (const button of toolButtons) {
  button.addEventListener("click", () => {
    setTool(button.dataset.tool || "navigate");
  });
}
for (const button of selectionOperationButtons) {
  button.addEventListener("click", () => {
    const operation = button.dataset.selectionOperation || "replace";
    if (operation === "add" || operation === "subtract") {
      setSelectionOperationMode(operation);
    }
  });
}
for (const button of proposalSortButtons) {
  button.addEventListener("click", () => {
    setProposalSortMode(button.dataset.proposalSort || "index");
  });
}
resetButton.addEventListener("click", resetView);
clearSelectionButton.addEventListener("click", clearSelection);
runSam2Button.addEventListener("click", (event) => runSam2(event));
clearSam2Button.addEventListener("click", clearSamState);
detectKeyframesButton.addEventListener("click", runKeyframeDetection);
reloadKeyframesButton.addEventListener("click", refreshKeyframeStatus);
loadProposalsButton.addEventListener("click", loadExistingProposals);
runProposalsButton.addEventListener("click", runAllProposals);
regenerateProposalsButton.addEventListener("click", regenerateProposals);
generateNormalEvidenceButton.addEventListener("click", () => runViewEvidenceTarget("normal"));
generateDepthEvidenceButton.addEventListener("click", () => runViewEvidenceTarget("depth"));
clearProposalSelectionButton.addEventListener("click", clearProposalSelection);
updateProposalButton.addEventListener("click", queueProposalUpdate);
runPropagationButton.addEventListener("click", runPropagationTest);
closePropagationModalButton.addEventListener("click", closePropagationModal);
assignSelectionButton.addEventListener("click", assignSelectionToTarget);
extractSelectionButton.addEventListener("click", extractSelectionToNewLabel);
mergeSelectedIdsButton.addEventListener("click", mergeSelectedIdsToTarget);
saveEditsButton.addEventListener("click", savePendingEdits);
targetLabelInput.addEventListener("input", syncSelectionControls);
pointSizeInput.addEventListener("input", () => {
  state.pointSize = Number(pointSizeInput.value);
  render();
});
showPointCloudInput.addEventListener("change", () => {
  state.showPointCloud = showPointCloudInput.checked;
  render();
});
propagationNeighborsInput.addEventListener("input", syncPropagationControls);
showFrameImageInput.addEventListener("change", () => {
  state.showFrameImage = showFrameImageInput.checked;
  render();
});
frameBackgroundModeInput.addEventListener("change", () => {
  state.frameBackgroundMode = frameBackgroundModeInput.value || "rgb";
  if (state.selectedFrame) {
    loadSelectedFrameImage(state.selectedFrame);
  }
  render();
});
rotateFramesInput.addEventListener("change", () => {
  state.rotateFrames = rotateFramesInput.checked;
  if (state.selectedFrame) {
    focusFrame(state.selectedFrame);
  } else {
    render();
  }
  markActiveFrame();
});
showProposalsInput.addEventListener("change", () => {
  state.showProposals = showProposalsInput.checked && proposalReady();
  updateProposalThumbnails();
  if (state.selectedFrame) {
    loadFrameProposalLayers(state.selectedFrame.id).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  } else {
    state.proposalOverlayImage = null;
    render();
  }
});
canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointerenter", onPointerEnter);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerleave", onPointerLeave);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", onWheel, { passive: false });
document.addEventListener("mouseleave", hideCursorBadge);
window.addEventListener("mouseout", (event) => {
  if (!event.relatedTarget && !event.toElement) hideCursorBadge();
});
window.addEventListener("resize", () => {
  const resized = syncCanvasSize();
  if (resized && !state.selectedFrame) resetView();
  render();
});
window.addEventListener("keydown", (event) => {
  syncModifierSelectionOperation(event);
  const isUndo = (event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "z";
  if (!isUndo) return;
  event.preventDefault();
  undoLastEdit();
});
window.addEventListener("keyup", (event) => {
  syncModifierSelectionOperation(event);
});
window.addEventListener("blur", () => {
  clearModifierSelectionOperation();
  hideCursorBadge();
});
window.addEventListener("beforeunload", (event) => {
  if (!hasUnsavedEdits()) return;
  event.preventDefault();
  event.returnValue = "";
});

initialize().catch((error) => {
  console.error(error);
  setStatus("Failed to load");
  hud.textContent = error.message;
});
