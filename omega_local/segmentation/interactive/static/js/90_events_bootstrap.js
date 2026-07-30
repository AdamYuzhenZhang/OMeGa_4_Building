// Event wiring and initial project load.
function pointerMatchesActiveGesture(event) {
  return state.activePointerId === null || state.activePointerId === undefined || Number(event.pointerId) === Number(state.activePointerId);
}

function captureActivePointer(event) {
  state.activePointerId = event.pointerId;
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
}

function releaseActivePointer(event) {
  if (!pointerMatchesActiveGesture(event)) return;
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
  state.activePointerId = null;
}

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
  if (event.button === 0 && (state.tool === "normal-grow" || state.tool === "rgbd-cue-select")) {
    growEvidenceSelectionAt(event, state.tool);
    return;
  }
  if (event.button === 0 && state.tool === "mask-lasso") {
    beginMaskLasso(event);
    if (state.lasso) captureActivePointer(event);
    return;
  }
  if (event.button === 0 && state.tool === "lasso") {
    beginLasso(event);
    if (state.lasso) captureActivePointer(event);
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
  captureActivePointer(event);
}

function onPointerMove(event) {
  if ((state.lasso || state.dragging) && !pointerMatchesActiveGesture(event)) return;
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
  if (state.lasso || state.dragging) return;
  state.pointerInsideViewer = false;
  updateCursorBadge();
}

function hideCursorBadge() {
  state.pointerInsideViewer = false;
  updateCursorBadge();
}

function onPointerUp(event) {
  if (state.lasso) {
    if (!pointerMatchesActiveGesture(event)) return;
    event.preventDefault();
    finishLasso();
    releaseActivePointer(event);
    return;
  }
  if (!state.dragging) return;
  if (!pointerMatchesActiveGesture(event)) return;
  state.dragging = false;
  state.dragButton = null;
  canvas.classList.remove("dragging");
  updateCursorBadge();
  releaseActivePointer(event);
}

function onWindowPointerMove(event) {
  if (!state.lasso && !state.dragging) return;
  if (!pointerMatchesActiveGesture(event)) return;
  if (event.target === canvas) return;
  onPointerMove(event);
}

function onWindowPointerUp(event) {
  if (!state.lasso && !state.dragging) return;
  if (!pointerMatchesActiveGesture(event)) return;
  if (event.target === canvas) return;
  onPointerUp(event);
}

function onWheel(event) {
  event.preventDefault();
  cancelTransition();
  if (!zoomFrameView(event, event.deltaY)) {
    zoomCamera(event.deltaY);
  }
  render();
}

function frameImageAspect(frame) {
  const width = Number(frame && frame.width) || 0;
  const height = Number(frame && frame.height) || 0;
  if (width > 0 && height > 0) return width / height;
  return 1.5;
}

function syncFilmstripThumbnailLayout() {
  const tiles = [...filmstrip.querySelectorAll(".frame-tile")];
  if (tiles.length === 0) return;
  const firstTileHeight = tiles[0].getBoundingClientRect().height;
  const tileHeight = Math.max(40, firstTileHeight || 108);
  for (const tile of tiles) {
    const aspect = Math.max(0.1, Number(tile.dataset.frameAspect) || 1.5);
    const rotatedWidth = tileHeight;
    const rotatedHeight = tileHeight / aspect;
    tile.style.setProperty("--rotated-thumb-width", `${rotatedWidth.toFixed(2)}px`);
    tile.style.setProperty("--rotated-thumb-height", `${rotatedHeight.toFixed(2)}px`);
    tile.style.flexBasis = state.rotateFrames ? `${rotatedHeight.toFixed(2)}px` : "";
  }
}

function markActiveFrame() {
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    tile.classList.toggle("active", Number(tile.dataset.frameId) === (state.selectedFrame ? state.selectedFrame.id : -1));
    tile.classList.toggle("portrait", state.rotateFrames);
  }
  syncFilmstripThumbnailLayout();
}

function buildFilmstrip() {
  filmstrip.textContent = "";
  for (const frame of state.frames) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "frame-tile";
    tile.dataset.frameId = String(frame.id);
    tile.dataset.frameAspect = String(frameImageAspect(frame));
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
    label.textContent = String(frame.id);

    const keyframeMarker = document.createElement("span");
    keyframeMarker.className = "keyframe-marker";
    keyframeMarker.hidden = true;

    const completeMarker = document.createElement("span");
    completeMarker.className = "complete-marker";
    completeMarker.hidden = true;

    tile.append(image, proposalOverlay, keyframeMarker, completeMarker, label);
    tile.addEventListener("click", () => focusFrame(frame));
    filmstrip.appendChild(tile);
  }
  markActiveFrame();
  updateKeyframeThumbnails();
  updateProposalThumbnails();
  if (typeof updateRegionCompletionThumbnails === "function") updateRegionCompletionThumbnails();
}

async function initialize() {
  setStatus("Loading project");
  const [project, frames, points, pointCloudSourcesStatus, labelSummary, proposalStatus, proposalLayerStatus, keyframeStatus, viewEvidenceStatus, regionStatus, segmentation3dStatus] = await Promise.all([
    loadJson("/api/project"),
    loadJson("/api/frames"),
    loadJson("/api/points"),
    loadJson("/api/point-clouds/status"),
    loadJson("/api/labels/summary"),
    loadJson("/api/proposals/sam2/status"),
    loadJson("/api/proposals/layers/status"),
    loadJson("/api/keyframes/status"),
    loadJson("/api/view-evidence/status"),
    loadJson("/api/regions/status"),
    loadJson("/api/3d-segmentation/status"),
  ]);

  state.project = project;
  state.frames = frames;
  state.positions = Float32Array.from(points.positions);
  state.labels = Int32Array.from(points.labels);
  state.sourceIndices = Array.isArray(points.indices) ? points.indices : [];
  state.labelSummary = labelSummary;
  state.proposalStatus = proposalStatus;
  state.proposalLayerStatus = proposalLayerStatus;
  state.keyframeStatus = keyframeStatus;
  state.viewEvidenceStatus = viewEvidenceStatus;
  state.regionStatus = regionStatus;
  state.bounds = points.bounds;
  state.pointSize = Number(pointSizeInput.value);
  state.showPointCloud = showPointCloudInput ? showPointCloudInput.checked : true;
  initializePointCloudSources(pointCloudSourcesStatus);
  initializeSegmentation3dControls(segmentation3dStatus);
  state.showFrameImage = showFrameImageInput.checked;
  state.frameBackgroundMode = frameBackgroundModeInput ? frameBackgroundModeInput.value : "rgb";
  state.rotateFrames = rotateFramesInput.checked;
  initializeProposalLayerState();
  state.propagationMethodId = String(
    proposalLayerStatus.defaultPropagationMethodId ||
    (Array.isArray(proposalLayerStatus.methods) && proposalLayerStatus.methods[0]
      ? proposalLayerStatus.methods[0].methodId
      : "")
  );
  state.sourceRefinementMethodId = String(
    proposalLayerStatus.defaultSourceRefinementMethodId || ""
  );
  state.showRegions = showRegionsInput ? showRegionsInput.checked : true;
  state.showPromptDots = showPromptDotsInput ? showPromptDotsInput.checked : true;
  state.selectionProtectRegions = lockRegionSelectionInput ? lockRegionSelectionInput.checked : false;
  state.sam2MaskSize = sam2SizeInput ? Number(sam2SizeInput.value) || 0.0 : 0.0;
  state.normalGrowAngleDeg = normalGrowAngleInput ? Number(normalGrowAngleInput.value) || 14 : 14;
  state.cueSmoothness = depthGrowEdgeInput ? Number(depthGrowEdgeInput.value) || 0.10 : 0.10;
  state.cueSuperpixels = rgbdCueSuperpixelsInput ? Number(rgbdCueSuperpixelsInput.value) || 800 : 800;

  const labelText = project.labelCount > 0 ? `${project.labelCount.toLocaleString()} ids` : "raw unsegmented";
  setStatus(`${project.baselineName} | ${labelText} | ${project.servedPointCount.toLocaleString()} points | ${project.frameCount} frames`);
  if (pointCloudStatusEl) {
    pointCloudStatusEl.textContent = `${project.servedPointCount.toLocaleString()} / ${project.pointCount.toLocaleString()} pts`;
    pointCloudStatusEl.title = `${project.labelSource || "raw"} labels | ${project.pointsPath || ""}`;
  }
  syncSelectionControls();
  syncGrowControls();
  syncToolButtons();
  syncMaskEditControls();
  syncKeyframeControls();
  syncProposalControls();
  syncViewEvidenceControls();
  if (typeof syncRegionControls === "function") syncRegionControls();
  if (typeof syncPreviewExportControl === "function") syncPreviewExportControl();
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
resetButton.addEventListener("click", resetView);
if (clearSelectionButton) clearSelectionButton.addEventListener("click", clearSelection);
detectKeyframesButton.addEventListener("click", runKeyframeDetection);
reloadKeyframesButton.addEventListener("click", refreshKeyframeStatus);
loadProposalsButton.addEventListener("click", loadExistingProposals);
runProposalsButton.addEventListener("click", runAllProposals);
regenerateProposalsButton.addEventListener("click", regenerateProposals);
generateNormalEvidenceButton.addEventListener("click", () => runViewEvidenceTarget("normal"));
generateDepthEvidenceButton.addEventListener("click", () => runViewEvidenceTarget("depth"));
if (generateDinoEvidenceButton) {
  generateDinoEvidenceButton.addEventListener("click", () => runViewEvidenceTarget("dinov3"));
}
clearProposalSelectionButton.addEventListener("click", clearProposalSelection);
createRegionFromSelectionButton.addEventListener("click", createPersistentRegionFromSelection);
assignSelectionToRegionButton.addEventListener("click", assignSelectionToPersistentRegion);
addSelectionToRegionButton.addEventListener("click", addSelectionToPersistentRegion);
clearSelectionFromRegionsButton.addEventListener("click", clearSelectionFromPersistentRegions);
clearRegionFrameButton.addEventListener("click", clearActiveRegionFromFrame);
renameRegionButton.addEventListener("click", renameActiveRegion);
deleteRegionButton.addEventListener("click", deleteActiveRegion);
closeActiveRegionButton.addEventListener("click", closeActiveRegion);
if (useSuggestedRegionButton) {
  useSuggestedRegionButton.addEventListener("click", useSuggestedRegion);
}
if (runPropagationButton) {
  runPropagationButton.addEventListener("click", () => rebuildPropagatedRegionLayer(selectedPropagationMethodId()));
}
if (openPropagationPreviewButton) {
  openPropagationPreviewButton.addEventListener("click", () => openSavedPropagationPreview(selectedPropagationMethodId()));
}
if (runSourceRefinementButton) {
  runSourceRefinementButton.addEventListener("click", () => {
    rebuildPropagatedRegionLayer(selectedSourceRefinementMethodId());
  });
}
if (openSourceRefinementPreviewButton) {
  openSourceRefinementPreviewButton.addEventListener("click", () => {
    openSavedPropagationPreview(selectedSourceRefinementMethodId());
  });
}
if (testRegionPairButton) {
  testRegionPairButton.addEventListener("click", testSelectedRegionPair);
}
if (regionPairMethodInput) {
  regionPairMethodInput.addEventListener("change", () => {
    state.regionPairMethodId = String(regionPairMethodInput.value || "");
    syncPropagationControls();
  });
}
if (propagationMethodInput) {
  propagationMethodInput.addEventListener("change", () => {
    state.propagationMethodId = String(propagationMethodInput.value || "");
    syncPropagationControls();
    if (propagationModal && !propagationModal.hidden && !state.propagationBusy) {
      renderSavedPropagationPreview();
    }
  });
}
if (sourceRefinementMethodInput) {
  sourceRefinementMethodInput.addEventListener("change", () => {
    state.sourceRefinementMethodId = String(sourceRefinementMethodInput.value || "");
    syncPropagationControls();
    if (propagationModal && !propagationModal.hidden && !state.propagationBusy) {
      renderSavedPropagationPreview(selectedSourceRefinementLayer());
    }
  });
}
closePropagationModalButton.addEventListener("click", closePropagationModal);
if (runSegmentation3dButton) runSegmentation3dButton.addEventListener("click", runSegmentation3d);
if (segmentation3dMethodInput) {
  segmentation3dMethodInput.addEventListener("change", () => {
    refreshSegmentation3dInputOptions();
    refreshSegmentation3dInputSettings();
    syncSegmentation3dControls();
  });
}
if (segmentation3dInput) {
  segmentation3dInput.addEventListener("change", () => {
    refreshSegmentation3dInputSettings();
    syncSegmentation3dControls();
  });
}
if (segmentation3dSourceInput) {
  segmentation3dSourceInput.addEventListener("change", syncSegmentation3dControls);
}
if (segmentation3dManualWeightInput) {
  segmentation3dManualWeightInput.addEventListener("change", syncSegmentation3dControls);
}
if (exportPreviewButton) {
  exportPreviewButton.addEventListener("click", exportCurrentPreviewJpeg);
}
pointSizeInput.addEventListener("input", () => {
  state.pointSize = Number(pointSizeInput.value);
  render();
});
if (sam2SizeInput) {
  sam2SizeInput.addEventListener("input", () => {
    setSam2MaskSize(sam2SizeInput.value);
  });
}
if (normalGrowAngleInput) {
  normalGrowAngleInput.addEventListener("input", () => {
    setNormalGrowAngle(normalGrowAngleInput.value);
  });
}
if (depthGrowEdgeInput) {
  depthGrowEdgeInput.addEventListener("input", () => {
    setDepthGrowEdgeTolerance(depthGrowEdgeInput.value);
  });
}
if (rgbdCueSuperpixelsInput) {
  rgbdCueSuperpixelsInput.addEventListener("input", () => {
    setRgbdCueSuperpixels(rgbdCueSuperpixelsInput.value);
  });
}
if (runRgbdCueDebugButton) {
  runRgbdCueDebugButton.addEventListener("click", runRgbdCueDebug);
}
if (rgbdCueDebugSelect) {
  rgbdCueDebugSelect.addEventListener("change", () => {
    setRgbdCueDebugOverlay(rgbdCueDebugSelect.value);
  });
}
showPointCloudInput.addEventListener("change", () => {
  state.showPointCloud = showPointCloudInput.checked;
  if (state.showPointCloud) deactivateGaussianViewport({ redraw: false });
  render();
});
if (showColmapPointCloudInput) {
  showColmapPointCloudInput.addEventListener("change", () => {
    setColmapPointCloudVisible(showColmapPointCloudInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (useCleanedColmapInput) {
  useCleanedColmapInput.addEventListener("change", () => {
    setColmapPointCloudCleaned(useCleanedColmapInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (useSegmentedColmapInput) {
  useSegmentedColmapInput.addEventListener("change", () => {
    setColmapPointCloudSegmented(useSegmentedColmapInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (showFeedforwardPointCloudInput) {
  showFeedforwardPointCloudInput.addEventListener("change", () => {
    setFeedforwardPointCloudVisible(showFeedforwardPointCloudInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (useSegmentedFeedforwardInput) {
  useSegmentedFeedforwardInput.addEventListener("change", () => {
    setFeedforwardPointCloudSegmented(useSegmentedFeedforwardInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (showOmegaFinalPointCloudInput) {
  showOmegaFinalPointCloudInput.addEventListener("change", () => {
    setOmegaFinalPointCloudVisible(showOmegaFinalPointCloudInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (useHybridOmegaFinalInput) {
  useHybridOmegaFinalInput.addEventListener("change", () => {
    setOmegaFinalPointCloudHybrid(useHybridOmegaFinalInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
if (useSegmentedOmegaFinalInput) {
  useSegmentedOmegaFinalInput.addEventListener("change", () => {
    setOmegaFinalPointCloudSegmented(useSegmentedOmegaFinalInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
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
if (showRegionsInput) {
  showRegionsInput.addEventListener("change", () => {
    state.showRegions = showRegionsInput.checked;
    if (state.selectedFrame) {
      loadActiveRegionOverlay(state.selectedFrame.id).catch((error) => {
        console.error(error);
        setSelectionStatus(error.message);
      });
    } else {
      state.regionOverlayImage = null;
      render();
    }
  });
}
if (frameCompleteInput) {
  frameCompleteInput.addEventListener("change", () => {
    setCurrentFrameComplete(frameCompleteInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
      syncRegionControls();
    });
  });
}
if (regionNameInput) {
  regionNameInput.addEventListener("input", syncRegionControls);
}
if (showPromptDotsInput) {
  showPromptDotsInput.addEventListener("change", () => {
    state.showPromptDots = showPromptDotsInput.checked;
    render();
  });
}
if (lockRegionSelectionInput) {
  lockRegionSelectionInput.addEventListener("change", () => {
    setRegionSelectionLock(lockRegionSelectionInput.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
}
canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointerenter", onPointerEnter);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerleave", onPointerLeave);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", onWheel, { passive: false });
window.addEventListener("pointermove", onWindowPointerMove, true);
window.addEventListener("pointerup", onWindowPointerUp, true);
window.addEventListener("pointercancel", onWindowPointerUp, true);
document.addEventListener("mouseleave", hideCursorBadge);
window.addEventListener("mouseout", (event) => {
  if (!event.relatedTarget && !event.toElement) hideCursorBadge();
});
window.addEventListener("resize", () => {
  const resized = syncCanvasSize();
  syncFilmstripThumbnailLayout();
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
initialize().catch((error) => {
  console.error(error);
  setStatus("Failed to load");
  hud.textContent = error.message;
});
