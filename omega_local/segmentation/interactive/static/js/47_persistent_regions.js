// Phase 4 persistent regions: dataset-level segmentation intent above local proposals.
function regionRows() {
  return state.regionStatus && Array.isArray(state.regionStatus.regions)
    ? state.regionStatus.regions
    : [];
}

function regionSortName(region) {
  const id = Number(region && region.id) || 0;
  return String(region && region.name ? region.name : `region_${id}`).trim().toLocaleLowerCase();
}

function sortedRegionRows() {
  const rows = regionRows().map((region) => ({ ...region }));
  if (state.regionSortMode === "name") {
    rows.sort((a, b) => {
      const nameDelta = regionSortName(a).localeCompare(regionSortName(b), undefined, { numeric: true, sensitivity: "base" });
      if (nameDelta !== 0) return nameDelta;
      return (Number(a.id) || 0) - (Number(b.id) || 0);
    });
    return rows;
  }
  rows.sort((a, b) => (Number(a.id) || 0) - (Number(b.id) || 0));
  return rows;
}

function regionSortControl() {
  return sortButtonGroup("Persistent region sort mode", [
    {
      label: "ID",
      active: state.regionSortMode !== "name",
      dataset: { regionSort: "id" },
      onClick: () => setRegionSortMode("id"),
    },
    {
      label: "Name",
      active: state.regionSortMode === "name",
      dataset: { regionSort: "name" },
      onClick: () => setRegionSortMode("name"),
    },
  ]);
}

function syncRegionSortControls() {
  for (const button of document.querySelectorAll("[data-region-sort]")) {
    const mode = button.dataset.regionSort || "id";
    const active = (mode === "name") === (state.regionSortMode === "name");
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

function setRegionSortMode(mode) {
  state.regionSortMode = mode === "name" ? "name" : "id";
  syncRegionSortControls();
  renderIdPanel();
}

function completeFrameSet() {
  const rows = state.regionStatus && Array.isArray(state.regionStatus.completeFrames)
    ? state.regionStatus.completeFrames
    : [];
  return new Set(rows.map(Number).filter((value) => Number.isInteger(value) && value >= 0));
}

function currentFrameIsComplete() {
  const frameId = state.selectedFrame ? Number(state.selectedFrame.id) : -1;
  if (!Number.isInteger(frameId) || frameId < 0) return false;
  if (state.regionFrameInfo && Number(state.regionFrameInfo.frameId) === frameId) {
    return Boolean(state.regionFrameInfo.complete);
  }
  return completeFrameSet().has(frameId);
}

function currentFrameHasRegionPixels() {
  return Boolean(
    state.regionFrameInfo &&
    state.regionFrameInfo.ready &&
    Array.isArray(state.regionFrameInfo.labels) &&
    state.regionFrameInfo.labels.length > 0
  );
}

function updateRegionCompletionThumbnails() {
  if (!filmstrip) return;
  const completeFrames = completeFrameSet();
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    const frameId = Number(tile.dataset.frameId);
    const isComplete = completeFrames.has(frameId);
    tile.classList.toggle("complete", isComplete);
    const marker = tile.querySelector(".complete-marker");
    if (marker) {
      marker.hidden = !isComplete;
      marker.title = isComplete ? "Complete propagation anchor" : "";
    }
    const title = tile.title || tile.dataset.baseTitle || "";
    const cleanTitle = title.replace(/\ncomplete propagation anchor/g, "");
    tile.title = isComplete ? `${cleanTitle}\ncomplete propagation anchor` : cleanTitle;
  }
}

function activeRegion() {
  const activeId = Number(state.activeRegionId) || 0;
  return regionRows().find((region) => Number(region.id) === activeId) || null;
}

function regionColor(region) {
  const color = region && typeof region.color === "string" && region.color.length
    ? region.color
    : palette(Number(region && region.id) || 0);
  return color;
}

function regionOverlayStamp() {
  return state.regionStatus && state.regionStatus.updatedUtc
    ? String(state.regionStatus.updatedUtc)
    : "none";
}

function regionOverlayUrl(frameId) {
  return datasetUrl(`/api/regions/frame/${frameId}/overlay?v=${encodeURIComponent(regionOverlayStamp())}`);
}

function singleRegionOverlayUrl(frameId, regionId) {
  return datasetUrl(`/api/regions/frame/${frameId}/overlay/${regionId}?v=${encodeURIComponent(regionOverlayStamp())}`);
}

function manualRegionReferenceRows(region) {
  if (!region) return [];
  const byFrame = new Map();
  const frameAreas = new Map();
  if (Array.isArray(region.frameAreas)) {
    for (const row of region.frameAreas) {
      const frameId = Number(row && row.frameId);
      const areaPixels = Number(row && row.areaPixels) || 0;
      if (Number.isInteger(frameId) && frameId >= 0 && areaPixels > 0) {
        frameAreas.set(frameId, areaPixels);
      }
    }
  }
  const currentFrameIds = Array.isArray(region.frameIds)
    ? region.frameIds.map(Number).filter((value) => Number.isInteger(value) && value >= 0)
    : [];
  for (const frameId of currentFrameIds) {
    byFrame.set(frameId, {
      frameId,
      proposalIds: new Set(),
      areaPixels: Number(frameAreas.get(frameId)) || 0,
      operation: "current_region",
      count: 0,
    });
  }
  if (!byFrame.size) return [];
  const validFrames = new Set(byFrame.keys());
  const assignments = Array.isArray(region.assignments) ? region.assignments : [];
  for (const assignment of assignments) {
    const operation = String(assignment.operation || "");
    if (!["create_region", "assign_region", "new_region", "replace_region", "add_region"].includes(operation)) continue;
    const frameId = Number(assignment.frameId);
    if (!Number.isInteger(frameId) || frameId < 0) continue;
    if (!validFrames.has(frameId)) continue;
    const entry = byFrame.get(frameId);
    if (!entry) continue;
    entry.count += 1;
    if (operation === "create_region" || operation === "new_region") entry.operation = operation;
    const sourceIds = Array.isArray(assignment.sourceIds) ? assignment.sourceIds : assignment.proposalIds;
    if (Array.isArray(sourceIds)) {
      for (const proposalId of sourceIds) {
        const id = Number(proposalId);
        if (id > 0) entry.proposalIds.add(id);
      }
    }
    byFrame.set(frameId, entry);
  }
  return [...byFrame.values()]
    .map((entry) => ({
      ...entry,
      proposalIds: [...entry.proposalIds].sort((a, b) => a - b),
    }))
    .sort((a, b) => a.frameId - b.frameId);
}

function frameById(frameId) {
  return state.frames.find((frame) => Number(frame.id) === Number(frameId)) || null;
}

function activeRegionFrameLabel() {
  const active = activeRegion();
  if (!active || !state.regionFrameInfo || !Array.isArray(state.regionFrameInfo.labels)) return null;
  const rid = Number(active.id) || 0;
  return state.regionFrameInfo.labels.find((row) => Number(row.labelId || row.regionId) === rid) || null;
}

function activePropagatedProposalRow() {
  const activeLayer = normalizeProposalLayer(state.activeProposalLayer || "sam2");
  if (!isPropagatedProposalLayer(activeLayer)) return null;
  const labelId = Number(state.activeProposalId) || 0;
  if (labelId <= 0 || !state.proposalFrameInfo || !Array.isArray(state.proposalFrameInfo.labels)) return null;
  return state.proposalFrameInfo.labels.find((row) => (
    normalizeProposalLayer(row.layer || "sam2") === activeLayer &&
    Number(row.labelId) === labelId
  )) || null;
}

function activePropagatedRegionSuggestion() {
  const proposal = activePropagatedProposalRow();
  if (!proposal) return null;
  const regionId = proposalRegionId(proposal);
  if (regionId <= 0) return null;
  const region = regionRows().find((row) => Number(row.id) === regionId) || null;
  const name = proposalRegionName(proposal) || (region ? region.name : `region_${regionId}`);
  return {
    proposal,
    region,
    regionId,
    name,
    color: (region && regionColor(region)) || palette(regionId + 900),
    sourceFrameId: Number(proposal.sourceFrameId),
    sourceFrameIds: Array.isArray(proposal.sourceFrameIds) ? proposal.sourceFrameIds.map(Number).filter(Number.isFinite) : [],
    pixels: Number(proposal.assignedPixels || proposal.rawAreaPixels || 0),
  };
}

function renderRegionSuggestionCard() {
  if (!regionSuggestionCard) return;
  const suggestion = activePropagatedRegionSuggestion();
  const alreadyActive = suggestion && Number(state.activeRegionId) === Number(suggestion.regionId);
  regionSuggestionCard.hidden = !suggestion || alreadyActive;
  if (!suggestion || alreadyActive) {
    if (regionSuggestionReferenceStrip) {
      regionSuggestionReferenceStrip.textContent = "";
      regionSuggestionReferenceStrip.hidden = true;
    }
    return;
  }

  if (regionSuggestionSwatch) regionSuggestionSwatch.style.background = suggestion.color;
  if (regionSuggestionTitle) regionSuggestionTitle.textContent = suggestion.name;
  if (regionSuggestionMeta) {
    const sourceText = suggestion.sourceFrameIds.length > 1
      ? `sources ${suggestion.sourceFrameIds.join(", ")}`
      : suggestion.sourceFrameIds.length === 1
        ? `source frame ${suggestion.sourceFrameIds[0]}`
        : Number.isFinite(suggestion.sourceFrameId) && suggestion.sourceFrameId >= 0
          ? `source frame ${suggestion.sourceFrameId}`
          : "source frame unknown";
    const pixelText = suggestion.pixels > 0 ? `${formatCount(suggestion.pixels)} px` : "proposal pixels";
    const missingText = suggestion.region ? "" : " | source region was deleted";
    regionSuggestionMeta.textContent = `R${suggestion.regionId} | ${sourceText} | ${pixelText}${missingText}`;
  }
  renderRegionReferenceStripContent(regionSuggestionReferenceStrip, suggestion.region, {
    compact: true,
    emptyText: suggestion.region ? "No manual keyframe assignments yet." : "The source region is not available.",
  });
  if (useSuggestedRegionButton) {
    const hasPixels = hasPixelSelection();
    useSuggestedRegionButton.disabled = state.regionBusy || !suggestion.region || !hasPixels;
    useSuggestedRegionButton.textContent = "Accept Source";
    useSuggestedRegionButton.title = suggestion.region
      ? hasPixels
        ? "Accept the selected propagated pixels as this persistent region on the current frame."
        : "Select propagated pixels before accepting the source region."
      : "The persistent region referenced by this propagated proposal no longer exists.";
  }
}

async function useSuggestedRegion() {
  const suggestion = activePropagatedRegionSuggestion();
  if (!suggestion || !suggestion.region) return;
  setActiveRegion(suggestion.regionId);
  if (!hasPixelSelection()) {
    setSelectionStatus(`Selected source R${suggestion.regionId} ${suggestion.name}`);
    return;
  }
  await assignSelectionToPersistentRegion();
}

function renderSelectedRegionCard() {
  if (!selectedRegionCard) return;
  const region = activeRegion();
  selectedRegionCard.hidden = !region;
  if (!region) {
    if (selectedRegionTitle) selectedRegionTitle.textContent = "No region selected";
    if (selectedRegionMeta) selectedRegionMeta.textContent = "Select a region from the left panel.";
    if (selectedRegionSwatch) selectedRegionSwatch.style.background = "transparent";
    return;
  }

  const id = Number(region.id) || 0;
  const name = region.name || `region_${id}`;
  const frameCount = Number(region.frameCount) || 0;
  const pixelCount = Number(region.pixelCount) || 0;
  const frameLabel = activeRegionFrameLabel();
  const frameArea = frameLabel ? Number(frameLabel.assignedPixels || frameLabel.rawAreaPixels || 0) : 0;
  if (selectedRegionSwatch) selectedRegionSwatch.style.background = regionColor(region);
  if (selectedRegionTitle) selectedRegionTitle.textContent = name;
  if (selectedRegionMeta) {
    const currentText = state.selectedFrame
      ? frameLabel
        ? `current frame ${formatCount(frameArea)} px`
        : "not on current frame"
      : "no active frame";
    selectedRegionMeta.textContent = `R${id} | ${formatCount(pixelCount)} px | ${formatCount(frameCount)} frame${frameCount === 1 ? "" : "s"} | ${currentText}`;
  }
}

function renderRegionReferenceStrip() {
  renderRegionReferenceStripContent(regionReferenceStrip, activeRegion(), {
    emptyText: "No manual keyframe assignments yet.",
  });
}

function renderRegionReferenceStripContent(container, region, options = {}) {
  if (!container) return;
  const rows = manualRegionReferenceRows(region);
  container.textContent = "";
  container.classList.toggle("compact", Boolean(options.compact));
  container.hidden = !region;
  if (!region) return;

  const header = document.createElement("div");
  header.className = "region-reference-header";
  const title = document.createElement("span");
  title.textContent = "Manual frames";
  const count = document.createElement("span");
  count.textContent = rows.length ? `${formatCount(rows.length)}` : "none";
  header.append(title, count);
  container.appendChild(header);

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "region-reference-empty";
    empty.textContent = options.emptyText || "No manual keyframe assignments yet.";
    container.appendChild(empty);
    return;
  }

  const scroller = document.createElement("div");
  scroller.className = "region-reference-scroller";
  for (const row of rows) {
    const frame = frameById(row.frameId);
    const card = document.createElement("button");
    card.type = "button";
    card.className = "region-reference-card portrait";
    card.classList.toggle("active", state.selectedFrame && Number(state.selectedFrame.id) === Number(row.frameId));
    card.title = `Frame ${row.frameId} | ${row.proposalIds.length ? `source IDs ${row.proposalIds.join(", ")}` : "manual pixels"}`;

    const imageWrap = document.createElement("span");
    imageWrap.className = "region-reference-image";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = frame ? frame.imageName : `Frame ${row.frameId}`;
    image.src = frame ? frame.imageUrl : datasetUrl(`/api/frame/${row.frameId}/image`);
    const overlay = document.createElement("img");
    overlay.loading = "lazy";
    overlay.alt = "";
    overlay.src = singleRegionOverlayUrl(row.frameId, Number(region.id));

    const label = document.createElement("span");
    label.className = "region-reference-label";
    label.textContent = `Frame ${row.frameId}`;
    imageWrap.append(image, overlay, label);

    card.append(imageWrap);
    card.addEventListener("click", () => {
      const target = frameById(row.frameId);
      if (target) focusFrame(target);
    });
    scroller.appendChild(card);
  }
  container.appendChild(scroller);
}

function renderRegionPanel() {
  if (!regionList) return;
  const rows = sortedRegionRows();
  regionList.textContent = "";

  const title = idPanelSectionHeader(
    `Regions | ${formatCount(rows.length)}`,
    regionSortControl()
  );
  title.classList.add("region-section-title");
  regionList.appendChild(title);
  syncRegionSortControls();

  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "id-empty region-empty";
    empty.textContent = "Select pixels on a keyframe, then create a persistent region.";
    regionList.appendChild(empty);
    return;
  }

  for (const region of rows) {
    const id = Number(region.id) || 0;
    if (id <= 0) continue;
    const row = document.createElement("div");
    row.className = "id-row region-row";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.classList.toggle("active", id === state.activeRegionId);
    row.title = id === state.activeRegionId
      ? "Click again to deselect this persistent region."
      : "Click to make this the active persistent region.";

    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = regionColor(region);

    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("div");
    name.className = "id-name";
    name.textContent = `${region.name || `region_${id}`}`;
    const count = document.createElement("div");
    count.className = "id-count";
    const frameCount = Number(region.frameCount) || 0;
    const pixelCount = Number(region.pixelCount) || 0;
    count.textContent = `${formatCount(pixelCount)} px | ${formatCount(frameCount)} frame${frameCount === 1 ? "" : "s"}`;
    main.append(name, count);

    const selected = document.createElement("span");
    selected.className = "id-selected";
    selected.textContent = id === state.activeRegionId ? "active" : `R${id}`;

    row.append(swatch, main, selected);
    row.addEventListener("click", () => setActiveRegion(id));
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      setActiveRegion(id);
    });
    regionList.appendChild(row);
  }
}

function setActiveRegion(regionId) {
  const id = Number(regionId) || 0;
  if (id > 0 && id === state.activeRegionId) {
    closeActiveRegion();
    return;
  }
  state.activeRegionId = id;
  const region = activeRegion();
  if (region && regionNameInput) {
    regionNameInput.value = region.name || `region_${id}`;
  } else if (regionNameInput) {
    regionNameInput.value = "";
  }
  syncRegionControls();
  if (typeof syncPropagationControls === "function") syncPropagationControls();
  renderIdPanel();
  render();
}

function closeActiveRegion() {
  state.activeRegionId = 0;
  if (regionNameInput) regionNameInput.value = "";
  setSelectionStatus("Persistent region closed");
  syncRegionControls();
  if (typeof syncPropagationControls === "function") syncPropagationControls();
  renderIdPanel();
  render();
}

function syncRegionControls() {
  const rows = regionRows();
  const hasFrame = Boolean(state.selectedFrame && state.exactFrameView);
  const hasPixels = hasPixelSelection();
  const active = activeRegion();
  const activeOnFrame = Boolean(activeRegionFrameLabel());
  const frameHasRegionPixels = currentFrameHasRegionPixels();
  const frameComplete = currentFrameIsComplete();
  if (regionStatusEl) {
    if (!rows.length) {
      regionStatusEl.textContent = "No persistent regions";
    } else {
      const frameText = state.regionFrameInfo && state.regionFrameInfo.ready
        ? ` | frame has ${formatCount((state.regionFrameInfo.labels || []).length)}`
        : "";
      const completeCount = Number(state.regionStatus && state.regionStatus.completeFrameCount) || 0;
      const completeText = completeCount > 0 ? ` | ${formatCount(completeCount)} complete` : "";
      regionStatusEl.textContent = `${formatCount(rows.length)} region${rows.length === 1 ? "" : "s"}${frameText}${completeText}`;
    }
  }
  if (regionNameLabel) {
    regionNameLabel.textContent = active ? "Name" : "New name";
  }
  if (regionNameInput) {
    regionNameInput.placeholder = active ? "selected region name" : "new region name";
  }
  if (createRegionFromSelectionButton) {
    createRegionFromSelectionButton.disabled = state.regionBusy || !hasFrame || !hasPixels || Boolean(active);
    createRegionFromSelectionButton.title = active
      ? "Close the selected region before creating a new one."
      : "Create a new persistent region from selected pixels.";
  }
  if (assignSelectionToRegionButton) {
    assignSelectionToRegionButton.disabled = state.regionBusy || !hasFrame || !hasPixels || !active;
    assignSelectionToRegionButton.title = active
      ? "Assign selected pixels as this region on the current frame, replacing its old pixels on this frame."
      : "Select a persistent region first.";
  }
  if (addSelectionToRegionButton) {
    addSelectionToRegionButton.disabled = state.regionBusy || !hasFrame || !hasPixels || !active;
    addSelectionToRegionButton.title = active
      ? "Add selected pixels to this region on the current frame."
      : "Select a persistent region first.";
  }
  if (clearSelectionFromRegionsButton) {
    clearSelectionFromRegionsButton.disabled = state.regionBusy || !hasFrame || !hasPixels;
    clearSelectionFromRegionsButton.title = "Subtract selected pixels from confirmed region labels.";
  }
  if (clearRegionFrameButton) {
    clearRegionFrameButton.disabled = state.regionBusy || !hasFrame || !active || !activeOnFrame;
    clearRegionFrameButton.title = active
      ? activeOnFrame
        ? "Remove this whole region from the current frame only."
        : "This region is not present on the current frame."
      : "Select a persistent region first.";
  }
  if (renameRegionButton) {
    renameRegionButton.hidden = !active;
    renameRegionButton.disabled = state.regionBusy || !active || !regionNameInput || !regionNameInput.value.trim();
    renameRegionButton.title = active
      ? `Rename "${active.name || `region_${active.id}`}"`
      : "Select a persistent region first.";
  }
  if (deleteRegionButton) {
    deleteRegionButton.disabled = state.regionBusy || !active;
    deleteRegionButton.title = active
      ? "Delete this persistent region from all frames."
      : "Select a persistent region first.";
  }
  if (closeActiveRegionButton) {
    closeActiveRegionButton.disabled = state.regionBusy || !active;
  }
  if (showRegionsInput) {
    showRegionsInput.checked = state.showRegions;
  }
  if (frameCompleteInput) {
    frameCompleteInput.checked = frameComplete;
    frameCompleteInput.disabled = state.regionBusy || !hasFrame || !frameHasRegionPixels;
    frameCompleteInput.title = frameHasRegionPixels
      ? "Use this frame as a complete propagation anchor."
      : "Create at least one persistent region on this frame before marking it complete.";
  }
  updateRegionCompletionThumbnails();
  renderRegionSuggestionCard();
  renderSelectedRegionCard();
  renderRegionReferenceStrip();
}

async function refreshRegionStatus() {
  try {
    state.regionStatus = await loadJson("/api/regions/status");
  } catch (error) {
    console.error(error);
    state.regionStatus = {
      ready: false,
      failed: true,
      regions: [],
      message: error.message,
    };
  }
  if (state.activeRegionId > 0 && !activeRegion()) {
    state.activeRegionId = 0;
  }
  syncRegionControls();
  renderIdPanel();
  return state.regionStatus;
}

async function loadRegionFrameInfo(frameId) {
  let nextInfo = null;
  try {
    nextInfo = await loadJson(`/api/regions/frame/${frameId}/summary`);
  } catch (error) {
    console.error(error);
    nextInfo = null;
  }
  if (state.selectedFrame && Number(state.selectedFrame.id) !== Number(frameId)) {
    return false;
  }
  state.regionFrameInfo = nextInfo;
  syncRegionControls();
  if (typeof syncPropagationControls === "function") syncPropagationControls();
  return true;
}

async function setCurrentFrameComplete(complete) {
  if (state.regionBusy || !state.selectedFrame) return;
  const frameId = Number(state.selectedFrame.id);
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(complete ? `Marking frame ${frameId} complete` : `Marking frame ${frameId} partial`);
  try {
    const result = await postJson(`/api/regions/frame/${frameId}/complete`, {
      complete: Boolean(complete),
    });
    state.regionStatus = result.status || state.regionStatus;
    if (result.frame) state.regionFrameInfo = result.frame;
    if (!state.regionStatus) await refreshRegionStatus();
    updateRegionCompletionThumbnails();
    if (typeof syncPropagationControls === "function") syncPropagationControls();
    setSelectionStatus(complete ? `Frame ${frameId} is a complete anchor` : `Frame ${frameId} is partial`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
    await loadRegionFrameInfo(frameId);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function loadActiveRegionOverlay(frameId) {
  const isCurrentFrame = await loadRegionFrameInfo(frameId);
  if (!isCurrentFrame) return;
  if (!state.showRegions || !state.regionFrameInfo || !state.regionFrameInfo.ready) {
    state.regionOverlayImage = null;
    render();
    return;
  }
  const key = `${frameId}:${regionOverlayStamp()}:regions`;
  let image = state.regionOverlayImages.get(key);
  if (!image) {
    try {
      image = await loadImage(regionOverlayUrl(frameId));
      state.regionOverlayImages.set(key, image);
    } catch (error) {
      console.error(error);
      state.regionOverlayImage = null;
      render();
      return;
    }
  }
  if (state.selectedFrame && Number(state.selectedFrame.id) === Number(frameId)) {
    state.regionOverlayImage = image;
    render();
  }
}

function regionSelectionPayload(extra = {}) {
  if (!state.selectedFrame) return null;
  return {
    frameId: state.selectedFrame.id,
    selectionOps: cloneSelectionOps(state.selectionOps),
    protectRegions: state.selectionProtectRegions,
    ...extra,
  };
}

async function createPersistentRegionFromSelection() {
  if (state.regionBusy || !hasPixelSelection() || !state.selectedFrame) return;
  const payload = regionSelectionPayload({
    name: regionNameInput ? regionNameInput.value.trim() : "",
  });
  if (!payload) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus("Creating persistent region");
  try {
    const result = await postJson("/api/regions/create-from-selection", payload);
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    const region = result.region || {};
    state.activeRegionId = Number(region.id) || state.activeRegionId;
    if (regionNameInput && region.name) regionNameInput.value = region.name;
    state.regionOverlayImages.clear();
    await loadActiveRegionOverlay(state.selectedFrame.id);
    syncRegionControls();
    renderIdPanel();
    setSelectionStatus(`Created ${region.name || `region ${state.activeRegionId}`}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function assignSelectionToPersistentRegion() {
  if (state.regionBusy || !hasPixelSelection() || !state.selectedFrame || state.activeRegionId <= 0) return;
  const payload = regionSelectionPayload({
    regionId: state.activeRegionId,
  });
  if (!payload) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(`Replacing region ${state.activeRegionId} on this frame`);
  try {
    const result = await postJson("/api/regions/assign-selection", payload);
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    const region = result.region || activeRegion() || {};
    state.activeRegionId = Number(region.id) || state.activeRegionId;
    state.regionOverlayImages.clear();
    await loadActiveRegionOverlay(state.selectedFrame.id);
    syncRegionControls();
    renderIdPanel();
    setSelectionStatus(`Assigned ${region.name || `region ${state.activeRegionId}`} on this frame`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function addSelectionToPersistentRegion() {
  if (state.regionBusy || !hasPixelSelection() || !state.selectedFrame || state.activeRegionId <= 0) return;
  const payload = regionSelectionPayload({
    regionId: state.activeRegionId,
  });
  if (!payload) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(`Adding selection to region ${state.activeRegionId}`);
  try {
    const result = await postJson("/api/regions/add-selection", payload);
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    const region = result.region || activeRegion() || {};
    state.activeRegionId = Number(region.id) || state.activeRegionId;
    state.regionOverlayImages.clear();
    await loadActiveRegionOverlay(state.selectedFrame.id);
    syncRegionControls();
    renderIdPanel();
    setSelectionStatus(`Added pixels to ${region.name || `region ${state.activeRegionId}`}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function clearSelectionFromPersistentRegions() {
  if (state.regionBusy || !hasPixelSelection() || !state.selectedFrame) return;
  const payload = regionSelectionPayload();
  if (!payload) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus("Clearing selected pixels from confirmed regions");
  try {
    const result = await postJson("/api/regions/clear-selection", payload);
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    state.regionOverlayImages.clear();
    await loadActiveRegionOverlay(state.selectedFrame.id);
    syncRegionControls();
    renderIdPanel();
    const area = Number(result.assignment && result.assignment.areaPixels) || 0;
    setSelectionStatus(`Cleared ${area.toLocaleString()} px from confirmed regions`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function clearActiveRegionFromFrame() {
  const active = activeRegion();
  if (state.regionBusy || !active || !state.selectedFrame || !activeRegionFrameLabel()) return;
  const frameId = Number(state.selectedFrame.id);
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(`Clearing ${active.name || `region ${active.id}`} from frame ${frameId}`);
  try {
    const result = await postJson("/api/regions/clear-frame", {
      frameId,
      regionId: Number(active.id),
    });
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    state.regionOverlayImages.clear();
    await loadActiveRegionOverlay(frameId);
    syncRegionControls();
    renderIdPanel();
    const area = Number(result.areaPixels) || 0;
    setSelectionStatus(`Cleared ${area.toLocaleString()} px from this frame`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}

async function renameActiveRegion() {
  const active = activeRegion();
  if (state.regionBusy || !active || !regionNameInput) return;
  const name = regionNameInput.value.trim();
  if (!name) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(`Renaming region ${active.id}`);
  try {
    const result = await postJson("/api/regions/rename", {
      regionId: Number(active.id),
      name,
    });
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    const region = result.region || activeRegion();
    if (region) {
      state.activeRegionId = Number(region.id) || state.activeRegionId;
      regionNameInput.value = region.name || name;
    }
    syncRegionControls();
    renderIdPanel();
    setSelectionStatus(`Renamed ${region ? region.name : name}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
  }
}

async function deleteActiveRegion() {
  const active = activeRegion();
  if (state.regionBusy || !active) return;
  const frameCount = Number(active.frameCount) || 0;
  const pixelCount = Number(active.pixelCount) || 0;
  const ok = window.confirm(
    `Delete "${active.name || `region_${active.id}`}" everywhere?\n\n` +
      `This removes the persistent region and clears ${pixelCount.toLocaleString()} px from ${frameCount.toLocaleString()} frame${frameCount === 1 ? "" : "s"}.\n\n` +
      `Use "Clear Frame" if you only want to remove it from the current frame.`
  );
  if (!ok) return;
  state.regionBusy = true;
  syncRegionControls();
  setSelectionStatus(`Deleting ${active.name}`);
  try {
    const result = await postJson("/api/regions/delete", {
      regionId: Number(active.id),
    });
    state.regionStatus = result.status || null;
    if (!state.regionStatus) await refreshRegionStatus();
    state.activeRegionId = 0;
    if (regionNameInput) regionNameInput.value = "";
    state.regionOverlayImages.clear();
    if (state.selectedFrame) {
      await loadActiveRegionOverlay(state.selectedFrame.id);
    }
    syncRegionControls();
    renderIdPanel();
    setSelectionStatus(`Deleted ${active.name}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.regionBusy = false;
    syncRegionControls();
    render();
  }
}
