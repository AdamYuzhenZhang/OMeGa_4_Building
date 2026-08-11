// Shared ObjectGS progress, calibrated renders, masks, and object-aware anchors.
function objectgsFrameMode(runId) {
  return `objectgs:${String(runId)}`;
}

function objectgsMaskLayer(run) {
  const key = normalizeProposalLayer(run.maskLayerKey || "");
  const rows = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  return rows.find((row) => normalizeProposalLayer(row.key || "") === key) || null;
}

function syncObjectgsBackgroundOptions() {
  if (!frameBackgroundModeInput) return;
  for (const option of [...frameBackgroundModeInput.options]) {
    if (option.dataset.objectgs === "true") option.remove();
  }
  const readyRuns = objectgsSegmentationRuns().filter((run) => run.rgbRendersReady);
  for (const run of readyRuns) {
    const option = document.createElement("option");
    option.value = objectgsFrameMode(run.runId);
    option.textContent = `ObjectGS · ${run.baseRunId || run.objectgsRunId || "Joint"}`;
    option.dataset.objectgs = "true";
    frameBackgroundModeInput.appendChild(option);
  }
  const available = new Set(readyRuns.map((run) => objectgsFrameMode(run.runId)));
  if (state.frameBackgroundMode.startsWith("objectgs:") && !available.has(state.frameBackgroundMode)) {
    state.frameBackgroundMode = "rgb";
  }
  frameBackgroundModeInput.value = state.frameBackgroundMode || "rgb";
}

function setObjectgsFrameMode(run, enabled) {
  state.frameBackgroundMode = enabled ? objectgsFrameMode(run.runId) : "rgb";
  syncObjectgsBackgroundOptions();
  if (state.selectedFrame) loadSelectedFrameImage(state.selectedFrame);
  syncObjectgsResultList();
  render();
}

function objectgsRgbControl(run) {
  const row = document.createElement("label");
  row.className = "gaussian-artifact-row";
  row.title = "Show the final ObjectGS neural scene rendered at the active calibrated capture camera.";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = state.frameBackgroundMode === objectgsFrameMode(run.runId);
  input.addEventListener("change", () => setObjectgsFrameMode(run, input.checked));
  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = "Final Neural RGB (Views)";
  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = input.checked ? "Visible" : "Ready";
  row.append(input, name, status);
  return row;
}

function objectgsAnchorControl(run) {
  const layer = segmentation3dResultLayer(run);
  const row = document.createElement("label");
  row.className = "control checkbox point-source-row";
  row.title = "Learned ObjectGS scaffold anchors colored by persistent region. Anchors generate view-dependent Gaussians and are not explicit 3DGS splats.";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(layer.visible);
  input.disabled = Boolean(layer.busy);
  input.addEventListener("change", () => {
    if (input.checked) {
      for (const other of objectgsSegmentationRuns()) {
        if (other.runId === run.runId) continue;
        const otherLayer = segmentation3dResultLayer(other, false);
        if (otherLayer) otherLayer.visible = false;
      }
    }
    setSegmentation3dResultVisible(run.runId, input.checked)
      .then(syncObjectgsResultList)
      .catch((error) => {
        console.error(error);
        layer.visible = false;
        setSelectionStatus(error.message);
        syncObjectgsResultList();
      });
  });
  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = "Neural Anchor Scaffold";
  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = layer.busy
    ? "Loading"
    : layer.visible && layer.result
      ? `${formatCount(layer.result.servedPointCount)} shown`
      : "Ready";
  row.append(input, name, status);
  return row;
}

function objectgsMeshControl(run, artifact) {
  const active = Boolean(
    state.meshViewportVisible &&
    state.meshViewportRunId === run.runId &&
    state.meshViewportVariantId === artifact.variantId
  );
  const row = document.createElement("label");
  row.className = "gaussian-artifact-row";
  row.title = "Official ObjectGS per-object RGB-D rendering and bounded TSDF meshes.";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = active;
  input.addEventListener("change", () => {
    const action = input.checked
      ? activateMeshViewport(run.runId, artifact.variantId)
      : Promise.resolve(deactivateMeshViewport());
    action.then(syncObjectgsResultList).catch((error) => {
      console.error(error);
      state.meshViewportError = error.message;
      setSelectionStatus(error.message);
      syncObjectgsResultList();
    });
  });
  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = artifact.displayName || "Object Meshes";
  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = active
    ? state.meshViewportReady
      ? `${formatCount(artifact.triangleCount)} faces`
      : `${Math.round(state.meshViewportProgress || 0)}%`
    : `${artifact.partCount} parts`;
  row.append(input, name, status);
  return row;
}

function objectgsRunKey(mapanythingRunId, objectgsRunId) {
  return `${String(mapanythingRunId || "")}/${String(objectgsRunId || "")}`;
}

function syncObjectgsResultList() {
  if (!objectgsResultsEl) return;
  syncObjectgsBackgroundOptions();
  const groups = new Map();
  const requireGroup = (mapanythingRunId, objectgsRunId) => {
    const key = objectgsRunKey(mapanythingRunId, objectgsRunId);
    const group = groups.get(key) || {
      key,
      mapanythingRunId,
      objectgsRunId,
      pipeline: null,
      result: null,
    };
    groups.set(key, group);
    return group;
  };
  for (const pipeline of objectgsPipelineRuns()) {
    requireGroup(pipeline.mapanythingRunId, pipeline.runId).pipeline = pipeline;
  }
  for (const run of objectgsSegmentationRuns()) {
    requireGroup(run.baseRunId, run.objectgsRunId || run.runId).result = run;
  }

  objectgsResultsEl.replaceChildren();
  const runningCount = [...groups.values()].filter(
    (group) => group.pipeline && group.pipeline.running,
  ).length;
  const readyCount = [...groups.values()].filter((group) => group.result).length;
  if (objectgsResultSummaryEl) {
    objectgsResultSummaryEl.textContent = runningCount
      ? `${runningCount} running · ${readyCount} ready`
      : groups.size
        ? `${groups.size} run${groups.size === 1 ? "" : "s"} · ${readyCount} ready`
        : "No saved runs";
  }
  if (!groups.size) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = "Run ObjectGS from a completed MapAnything Split";
    objectgsResultsEl.appendChild(empty);
    return;
  }

  for (const groupData of [...groups.values()].sort((a, b) => a.key.localeCompare(b.key))) {
    const group = document.createElement("section");
    group.className = "gaussian-run objectgs-run";
    const title = document.createElement("div");
    title.className = "gaussian-run-heading";
    const name = document.createElement("span");
    name.className = "gaussian-run-title";
    name.textContent = groupData.pipeline && groupData.pipeline.displayName
      ? groupData.pipeline.displayName
      : `ObjectGS Joint · ${groupData.mapanythingRunId}`;
    title.appendChild(name);
    if (groupData.pipeline) {
      const stateLabel = document.createElement("span");
      stateLabel.className = `mapanything-run-state ${groupData.pipeline.status || "paused"}`;
      stateLabel.textContent = groupData.pipeline.status || "paused";
      title.appendChild(stateLabel);
    }
    group.appendChild(title);
    if (groupData.pipeline) {
      group.appendChild(mapAnythingPipelineStatusControl(groupData.pipeline));
    }

    appendSplitSplatSection(group, "Joint Reconstruction");
    if (!groupData.result) {
      group.appendChild(mapAnythingExpectedRow("Final Neural RGB (Views)", "After export"));
      group.appendChild(mapAnythingExpectedRow("Rendered Region IDs", "After export"));
      group.appendChild(mapAnythingExpectedRow("Neural Anchor Scaffold", "After export"));
    } else {
      if (groupData.result.rgbRendersReady) {
        group.appendChild(objectgsRgbControl(groupData.result));
      } else {
        group.appendChild(mapAnythingExpectedRow("Final Neural RGB (Views)", "Unavailable"));
      }
      const maskLayer = objectgsMaskLayer(groupData.result);
      if (maskLayer) {
        group.appendChild(splitSplatMaskControl(maskLayer, "Rendered Region IDs"));
      } else {
        group.appendChild(mapAnythingExpectedRow("Rendered Region IDs", "Loading registry"));
      }
      group.appendChild(objectgsAnchorControl(groupData.result));
      const meshArtifacts = Array.isArray(groupData.result.meshArtifacts)
        ? groupData.result.meshArtifacts
        : [];
      if (meshArtifacts.length) {
        for (const artifact of meshArtifacts) {
          group.appendChild(objectgsMeshControl(groupData.result, artifact));
        }
      } else {
        group.appendChild(mapAnythingExpectedRow("Official Object Meshes", "Run mesh stage"));
      }
    }
    objectgsResultsEl.appendChild(group);
  }
}

function activeObjectgsAnchorLayer() {
  for (const run of objectgsSegmentationRuns()) {
    const layer = segmentation3dResultLayer(run, false);
    if (layer && layer.visible && layer.result) return { run, layer };
  }
  return null;
}

function setObjectgsRegionVisible(layer, regionId, visible) {
  layer.regionVisibility[String(regionId)] = Boolean(visible);
  renderIdPanel();
  render();
}

function setAllObjectgsRegionsVisible(layer, visible) {
  for (const region of layer.result.regions || []) {
    layer.regionVisibility[String(Number(region.id) || 0)] = Boolean(visible);
  }
  renderIdPanel();
  render();
}

function renderObjectgsAnchorPanel() {
  if (!idPanel || !regionList || !idList) return false;
  const active = activeObjectgsAnchorLayer();
  if (!active) return false;
  const { run, layer } = active;
  const regions = Array.isArray(layer.result.regions) ? layer.result.regions : [];
  const visibleCount = regions.filter(
    (region) => layer.regionVisibility[String(Number(region.id) || 0)] !== false,
  ).length;

  idPanel.classList.add("gaussian-part-mode");
  regionList.classList.add("gaussian-part-list");
  regionList.replaceChildren();
  idList.replaceChildren();

  const allControl = document.createElement("label");
  allControl.className = "gaussian-all-control";
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = visibleCount === regions.length;
  allInput.indeterminate = visibleCount > 0 && visibleCount < regions.length;
  allInput.addEventListener("change", () => {
    setAllObjectgsRegionsVisible(layer, allInput.checked);
  });
  const allLabel = document.createElement("span");
  allLabel.textContent = "All";
  allControl.append(allInput, allLabel);
  regionList.appendChild(idPanelSectionHeader(
    `Object Anchors | ${visibleCount}/${regions.length}`,
    allControl,
  ));

  const sceneName = document.createElement("div");
  sceneName.className = "gaussian-scene-name";
  sceneName.textContent = run.displayName || "Shared ObjectGS";
  regionList.appendChild(sceneName);

  for (const region of regions) {
    const regionId = Number(region.id) || 0;
    const checked = layer.regionVisibility[String(regionId)] !== false;
    const row = document.createElement("label");
    row.className = `id-row gaussian-part-row${checked ? " active" : ""}`;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    input.addEventListener("change", () => {
      setObjectgsRegionVisible(layer, regionId, input.checked);
    });
    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    const color = Array.isArray(region.color) ? region.color : [];
    swatch.style.background = color.length === 3
      ? `rgb(${color.join(", ")})`
      : palette(regionId);
    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("span");
    name.className = "id-name";
    name.textContent = regionId > 0 ? `Region ${regionId}` : "Unknown";
    const regionName = document.createElement("span");
    regionName.className = "gaussian-region-name";
    regionName.textContent = String(region.name || "");
    const count = document.createElement("span");
    count.className = "id-count";
    count.textContent = `${formatCount(Number(region.pointCount) || 0)} anchors`;
    main.appendChild(name);
    if (regionName.textContent && regionName.textContent !== name.textContent) {
      main.appendChild(regionName);
    }
    main.appendChild(count);
    row.append(input, swatch, main);
    regionList.appendChild(row);
  }
  return true;
}
