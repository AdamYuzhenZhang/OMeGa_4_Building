// Step 6: registry-driven 3D segmentation experiments and result display.
function segmentation3dMethods() {
  return state.segmentation3dStatus && Array.isArray(state.segmentation3dStatus.methods)
    ? state.segmentation3dStatus.methods
    : [];
}

function selectedSegmentation3dMethodId() {
  return String(segmentation3dMethodInput ? segmentation3dMethodInput.value : "");
}

function selectedSegmentation3dInputId() {
  return String(segmentation3dInput ? segmentation3dInput.value : "");
}

function selectedSegmentation3dInput() {
  return segmentation3dInputs().find((row) => row.inputId === selectedSegmentation3dInputId()) || null;
}

function selectedSegmentation3dSourceId() {
  return String(segmentation3dSourceInput ? segmentation3dSourceInput.value : "");
}

function selectedSegmentation3dManualWeight() {
  return Number(segmentation3dManualWeightInput ? segmentation3dManualWeightInput.value : 1) || 1;
}

function segmentation3dInputs(methodId = selectedSegmentation3dMethodId()) {
  const inputs = state.segmentation3dStatus && state.segmentation3dStatus.inputs;
  return inputs && Array.isArray(inputs[methodId]) ? inputs[methodId] : [];
}

function segmentation3dRunId() {
  const methodId = selectedSegmentation3dMethodId();
  const inputId = selectedSegmentation3dInputId();
  if (!methodId || !inputId) return "";
  const input = selectedSegmentation3dInput();
  if (!input || !Array.isArray(input.sourceOptions) || input.sourceOptions.length === 0) {
    return `${methodId}_${inputId}`;
  }
  const sourceId = selectedSegmentation3dSourceId();
  return sourceId
    ? `${methodId}_${inputId}_${sourceId}_w${selectedSegmentation3dManualWeight()}`
    : "";
}

function savedSegmentation3dRun(runId = segmentation3dRunId()) {
  return savedSegmentation3dRuns().find((row) => String(row.runId) === String(runId)) || null;
}

function savedSegmentation3dRuns() {
  return state.segmentation3dStatus && Array.isArray(state.segmentation3dStatus.runs)
    ? state.segmentation3dStatus.runs
    : [];
}

function sai3dSegmentationRuns() {
  return savedSegmentation3dRuns().filter((run) => String(run.methodId) === "sai3d");
}

function splitSplatSegmentationRuns() {
  return savedSegmentation3dRuns().filter(
    (run) => String(run.experimentFamily) === "split_splat",
  );
}

function mapAnything3dgsRuns() {
  return savedSegmentation3dRuns().filter(
    (run) => String(run.experimentFamily) === "mapanything_region_3dgs",
  );
}

function mapAnythingPipelineRuns() {
  const rows = state.segmentation3dStatus &&
    state.segmentation3dStatus.reconstructionPipelines;
  return Array.isArray(rows) ? rows : [];
}

function segmentation3dResultLayerKey(runId) {
  return `segmentation3d:${String(runId)}`;
}

function segmentation3dResultLayer(run, create = true) {
  const key = segmentation3dResultLayerKey(run.runId);
  let layer = state.evidencePointClouds[key];
  if (!layer && create) {
    layer = {
      visible: false,
      mode: "result",
      result: null,
      positions: new Float32Array(),
      colors: new Uint8Array(),
      busy: false,
      displayName: segmentation3dRunDisplayName(run),
    };
    state.evidencePointClouds[key] = layer;
  }
  if (layer) layer.displayName = segmentation3dRunDisplayName(run);
  return layer || null;
}

function segmentation3dSourceDisplayName(sourceId) {
  for (const rows of Object.values((state.segmentation3dStatus && state.segmentation3dStatus.inputs) || {})) {
    for (const input of Array.isArray(rows) ? rows : []) {
      const source = Array.isArray(input.sourceOptions)
        ? input.sourceOptions.find((row) => String(row.sourceId) === String(sourceId))
        : null;
      if (source) return source.displayName || source.sourceId;
    }
  }
  return String(sourceId || "Propagation");
}

function segmentation3dRunDisplayName(run) {
  if (run.displayName) return String(run.displayName);
  const geometryName = String(run.geometrySource) === "omega_final_clean_hybrid"
    ? "Hybrid"
    : "Vertices";
  if (String(run.inputId) === "sam2_auto") return `SAM2 Automatic · ${geometryName}`;
  if (run.sourceId) {
    return `${segmentation3dSourceDisplayName(run.sourceId)} · ${Number(run.manualFrameWeight) || 1}x · ${geometryName}`;
  }
  const inputs = segmentation3dInputs(String(run.methodId));
  const input = inputs.find((row) => String(row.inputId) === String(run.inputId));
  const inputName = input ? (input.displayName || input.inputId) : String(run.inputId || run.runId);
  return `${inputName} · ${geometryName}`;
}

function segmentation3dPointBudgetLabel(value) {
  const budget = Number(value) || 0;
  return budget > 0 ? formatCount(budget) : "All";
}

function segmentation3dGaussianArtifacts(run) {
  const artifacts = run && Array.isArray(run.gaussianArtifacts)
    ? run.gaussianArtifacts
    : [];
  return artifacts
    .map((artifact) => (
      String(artifact.variantId) === "segments"
        ? { ...artifact, displayName: "Instance ID Colors" }
        : artifact
    ));
}

function splitSplatArtifactStatus(artifact) {
  const pointCount = Number(artifact.pointCount) || 0;
  const labelCount = Number(artifact.labelCount) || 0;
  const labeledPointCount = Number(artifact.labeledPointCount) || 0;
  if (labelCount > 0) {
    const coverage = pointCount > 0
      ? Math.round(100 * labeledPointCount / pointCount)
      : 0;
    return `${formatCount(labelCount)} IDs · ${coverage}% labeled`;
  }
  return pointCount ? `${formatCount(pointCount)} splats` : "Ready";
}

function removeSegmentation3dResultLayer(runId) {
  delete state.evidencePointClouds[segmentation3dResultLayerKey(runId)];
}

function syncSegmentation3dResultList() {
  if (!segmentation3dResultsEl) return;
  const runs = sai3dSegmentationRuns();
  const pointRuns = [
    ...runs,
    ...mapAnything3dgsRuns().filter(
      (run) => String(run.artifactRole) === "split_labels",
    ),
  ];
  const validKeys = new Set(pointRuns.map((run) => segmentation3dResultLayerKey(run.runId)));
  for (const key of Object.keys(state.evidencePointClouds)) {
    if (key.startsWith("segmentation3d:") && !validKeys.has(key)) delete state.evidencePointClouds[key];
  }
  segmentation3dResultsEl.replaceChildren();
  if (segmentation3dResultSummaryEl) {
    segmentation3dResultSummaryEl.textContent = runs.length
      ? `${runs.length} saved`
      : "No saved runs";
  }
  if (!runs.length) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = state.segmentation3dBusy ? "SAI3D is running" : "Run SAI3D in Step 6";
    segmentation3dResultsEl.appendChild(empty);
    syncReconstructionResultLists();
    syncMapAnything3dgsResultList();
    return;
  }

  for (const run of runs) {
    const layer = segmentation3dResultLayer(run);
    const row = document.createElement("label");
    row.className = "control checkbox point-source-row";
    row.title = [
      run.runDir || "",
      `${segmentation3dPointBudgetLabel(run.pointBudget)} points`,
      `${formatCount(Number(run.superpointTarget) || 0)} superpoints`,
    ].filter(Boolean).join(" | ");

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(layer.visible);
    input.disabled = Boolean(layer.busy);
    input.addEventListener("change", () => {
      setSegmentation3dResultVisible(run.runId, input.checked).catch((error) => {
        console.error(error);
        layer.visible = false;
        syncSegmentation3dResultList();
      });
    });

    const name = document.createElement("span");
    name.className = "point-source-name";
    name.textContent = segmentation3dRunDisplayName(run);

    const status = document.createElement("span");
    status.className = "step-status";
    status.textContent = layer.busy
      ? "Loading"
      : layer.visible && layer.result
        ? `${formatCount(layer.result.servedPointCount)} shown`
        : `${segmentation3dPointBudgetLabel(run.pointBudget)} / ${formatCount(Number(run.superpointTarget) || 0)}`;

    row.append(input, name, status);
    segmentation3dResultsEl.appendChild(row);
  }
  syncReconstructionResultLists();
  syncMapAnything3dgsResultList();
}

function fillSegmentation3dSelect(
  select,
  values,
  selectedValue,
  labelFor,
  disabledFor = () => false,
  valueFor = (value) => value,
) {
  if (!select) return;
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = String(valueFor(value));
    option.textContent = labelFor(value);
    option.disabled = Boolean(disabledFor(value));
    select.appendChild(option);
  }
  const preferred = [...select.options].find((option) => option.value === String(selectedValue) && !option.disabled);
  const fallback = [...select.options].find((option) => !option.disabled);
  select.value = preferred ? preferred.value : fallback ? fallback.value : "";
}

function initializeSegmentation3dControls(status) {
  state.segmentation3dStatus = status || {};
  const methodId = state.segmentation3dStatus.defaultMethodId || "";
  fillSegmentation3dSelect(
    segmentation3dMethodInput,
    segmentation3dMethods(),
    methodId,
    (row) => row.displayName || row.methodId,
    (row) => !row.available,
    (row) => row.methodId,
  );
  refreshSegmentation3dInputOptions();
  refreshSegmentation3dInputSettings();
  fillSegmentation3dSelect(
    segmentation3dPointBudgetInput,
    state.segmentation3dStatus.pointBudgetOptions || [],
    state.segmentation3dStatus.defaultPointBudget,
    (value) => Number(value) === 0 ? "All points" : formatCount(Number(value)),
  );
  fillSegmentation3dSelect(
    segmentation3dSuperpointsInput,
    state.segmentation3dStatus.superpointTargetOptions || [],
    state.segmentation3dStatus.defaultSuperpointTarget,
    (value) => formatCount(Number(value)),
  );
  const running = Array.isArray(state.segmentation3dStatus.jobs)
    ? state.segmentation3dStatus.jobs.find((job) => job.running)
    : null;
  if (running) {
    state.segmentation3dBusy = true;
    state.segmentation3dJobId = String(running.jobId || "");
    startSegmentation3dPolling();
  }
  syncSegmentation3dResultList();
  syncSegmentation3dControls();
  scheduleMapAnythingPipelinePolling();
}

function refreshSegmentation3dInputOptions() {
  const previous = segmentation3dInput ? segmentation3dInput.value : "";
  const preferred = previous || (state.segmentation3dStatus && state.segmentation3dStatus.defaultInputId) || "";
  fillSegmentation3dSelect(
    segmentation3dInput,
    segmentation3dInputs(),
    preferred,
    (row) => row.displayName || row.inputId,
    (row) => !row.implemented || !row.ready,
    (row) => row.inputId,
  );
}

function refreshSegmentation3dInputSettings() {
  const input = selectedSegmentation3dInput();
  const sources = input && Array.isArray(input.sourceOptions) ? input.sourceOptions : [];
  const hasSources = sources.length > 0;
  if (segmentation3dSourceField) segmentation3dSourceField.hidden = !hasSources;
  if (segmentation3dManualWeightField) segmentation3dManualWeightField.hidden = !hasSources;
  if (!hasSources) {
    if (segmentation3dSourceInput) segmentation3dSourceInput.replaceChildren();
    if (segmentation3dManualWeightInput) segmentation3dManualWeightInput.replaceChildren();
    return;
  }
  const previousSource = selectedSegmentation3dSourceId();
  fillSegmentation3dSelect(
    segmentation3dSourceInput,
    sources,
    previousSource || input.defaultSourceId,
    (row) => row.displayName || row.sourceId,
    (row) => !row.ready,
    (row) => row.sourceId,
  );
  const weights = Array.isArray(input.manualWeightOptions) ? input.manualWeightOptions : [];
  const previousWeight = selectedSegmentation3dManualWeight();
  fillSegmentation3dSelect(
    segmentation3dManualWeightInput,
    weights,
    weights.includes(previousWeight) ? previousWeight : input.defaultManualWeight,
    (value) => `${Number(value)}x`,
  );
}

function syncSegmentation3dControls(job = null) {
  const method = segmentation3dMethods().find((row) => row.methodId === selectedSegmentation3dMethodId());
  const input = selectedSegmentation3dInput();
  const source = input && Array.isArray(input.sourceOptions)
    ? input.sourceOptions.find((row) => row.sourceId === selectedSegmentation3dSourceId())
    : null;
  const run = savedSegmentation3dRun();
  const hasSources = Boolean(input && Array.isArray(input.sourceOptions) && input.sourceOptions.length);
  const sourceReady = !hasSources || Boolean(source && source.ready);
  const runnable = Boolean(method && method.available && input && input.implemented && input.ready && sourceReady);
  if (runSegmentation3dButton) runSegmentation3dButton.disabled = state.segmentation3dBusy || !runnable;
  syncSegmentation3dResultList();
  if (!segmentation3dStatusEl) return;
  if (job && job.running) {
    segmentation3dStatusEl.textContent = `${job.stageIndex || 0}/${job.stageCount || 3} ${job.message || "Running"}`;
  } else if (job && job.failed) {
    segmentation3dStatusEl.textContent = job.message || "Experiment failed";
  } else if (run) {
    segmentation3dStatusEl.textContent = `${method ? method.displayName : run.methodId} result ready`;
  } else if (source && !source.ready) {
    segmentation3dStatusEl.textContent = source.message || "Selected propagation is not ready";
  } else if (input) {
    segmentation3dStatusEl.textContent = input.message || "Ready to run";
  } else {
    segmentation3dStatusEl.textContent = "No compatible input";
  }
  const section = segmentation3dStatusEl.closest(".pipeline-step");
  if (section) {
    section.classList.toggle("running", state.segmentation3dBusy);
    section.classList.toggle("ready", Boolean(run) && !state.segmentation3dBusy);
    section.classList.toggle("failed", Boolean(job && job.failed));
  }
}

async function refreshSegmentation3dStatus() {
  const status = await loadJson("/api/3d-segmentation/status");
  const methodId = selectedSegmentation3dMethodId();
  const inputId = selectedSegmentation3dInputId();
  state.segmentation3dStatus = status;
  fillSegmentation3dSelect(
    segmentation3dMethodInput,
    segmentation3dMethods(),
    methodId || status.defaultMethodId,
    (row) => row.displayName || row.methodId,
    (row) => !row.available,
    (row) => row.methodId,
  );
  fillSegmentation3dSelect(
    segmentation3dInput,
    segmentation3dInputs(),
    inputId || status.defaultInputId,
    (row) => row.displayName || row.inputId,
    (row) => !row.implemented || !row.ready,
    (row) => row.inputId,
  );
  refreshSegmentation3dInputSettings();
  syncSegmentation3dControls();
}

async function runSegmentation3d() {
  if (state.segmentation3dBusy) return;
  const requestedRunId = segmentation3dRunId();
  removeSegmentation3dResultLayer(requestedRunId);
  if (state.segmentation3dStatus && Array.isArray(state.segmentation3dStatus.runs)) {
    state.segmentation3dStatus.runs = state.segmentation3dStatus.runs.filter(
      (run) => String(run.runId) !== String(requestedRunId)
    );
  }
  state.segmentation3dBusy = true;
  syncSegmentation3dControls({ running: true, message: "Starting experiment", stageIndex: 0, stageCount: 3 });
  try {
    const job = await postJson("/api/3d-segmentation/start", {
      methodId: selectedSegmentation3dMethodId(),
      inputId: selectedSegmentation3dInputId(),
      sourceId: selectedSegmentation3dSourceId(),
      manualFrameWeight: selectedSegmentation3dManualWeight(),
      pointBudget: Number(segmentation3dPointBudgetInput.value),
      superpointTarget: Number(segmentation3dSuperpointsInput.value),
    });
    state.segmentation3dJobId = String(job.jobId || "");
    state.segmentation3dRunId = String(job.runId || "");
    startSegmentation3dPolling();
  } catch (error) {
    state.segmentation3dBusy = false;
    syncSegmentation3dControls({ failed: true, message: error.message });
  }
}

function startSegmentation3dPolling() {
  if (state.segmentation3dPollTimer) clearTimeout(state.segmentation3dPollTimer);
  const poll = async () => {
    try {
      const job = await loadJson(`/api/3d-segmentation/jobs/${encodeURIComponent(state.segmentation3dJobId)}`);
      state.segmentation3dBusy = Boolean(job.running);
      state.segmentation3dRunId = String(job.runId || state.segmentation3dRunId || "");
      syncSegmentation3dControls(job);
      if (job.running) {
        state.segmentation3dPollTimer = setTimeout(poll, 1200);
        return;
      }
      await refreshSegmentation3dStatus();
    } catch (error) {
      state.segmentation3dBusy = false;
      syncSegmentation3dControls({ failed: true, message: error.message });
    }
  };
  state.segmentation3dPollTimer = setTimeout(poll, 250);
}

async function setSegmentation3dResultVisible(runId, visible) {
  const run = savedSegmentation3dRun(runId);
  if (!run) return;
  const layer = segmentation3dResultLayer(run);
  if (!visible) {
    layer.visible = false;
    syncSegmentation3dResultList();
    render();
    return;
  }
  deactivateGaussianViewport({ redraw: false });
  layer.busy = true;
  syncSegmentation3dResultList();
  try {
    if (!layer.result) {
      const payload = await loadJson(`/api/3d-segmentation/runs/${encodeURIComponent(run.runId)}/points`);
      layer.result = {
        positions: Float32Array.from(payload.positions || []),
        colors: Uint8Array.from(payload.colors || []),
        pointCount: Number(payload.pointCount) || 0,
        servedPointCount: Number(payload.servedPointCount) || 0,
      };
      if (layer.result.positions.length !== layer.result.colors.length) {
        throw new Error(`${run.runId} point positions and RGB colors have different lengths.`);
      }
    }
    layer.positions = layer.result.positions;
    layer.colors = layer.result.colors;
    layer.visible = true;
  } finally {
    layer.busy = false;
    syncSegmentation3dResultList();
    render();
  }
}
