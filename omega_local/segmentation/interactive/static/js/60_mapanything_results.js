// MapAnything reconstruction progress and result controls.
let mapAnythingPipelinePollTimer = null;
function mapAnythingPipelineStatusControl(run) {
  const wrapper = document.createElement("div");
  wrapper.className = `mapanything-pipeline-status ${run.status || "paused"}`;

  const stages = document.createElement("div");
  stages.className = "mapanything-stage-track";
  for (const stage of Array.isArray(run.stages) ? run.stages : []) {
    const item = document.createElement("span");
    item.className = `mapanything-stage ${stage.status || "pending"}`;
    item.textContent = stage.displayName || stage.stageId;
    item.title = `${stage.displayName || stage.stageId}: ${stage.status || "pending"}`;
    stages.appendChild(item);
  }

  const bar = document.createElement("div");
  bar.className = "mapanything-progress-track";
  const fill = document.createElement("span");
  fill.style.width = `${Math.max(0, Math.min(100, Number(run.overallProgress) || 0))}%`;
  bar.appendChild(fill);

  const detail = document.createElement("div");
  detail.className = "mapanything-progress-detail";
  const message = document.createElement("span");
  message.textContent = run.message || "Waiting";
  message.title = run.detailMessage || message.textContent;
  const percent = document.createElement("span");
  percent.textContent = `${Math.round(Number(run.overallProgress) || 0)}%`;
  detail.append(message, percent);
  wrapper.append(stages, bar, detail);
  return wrapper;
}

function mapAnythingExpectedRow(label, status) {
  const row = document.createElement("div");
  row.className = "mapanything-expected-row";
  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = label;
  const value = document.createElement("span");
  value.className = "step-status";
  value.textContent = status;
  row.append(name, value);
  return row;
}

function mapAnything3dgsPointControl(run) {
  const layer = segmentation3dResultLayer(run);
  const row = document.createElement("label");
  row.className = "control checkbox point-source-row";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(layer.visible);
  input.disabled = Boolean(layer.busy);
  input.addEventListener("change", () => {
    setSegmentation3dResultVisible(run.runId, input.checked)
      .then(syncMapAnything3dgsResultList)
      .catch((error) => {
        console.error(error);
        layer.visible = false;
        setSelectionStatus(error.message);
        syncMapAnything3dgsResultList();
      });
  });
  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = "Segmented MapAnything Points";
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

function syncMapAnything3dgsResultList() {
  if (!mapAnything3dgsResultsEl) return;
  const groups = new Map();
  const requireGroup = (baseRunId) => {
    const group = groups.get(baseRunId) || {
      baseRunId,
      pipelineRun: null,
      splitRun: null,
      reconstructionRun: null,
    };
    groups.set(baseRunId, group);
    return group;
  };
  for (const pipelineRun of mapAnythingPipelineRuns()) {
    const baseRunId = String(pipelineRun.runId || "");
    if (baseRunId) requireGroup(baseRunId).pipelineRun = pipelineRun;
  }
  for (const run of mapAnything3dgsRuns()) {
    const baseRunId = String(run.baseRunId || "");
    if (!baseRunId) continue;
    const group = requireGroup(baseRunId);
    if (String(run.artifactRole) === "split_labels") group.splitRun = run;
    if (String(run.artifactRole) === "reconstruction") group.reconstructionRun = run;
  }

  mapAnything3dgsResultsEl.replaceChildren();
  if (mapAnything3dgsResultSummaryEl) {
    const runningCount = [...groups.values()].filter(
      (group) => group.pipelineRun && group.pipelineRun.running,
    ).length;
    const reconstructionCount = [...groups.values()].filter(
      (group) => group.reconstructionRun,
    ).length;
    mapAnything3dgsResultSummaryEl.textContent = runningCount
      ? `${runningCount} running · ${reconstructionCount} recon`
      : groups.size
        ? `${groups.size} run${groups.size === 1 ? "" : "s"} · ${reconstructionCount} recon`
        : "No saved runs";
  }
  if (!groups.size) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = "Run MapAnything Split to generate region points";
    mapAnything3dgsResultsEl.appendChild(empty);
    return;
  }

  for (const groupData of [...groups.values()].sort(
    (a, b) => a.baseRunId.localeCompare(b.baseRunId),
  )) {
    const group = document.createElement("section");
    group.className = "gaussian-run mapanything-run";
    const title = document.createElement("div");
    title.className = "gaussian-run-heading";
    const name = document.createElement("span");
    name.className = "gaussian-run-title";
    name.textContent = groupData.pipelineRun && groupData.pipelineRun.displayName
      ? groupData.pipelineRun.displayName
      : groupData.baseRunId
        .split("_")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
    title.appendChild(name);
    if (groupData.pipelineRun) {
      const status = document.createElement("span");
      status.className = `mapanything-run-state ${groupData.pipelineRun.status || "paused"}`;
      status.textContent = groupData.pipelineRun.status || "paused";
      title.appendChild(status);
    }
    group.appendChild(title);
    if (groupData.pipelineRun) {
      group.appendChild(mapAnythingPipelineStatusControl(groupData.pipelineRun));
    }

    const cleanLayer = splitSplatLayer(groupData.baseRunId, "_split");
    const supportLayer = splitSplatLayer(groupData.baseRunId, "_support");
    appendSplitSplatSection(group, "Split Outputs");
    if (cleanLayer) {
      group.appendChild(splitSplatMaskControl(cleanLayer, "Identity-Cleaned Masks"));
    } else {
      group.appendChild(mapAnythingExpectedRow("Identity-Cleaned Masks", "Waiting"));
    }
    if (supportLayer) {
      group.appendChild(splitSplatMaskControl(supportLayer, "Projected Point Support"));
    } else {
      group.appendChild(mapAnythingExpectedRow("Projected Point Support", "Waiting"));
    }
    if (groupData.splitRun) {
      group.appendChild(mapAnything3dgsPointControl(groupData.splitRun));
    } else {
      group.appendChild(mapAnythingExpectedRow("Segmented MapAnything Points", "Waiting"));
    }

    appendSplitSplatSection(group, "Trained 3DGS");
    const artifacts = groupData.reconstructionRun
      ? segmentation3dGaussianArtifacts(groupData.reconstructionRun)
      : [];
    const composed = composedGaussianArtifact(artifacts);
    if (composed) {
      group.appendChild(
        gaussianArtifactControl(groupData.reconstructionRun, composed, "Composed 3DGS"),
      );
    } else {
      group.appendChild(mapAnythingExpectedRow("Composed RGB + Region IDs", "After training"));
    }
    if (!artifacts.some((row) => String(row.artifactGroup) === "rgb_objects")) {
      const pipeline = groupData.pipelineRun;
      group.appendChild(mapAnythingExpectedRow(
        "Region 3DGS Parts",
        pipeline && pipeline.regionCount
          ? `${Number(pipeline.trainedRegionCount) || 0}/${pipeline.regionCount}`
          : "Waiting",
      ));
    }
    mapAnything3dgsResultsEl.appendChild(group);
  }
}

function scheduleMapAnythingPipelinePolling(delay = 900) {
  if (mapAnythingPipelinePollTimer) {
    clearTimeout(mapAnythingPipelinePollTimer);
  }
  const poll = async () => {
    try {
      state.segmentation3dStatus = await loadJson("/api/3d-segmentation/status");
      syncSegmentation3dResultList();
    } catch (error) {
      console.error(error);
    }
    const running = mapAnythingPipelineRuns().some((run) => run.running);
    mapAnythingPipelinePollTimer = setTimeout(
      poll,
      document.hidden ? 8000 : running ? 1200 : 5000,
    );
  };
  mapAnythingPipelinePollTimer = setTimeout(poll, delay);
}
