// Gaussian Flats hybrid 2D/3D reconstruction progress and result controls.
function gaussianFlatsPlaneMaskActive(run, artifact) {
  const active = state.gaussianFlatsPlaneMask;
  return Boolean(
    active.visible &&
    String(active.runId) === String(run.runId) &&
    String(active.variantId) === String(artifact.variantId)
  );
}

function gaussianFlatsPlaneMaskUrl(runId, variantId, frameId, version) {
  return datasetUrl(
    `/api/3d-segmentation/runs/${encodeURIComponent(runId)}` +
    `/plane-masks/${encodeURIComponent(variantId)}/${Number(frameId)}.png` +
    `?v=${encodeURIComponent(version || "ready")}`,
  );
}

async function loadActiveGaussianFlatsPlaneMask(frameId) {
  const active = state.gaussianFlatsPlaneMask;
  if (!active.visible || !active.runId || !active.variantId) {
    active.image = null;
    render();
    return;
  }
  const key = `${active.runId}:${active.variantId}:${Number(frameId)}:${active.contentVersion}`;
  let image = active.images.get(key);
  if (!image) {
    try {
      image = await loadImage(
        gaussianFlatsPlaneMaskUrl(
          active.runId,
          active.variantId,
          frameId,
          active.contentVersion,
        ),
      );
      active.images.set(key, image);
    } catch (error) {
      console.error(error);
      image = null;
    }
  }
  if (
    state.selectedFrame &&
    Number(state.selectedFrame.id) === Number(frameId) &&
    active.visible
  ) {
    active.image = image;
    render();
  }
}

async function setGaussianFlatsPlaneMaskVisible(run, artifact, visible) {
  const active = state.gaussianFlatsPlaneMask;
  if (visible) {
    active.visible = true;
    active.runId = String(run.runId);
    active.variantId = String(artifact.variantId);
    active.contentVersion = String(artifact.contentVersion || "ready");
  } else if (gaussianFlatsPlaneMaskActive(run, artifact)) {
    active.visible = false;
    active.runId = "";
    active.variantId = "";
    active.contentVersion = "";
    active.image = null;
  }
  if (state.selectedFrame) {
    await loadActiveGaussianFlatsPlaneMask(state.selectedFrame.id);
  } else {
    render();
  }
  syncReconstructionResultLists();
}

function gaussianFlatsPlaneMaskControl(run, artifact) {
  const active = gaussianFlatsPlaneMaskActive(run, artifact);
  const row = document.createElement("label");
  row.className = `gaussian-artifact-row${active ? " active" : ""}`;

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = active;
  input.disabled = Number(artifact.frameCount) <= 0;
  input.addEventListener("change", () => {
    setGaussianFlatsPlaneMaskVisible(run, artifact, input.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
      syncReconstructionResultLists();
    });
  });

  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = artifact.displayName || "Plane Masks";
  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = `${formatCount(Number(artifact.nonemptyFrameCount) || 0)}` +
    ` / ${formatCount(Number(artifact.frameCount) || 0)} views`;
  row.append(input, name, status);
  return row;
}
function gaussianFlatsMeshControl(run, artifact) {
  const active = Boolean(
    state.meshViewportVisible &&
    String(state.meshViewportRunId) === String(run.runId) &&
    String(state.meshViewportVariantId) === String(artifact.variantId)
  );
  const row = document.createElement("label");
  row.className = `gaussian-artifact-row${active ? " active" : ""}`;

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = active;
  input.addEventListener("change", () => {
    const action = input.checked
      ? activateMeshViewport(run.runId, artifact.variantId)
      : Promise.resolve(deactivateMeshViewport());
    action.catch((error) => {
      console.error(error);
      state.meshViewportError = error.message;
      setSelectionStatus(error.message);
      syncReconstructionResultLists();
    });
  });

  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = artifact.displayName || "Extracted Meshes";
  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = active && !state.meshViewportReady
    ? state.meshViewportError
      ? "Failed"
      : `${Math.round(state.meshViewportProgress || 0)}%`
    : `${formatCount(Number(artifact.triangleCount) || 0)} faces`;
  row.append(input, name, status);
  return row;
}

function syncGaussianFlatsResultList() {
  if (!gaussianFlatsResultsEl) return;
  const pipelines = gaussianFlatsPipelineRuns();
  const results = gaussianFlatsRuns();
  const groups = new Map();
  const requireGroup = (runId) => {
    const key = String(runId || "");
    const group = groups.get(key) || { runId: key, pipeline: null, result: null };
    groups.set(key, group);
    return group;
  };
  for (const pipeline of pipelines) requireGroup(pipeline.runId).pipeline = pipeline;
  for (const result of results) requireGroup(result.baseRunId || result.runId).result = result;

  gaussianFlatsResultsEl.replaceChildren();
  const runningCount = pipelines.filter((row) => row.running).length;
  const readyCount = results.length;
  gaussianFlatsResultSummaryEl.textContent = runningCount
    ? `${runningCount} running · ${readyCount} viewable`
    : groups.size
      ? `${groups.size} run${groups.size === 1 ? "" : "s"} · ${readyCount} viewable`
      : "No saved runs";

  if (!groups.size) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = "Run Gaussian Flats to generate masks and meshes";
    gaussianFlatsResultsEl.appendChild(empty);
    return;
  }

  for (const groupData of [...groups.values()].sort((a, b) => a.runId.localeCompare(b.runId))) {
    const group = document.createElement("section");
    group.className = "gaussian-run gaussian-flats-run";
    const heading = document.createElement("div");
    heading.className = "gaussian-run-heading";
    const title = document.createElement("span");
    title.className = "gaussian-run-title";
    title.textContent = groupData.pipeline?.displayName || groupData.result?.displayName || groupData.runId;
    heading.appendChild(title);
    if (groupData.pipeline) {
      const runState = document.createElement("span");
      runState.className = `mapanything-run-state ${groupData.pipeline.status || "paused"}`;
      runState.textContent = groupData.pipeline.status || "paused";
      heading.appendChild(runState);
    }
    group.appendChild(heading);
    if (groupData.pipeline) group.appendChild(mapAnythingPipelineStatusControl(groupData.pipeline));

    appendSplitSplatSection(group, "Plane Masks");
    const planeMasks = groupData.result
      ? segmentation3dPlaneMaskArtifacts(groupData.result)
      : [];
    if (planeMasks.length) {
      for (const artifact of planeMasks) {
        group.appendChild(gaussianFlatsPlaneMaskControl(groupData.result, artifact));
      }
    } else {
      group.appendChild(mapAnythingExpectedRow("Per-view plane masks", "After preparation"));
    }

    appendSplitSplatSection(group, "Meshes");
    const meshArtifacts = groupData.result
      ? segmentation3dMeshArtifacts(groupData.result)
      : [];
    if (meshArtifacts.length) {
      for (const artifact of meshArtifacts) {
        group.appendChild(gaussianFlatsMeshControl(groupData.result, artifact));
      }
    } else {
      group.appendChild(mapAnythingExpectedRow("Full Hybrid + Plane-only", "After mesh extraction"));
    }
    gaussianFlatsResultsEl.appendChild(group);
  }
}
