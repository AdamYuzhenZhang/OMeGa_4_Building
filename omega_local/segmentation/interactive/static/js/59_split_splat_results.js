// Split&Splat experiment grouping and result controls.
function splitSplatArtifactActive(runId, variantId) {
  return Boolean(
    state.gaussianViewportVisible &&
    String(state.gaussianViewportRunId) === String(runId) &&
    String(state.gaussianViewportVariantId) === String(variantId)
  );
}

function splitSplatExperimentName(baseRunId) {
  if (baseRunId === "split_splat_official") return "Official Baseline";
  if (baseRunId === "split_splat_sam2_video") return "SAM2 Video Adaptation";
  if (baseRunId === "split_splat_anchored_sam2_video") return "Anchored 3D Adaptation";
  return baseRunId
    .replace(/^split_splat_/, "")
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function splitSplatExperimentGroups() {
  const groups = new Map();
  let sharedRun = null;
  for (const run of splitSplatSegmentationRuns()) {
    const role = String(run.artifactRole || "");
    if (role === "shared_global") {
      sharedRun = run;
      continue;
    }
    const baseRunId = String(run.baseRunId || "");
    if (!baseRunId) continue;
    const group = groups.get(baseRunId) || {
      baseRunId,
      displayName: splitSplatExperimentName(baseRunId),
      splitRun: null,
      reconstructionRuns: [],
    };
    if (role === "reconstruction") {
      group.reconstructionRuns.push(run);
    } else if (role === "split_labels") {
      group.splitRun = run;
    }
    groups.set(baseRunId, group);
  }
  for (const group of groups.values()) {
    group.reconstructionRuns.sort(
      (a, b) => splitSplatReconstructionPriority(a) - splitSplatReconstructionPriority(b),
    );
  }
  const experiments = [...groups.values()].sort((a, b) => {
    if (a.baseRunId === "split_splat_official") return -1;
    if (b.baseRunId === "split_splat_official") return 1;
    return a.displayName.localeCompare(b.displayName);
  });
  return { sharedRun, experiments };
}

function splitSplatReconstructionPriority(run) {
  if (String(run.reconstructionVariant) === "released_splat") return 0;
  if (String(run.maskRefinement) === "none") return 1;
  if (String(run.maskRefinement) === "paper_sam2") return 2;
  return 3;
}

function splitSplatReconstructionName(run) {
  if (String(run.reconstructionVariant) === "released_splat") return "Released Splat";
  if (String(run.reconstructionVariant) === "anchored_3dgs") {
    if (String(run.maskRefinement) === "paper_sam2") {
      return "Anchored Splat · Paper SAM2";
    }
    return "Anchored Splat · No SAM2";
  }
  return run.displayName || run.runId;
}

function splitSplatLayer(baseRunId, suffix = "") {
  const key = normalizeProposalLayer(`propagation_${baseRunId}${suffix}`);
  const rows = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  return rows.find((row) => normalizeProposalLayer(row.key || "") === key) || null;
}

function syncReconstructionResultLists() {
  syncSplitSplatResultList();
  syncMapAnything3dgsResultList();
}

function appendSplitSplatSection(group, label) {
  const section = document.createElement("div");
  section.className = "gaussian-artifact-section";
  section.textContent = label;
  group.appendChild(section);
}

function gaussianArtifactControl(run, artifact, displayName = "") {
  const active = splitSplatArtifactActive(run.runId, artifact.variantId);
  const row = document.createElement("label");
  row.className = `gaussian-artifact-row${active ? " active" : ""}`;

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = active;
  input.addEventListener("change", () => {
    const action = input.checked
      ? activateGaussianViewport(run.runId, artifact.variantId)
      : Promise.resolve(deactivateGaussianViewport());
    action.catch((error) => {
      console.error(error);
      state.gaussianViewportError = error.message;
      syncReconstructionResultLists();
      setSelectionStatus(error.message);
    });
  });

  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = displayName || artifact.displayName || artifact.variantId;
  name.title = artifact.colorSpace
    ? `${artifact.colorSpace}; unlabeled splats use ${artifact.unlabeledColor || "a neutral color"}`
    : name.textContent;

  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = active && !state.gaussianViewportReady
    ? state.gaussianViewportError
      ? "Failed"
      : `${Math.round(state.gaussianViewportProgress)}%`
    : splitSplatArtifactStatus(artifact);

  row.append(input, name, status);
  return row;
}

function composedGaussianArtifact(artifacts) {
  const rows = Array.isArray(artifacts) ? artifacts : [];
  return rows.find((row) => String(row.variantId) === "appearance") ||
    rows.find((row) => String(row.variantId) === "instance_ids") ||
    rows.find((row) => String(row.artifactGroup) === "composed_scene") || null;
}

function splitSplatMaskControl(layer, displayName) {
  const key = normalizeProposalLayer(layer.key || "");
  const row = document.createElement("label");
  row.className = "gaussian-artifact-row";
  row.title = layer.description || displayName;

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(state.proposalLayers[key]);
  input.disabled = !Boolean(layer.ready);
  input.addEventListener("change", () => {
    setProposalLayerVisible(key, input.checked)
      .then(syncReconstructionResultLists)
      .catch((error) => {
        console.error(error);
        setSelectionStatus(error.message);
        syncReconstructionResultLists();
      });
  });

  const name = document.createElement("span");
  name.className = "point-source-name";
  name.textContent = displayName;

  const status = document.createElement("span");
  status.className = "step-status";
  status.textContent = layer.ready
    ? `${formatCount(Number(layer.completedFrameCount) || 0)} frames`
    : "Unavailable";
  row.append(input, name, status);
  return row;
}

function appendSplitSplatExperiment(groupData) {
  const group = document.createElement("section");
  group.className = "gaussian-run split-splat-experiment";

  const title = document.createElement("div");
  title.className = "gaussian-run-heading";
  const name = document.createElement("span");
  name.className = "gaussian-run-title";
  name.textContent = groupData.displayName;
  const reconstructionArtifacts = groupData.reconstructionRuns.map((run) => ({
    run,
    artifacts: segmentation3dGaussianArtifacts(run),
  }));
  const reconstructedRegionCount = reconstructionArtifacts.reduce(
    (total, row) => total + row.artifacts.filter(
      (artifact) => String(artifact.artifactGroup) === "rgb_objects",
    ).length,
    0,
  );
  const count = document.createElement("span");
  count.className = "step-status";
  count.textContent = reconstructedRegionCount
    ? `${formatCount(groupData.reconstructionRuns.length)} recon · ${formatCount(reconstructedRegionCount)} regions`
    : "Split only";
  title.append(name, count);
  group.appendChild(title);

  const inputLayer = splitSplatLayer(groupData.baseRunId, "_proposals");
  const supportLayer = splitSplatLayer(groupData.baseRunId, "_support");
  const outputLayer = splitSplatLayer(groupData.baseRunId);
  const splatMaskLayer = splitSplatLayer(groupData.baseRunId, "_splat_masks");
  if (inputLayer || outputLayer || splatMaskLayer) {
    appendSplitSplatSection(group, "2D Masks");
    if (inputLayer) group.appendChild(splitSplatMaskControl(inputLayer, "Input Proposals"));
    if (outputLayer) {
      const outputName = String(outputLayer.labelSpace) === "persistent_region"
        ? "Anchored Masks"
        : "Split Masks (Preview)";
      group.appendChild(
        splitSplatMaskControl(outputLayer, outputName)
      );
    }
    if (splatMaskLayer) {
      group.appendChild(
        splitSplatMaskControl(splatMaskLayer, "Splat-Refined Masks")
      );
    }
  }
  if (supportLayer) {
    appendSplitSplatSection(group, "3D Evidence");
    group.appendChild(
      splitSplatMaskControl(supportLayer, "Projected 3D Label Support")
    );
  }

  if (groupData.splitRun) {
    const artifact = segmentation3dGaussianArtifacts(groupData.splitRun).find(
      (row) => String(row.variantId) === "instance_ids",
    );
    if (artifact) {
      appendSplitSplatSection(group, "Split Stage");
      group.appendChild(gaussianArtifactControl(groupData.splitRun, artifact, "Labeled Global 3DGS"));
    }
  }

  for (const { run, artifacts } of reconstructionArtifacts) {
    const artifact = composedGaussianArtifact(artifacts);
    appendSplitSplatSection(group, splitSplatReconstructionName(run));
    if (artifact) {
      group.appendChild(gaussianArtifactControl(run, artifact, "Composed 3DGS"));
    }
  }
  splitSplatResultsEl.appendChild(group);
}

function syncSplitSplatResultList() {
  if (!splitSplatResultsEl) return;
  const { sharedRun, experiments } = splitSplatExperimentGroups();
  splitSplatResultsEl.replaceChildren();
  if (splitSplatResultSummaryEl) {
    const reconstructionCount = experiments.reduce(
      (total, group) => total + group.reconstructionRuns.length,
      0,
    );
    const objectCount = experiments.reduce(
      (total, group) => total + group.reconstructionRuns.reduce(
        (subtotal, run) => subtotal + segmentation3dGaussianArtifacts(run).filter(
          (artifact) => String(artifact.artifactGroup) === "rgb_objects",
        ).length,
        0,
      ),
      0,
    );
    splitSplatResultSummaryEl.textContent = experiments.length
      ? `${experiments.length} splits · ${reconstructionCount} recon · ${formatCount(objectCount)} regions`
      : "No saved runs";
  }
  if (!experiments.length && !sharedRun) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = "Run Split&Splat to generate masks and Gaussians";
    splitSplatResultsEl.appendChild(empty);
    return;
  }
  if (sharedRun) {
    const group = document.createElement("div");
    group.className = "gaussian-run split-splat-shared";
    const title = document.createElement("div");
    title.className = "gaussian-run-title";
    title.textContent = "Shared Reconstruction";
    group.appendChild(title);
    const appearance = segmentation3dGaussianArtifacts(sharedRun).find(
      (artifact) => String(artifact.variantId) === "appearance",
    );
    if (appearance) group.appendChild(gaussianArtifactControl(sharedRun, appearance, "Global RGB 3DGS"));
    splitSplatResultsEl.appendChild(group);
  }
  for (const experiment of experiments) appendSplitSplatExperiment(experiment);
}
