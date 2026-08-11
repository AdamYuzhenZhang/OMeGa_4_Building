// Joint static 3DGS baselines with hard or learned persistent identities.
function staticSemantic3dgsRuns() {
  return savedSegmentation3dRuns().filter(
    (run) => String(run.experimentFamily) === "static_semantic_3dgs",
  );
}

function staticSemanticMethodDetail(run) {
  if (String(run.methodId) === "segment_then_splat_paper") {
    return "Paper AutoSeg + multilevel IDs";
  }
  if (String(run.methodId) === "segment_then_splat") return "Hard inherited IDs";
  if (String(run.methodId) === "gaussian_grouping") return "Learned identity field";
  return "Joint static scene";
}

function syncStaticSemanticResultList() {
  if (!staticSemanticResultsEl) return;
  const runs = staticSemantic3dgsRuns().sort(
    (a, b) => String(a.displayName || a.runId).localeCompare(String(b.displayName || b.runId)),
  );
  staticSemanticResultsEl.replaceChildren();
  if (staticSemanticResultSummaryEl) {
    staticSemanticResultSummaryEl.textContent = runs.length
      ? `${runs.length} method${runs.length === 1 ? "" : "s"}`
      : "No saved runs";
  }
  if (!runs.length) {
    const empty = document.createElement("span");
    empty.className = "point-result-empty";
    empty.textContent = "Run a joint static semantic 3DGS baseline";
    staticSemanticResultsEl.appendChild(empty);
    return;
  }

  for (const run of runs) {
    const group = document.createElement("section");
    group.className = "gaussian-run static-semantic-run";

    const heading = document.createElement("div");
    heading.className = "gaussian-run-heading";
    const name = document.createElement("span");
    name.className = "gaussian-run-title";
    name.textContent = run.displayName || run.methodId;
    const detail = document.createElement("span");
    detail.className = "step-status";
    detail.textContent = staticSemanticMethodDetail(run);
    heading.append(name, detail);
    group.appendChild(heading);

    const artifacts = segmentation3dGaussianArtifacts(run);
    const scene = artifacts.find(
      (artifact) => String(artifact.artifactGroup) === "composed_scene",
    );
    if (scene) {
      group.appendChild(gaussianArtifactControl(run, scene, "Joint RGB + Regions"));
    }

    const maskMethodId = String(run.maskMethodId || "");
    const maskLayer = maskMethodId ? splitSplatLayer(maskMethodId) : null;
    if (maskLayer) {
      appendSplitSplatSection(group, "Paper Masks");
      group.appendChild(
        splitSplatMaskControl(maskLayer, "AutoSeg + 3D Merge")
      );
    }

    const summary = document.createElement("div");
    summary.className = "mapanything-progress-detail";
    const foreground = Number(run.foregroundGaussianCount) || 0;
    const total = Number(run.gaussianCount) || Number(scene && scene.pointCount) || 0;
    const coverage = total > 0 ? Math.round(100 * foreground / total) : 0;
    summary.textContent = `${formatCount(total)} splats · ${formatCount(run.regionCount)} regions · ${coverage}% assigned`;
    group.appendChild(summary);
    staticSemanticResultsEl.appendChild(group);
  }
}
