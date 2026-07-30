// Left-side part inspector for multipart Gaussian scenes.
function gaussianPartDisplayName(part) {
  const instanceId = Number(part.instanceId) || 0;
  if (instanceId === 0 && String(part.partId || "").includes("000000")) return "Unassigned";
  if (instanceId > 0) return `Region ${instanceId}`;
  return String(part.displayName || part.partId || "Part");
}

function gaussianPartColor(part) {
  const rgb = Array.isArray(part.regionColor) ? part.regionColor : [];
  if (rgb.length === 3) {
    return "rgb(" + rgb.map((value) => Math.max(0, Math.min(255, Number(value) || 0))).join(", ") + ")";
  }
  return palette((Number(part.instanceId) || 0) + 1400);
}

function gaussianPartLoadStatus(part) {
  const progress = Number(
    state.gaussianViewportPartProgress[String(part.partId || part.variantId)]
  ) || 0;
  return progress >= 100 ? "Ready" : `${Math.round(progress)}%`;
}

function renderGaussianPartPanel() {
  if (!idPanel || !regionList || !idList || !gaussianPartSceneActive()) return false;
  const scene = state.gaussianViewportScene;
  const parts = Array.isArray(scene.parts) ? scene.parts : [];
  const visibleCount = parts.filter((part) => (
    state.gaussianViewportPartVisibility[String(part.partId || part.variantId)] !== false
  )).length;

  idPanel.classList.add("gaussian-part-mode");
  regionList.classList.add("gaussian-part-list");
  regionList.replaceChildren();
  idList.replaceChildren();

  const allControl = document.createElement("label");
  allControl.className = "gaussian-all-control";
  allControl.title = "Show or hide every loaded region";
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = visibleCount === parts.length;
  allInput.indeterminate = visibleCount > 0 && visibleCount < parts.length;
  allInput.addEventListener("change", () => setAllGaussianPartsVisible(allInput.checked));
  const allLabel = document.createElement("span");
  allLabel.textContent = "All";
  allControl.append(allInput, allLabel);
  regionList.appendChild(idPanelSectionHeader(
    `Splat Parts | ${visibleCount}/${parts.length}`,
    allControl,
  ));

  const sceneName = document.createElement("div");
  sceneName.className = "gaussian-scene-name";
  sceneName.textContent = scene.displayName || "Gaussian Scene";
  sceneName.title = sceneName.textContent;
  regionList.appendChild(sceneName);
  const colorModes = Array.isArray(scene.colorModes) ? scene.colorModes : [];
  if (colorModes.length > 1) {
    const switcher = document.createElement("div");
    switcher.className = "gaussian-color-modes";
    for (const mode of colorModes) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `gaussian-color-mode${state.gaussianViewportColorMode === mode.modeId ? " active" : ""}`;
      button.textContent = mode.displayName || mode.modeId;
      button.title = `Render splats with ${button.textContent.toLowerCase()} colors`;
      button.disabled = state.gaussianViewportColorSwitching;
      button.addEventListener("click", () => setGaussianColorMode(mode.modeId));
      switcher.appendChild(button);
    }
    regionList.appendChild(switcher);
    if (state.gaussianViewportColorSwitching) {
      const progress = document.createElement("span");
      progress.className = "gaussian-color-progress";
      progress.textContent = `Updating ${Math.round(state.gaussianViewportColorProgress || 0)}%`;
      switcher.appendChild(progress);
    }
  }


  for (const part of parts) {
    const partId = String(part.partId || part.variantId);
    const checked = state.gaussianViewportPartVisibility[partId] !== false;
    const row = document.createElement("label");
    row.className = `id-row gaussian-part-row${checked ? " active" : ""}`;
    row.title = [
      gaussianPartDisplayName(part),
      String(part.regionName || ""),
      String(part.displayName || ""),
    ].filter(Boolean).join(" | ");

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    input.addEventListener("change", () => {
      setGaussianPartVisibility(partId, input.checked);
    });

    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = gaussianPartColor(part);

    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("span");
    name.className = "id-name";
    name.textContent = gaussianPartDisplayName(part);
    const regionName = document.createElement("span");
    regionName.className = "gaussian-region-name";
    regionName.textContent = String(part.regionName || "");
    const count = document.createElement("span");
    count.className = "id-count";
    count.textContent = `${formatCount(Number(part.pointCount) || 0)} splats`;
    main.appendChild(name);
    if (regionName.textContent) main.appendChild(regionName);
    main.appendChild(count);

    const status = document.createElement("span");
    status.className = "id-selected";
    status.textContent = gaussianPartLoadStatus(part);
    row.append(input, swatch, main, status);
    regionList.appendChild(row);
  }
  return true;
}
