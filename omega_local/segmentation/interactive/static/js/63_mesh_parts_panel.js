// Left-side visibility and color controls for multipart object meshes.
function meshPartColor(part) {
  const rgb = Array.isArray(part.regionColor) ? part.regionColor : [];
  return rgb.length === 3
    ? `rgb(${rgb.map((value) => Number(value) || 0).join(", ")})`
    : palette((Number(part.instanceId) || 0) + 1700);
}

function meshPartLoadStatus(part) {
  const progress = Number(state.meshViewportPartProgress[String(part.partId)]) || 0;
  return progress >= 100 ? "Ready" : `${Math.round(progress)}%`;
}

function renderMeshPartPanel() {
  if (!idPanel || !regionList || !idList || !meshPartSceneActive()) return false;
  const scene = state.meshViewportScene;
  const parts = Array.isArray(scene.parts) ? scene.parts : [];
  const visibleCount = parts.filter(
    (part) => state.meshViewportPartVisibility[String(part.partId)] !== false
  ).length;

  idPanel.classList.add("gaussian-part-mode");
  regionList.classList.add("gaussian-part-list");
  regionList.replaceChildren();
  idList.replaceChildren();

  const allControl = document.createElement("label");
  allControl.className = "gaussian-all-control";
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = visibleCount === parts.length;
  allInput.indeterminate = visibleCount > 0 && visibleCount < parts.length;
  allInput.addEventListener("change", () => setAllMeshPartsVisible(allInput.checked));
  const allLabel = document.createElement("span");
  allLabel.textContent = "All";
  allControl.append(allInput, allLabel);
  regionList.appendChild(idPanelSectionHeader(
    `Mesh Objects | ${visibleCount}/${parts.length}`,
    allControl,
  ));

  const sceneName = document.createElement("div");
  sceneName.className = "gaussian-scene-name";
  sceneName.textContent = scene.displayName || "Object Meshes";
  regionList.appendChild(sceneName);

  const colorModes = Array.isArray(scene.colorModes) ? scene.colorModes : [];
  if (colorModes.length > 1) {
    const switcher = document.createElement("div");
    switcher.className = "gaussian-color-modes";
    for (const mode of colorModes) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `gaussian-color-mode${state.meshViewportColorMode === mode.modeId ? " active" : ""}`;
      button.textContent = mode.displayName || mode.modeId;
      button.addEventListener("click", () => setMeshColorMode(mode.modeId));
      switcher.appendChild(button);
    }
    regionList.appendChild(switcher);
  }

  for (const part of parts) {
    const partId = String(part.partId);
    const checked = state.meshViewportPartVisibility[partId] !== false;
    const row = document.createElement("label");
    row.className = `id-row gaussian-part-row${checked ? " active" : ""}`;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    input.addEventListener("change", () => setMeshPartVisibility(partId, input.checked));
    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = meshPartColor(part);
    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("span");
    name.className = "id-name";
    const instanceId = Number(part.instanceId) || 0;
    name.textContent = instanceId > 0
      ? `Region ${instanceId}`
      : String(part.displayName || "Scene Mesh");
    const regionName = document.createElement("span");
    regionName.className = "gaussian-region-name";
    regionName.textContent = String(part.regionName || part.displayName || "");
    const count = document.createElement("span");
    count.className = "id-count";
    count.textContent = `${formatCount(Number(part.triangleCount) || 0)} faces`;
    main.appendChild(name);
    if (regionName.textContent && regionName.textContent !== name.textContent) {
      main.appendChild(regionName);
    }
    main.appendChild(count);
    const status = document.createElement("span");
    status.className = "id-selected";
    status.textContent = meshPartLoadStatus(part);
    row.append(input, swatch, main, status);
    regionList.appendChild(row);
  }
  return true;
}
