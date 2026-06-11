function formatCount(value) {
  return Number(value).toLocaleString();
}

function computeObjectStats(state) {
  const stats = new Map();
  for (const label of state.labels) {
    const entry = stats.get(label) || { label, count: 0, selected: 0 };
    entry.count += 1;
    stats.set(label, entry);
  }
  for (const index of state.selectedIndices) {
    const label = state.labels[index];
    const entry = stats.get(label);
    if (entry) entry.selected += 1;
  }
  return [...stats.values()].sort((a, b) => {
    if (a.label <= 0 && b.label > 0) return 1;
    if (a.label > 0 && b.label <= 0) return -1;
    return a.label - b.label;
  });
}

export function createObjectPanel({
  state,
  elements,
  palette,
  labelVisible,
  labelsFromSelection,
  selectionModeFromEvent,
  selectLabel,
  render,
}) {
  const {
    objectList,
    objectSummary,
    objectFilterInput,
    showAllObjectsButton,
    hideSelectedObjectsButton,
    isolateSelectedObjectsButton,
    targetLabelInput,
  } = elements;

  function renderObjectList() {
    if (!objectList || !objectSummary) return;
    const stats = computeObjectStats(state);
    const positiveStats = stats.filter((item) => item.label > 0);
    const selectedLabels = labelsFromSelection();
    const filter = state.objectFilter.trim().toLowerCase();
    const visibleRows = stats.filter((item) => {
      if (!filter) return item.label > 0;
      return String(item.label).includes(filter);
    });

    objectSummary.textContent = `${formatCount(positiveStats.length)} objects | ${formatCount(state.labels.length)} points`;
    objectList.textContent = "";

    for (const item of visibleRows) {
      const row = document.createElement("div");
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.className = "object-row";
      row.classList.toggle("active", selectedLabels.has(item.label));
      row.classList.toggle("hidden", !labelVisible(item.label));
      row.classList.toggle("isolated", state.isolatedLabels.has(item.label));
      row.dataset.label = String(item.label);

      const swatch = document.createElement("span");
      swatch.className = "object-swatch";
      swatch.style.background = palette(item.label);

      const main = document.createElement("span");
      main.className = "object-main";
      const id = document.createElement("span");
      id.className = "object-id";
      id.textContent = item.label > 0 ? `Object ${item.label}` : "Unlabeled";
      const count = document.createElement("span");
      count.className = "object-count";
      count.textContent = `${formatCount(item.count)} pts`;
      main.append(id, count);

      const badge = document.createElement("span");
      badge.className = "object-badge";
      badge.textContent = item.selected ? formatCount(item.selected) : "";
      badge.title = item.selected ? "Selected points" : "No selected points";

      const eye = document.createElement("button");
      eye.type = "button";
      eye.className = "object-icon";
      eye.textContent = labelVisible(item.label) ? "Hide" : "Show";
      eye.title = labelVisible(item.label) ? "Hide object" : "Show object";
      eye.addEventListener("click", (event) => {
        event.stopPropagation();
        if (labelVisible(item.label)) {
          state.hiddenLabels.add(item.label);
          state.isolatedLabels.delete(item.label);
        } else {
          state.hiddenLabels.delete(item.label);
          if (state.isolatedLabels.size) {
            state.isolatedLabels.add(item.label);
          }
        }
        renderObjectList();
        render();
      });

      row.append(swatch, main, badge, eye);
      row.addEventListener("click", (event) => {
        if (targetLabelInput) targetLabelInput.value = String(item.label);
        selectLabel(item.label, selectionModeFromEvent(event));
      });
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        if (targetLabelInput) targetLabelInput.value = String(item.label);
        selectLabel(item.label, selectionModeFromEvent(event));
      });
      row.addEventListener("dblclick", (event) => {
        event.preventDefault();
        state.isolatedLabels = new Set([item.label]);
        state.hiddenLabels.delete(item.label);
        renderObjectList();
        render();
      });
      objectList.appendChild(row);
    }
  }

  function showAllObjects() {
    state.hiddenLabels.clear();
    state.isolatedLabels.clear();
    renderObjectList();
    render();
  }

  function hideSelectedObjects() {
    for (const label of labelsFromSelection()) {
      state.hiddenLabels.add(label);
      state.isolatedLabels.delete(label);
    }
    renderObjectList();
    render();
  }

  function isolateSelectedObjects() {
    const labels = labelsFromSelection();
    if (!labels.size) return;
    state.isolatedLabels = labels;
    for (const label of labels) {
      state.hiddenLabels.delete(label);
    }
    renderObjectList();
    render();
  }

  objectFilterInput?.addEventListener("input", () => {
    state.objectFilter = objectFilterInput.value;
    renderObjectList();
  });
  showAllObjectsButton?.addEventListener("click", showAllObjects);
  hideSelectedObjectsButton?.addEventListener("click", hideSelectedObjects);
  isolateSelectedObjectsButton?.addEventListener("click", isolateSelectedObjects);

  return {
    render: renderObjectList,
    showAll: showAllObjects,
    hideSelected: hideSelectedObjects,
    isolateSelected: isolateSelectedObjects,
  };
}
