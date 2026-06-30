// Later 3D point-label edits and explicit save/undo bookkeeping.
async function refreshLabelSummary() {
  state.labelSummary = await loadJson("/api/labels/summary");
  renderIdPanel();
}

function unsavedEditCount() {
  return state.pendingEdits.length;
}

function hasUnsavedEdits() {
  return unsavedEditCount() > 0;
}

function syncSaveControls() {
  const dirty = hasUnsavedEdits();
  if (saveEditsButton) {
    saveEditsButton.disabled = state.editBusy || !dirty;
    const group = saveEditsButton.closest(".save-group");
    if (group) group.classList.toggle("dirty", dirty);
  }
  if (pendingEditStatusEl) {
    if (state.editBusy) {
      pendingEditStatusEl.textContent = "Saving";
    } else {
      pendingEditStatusEl.textContent = dirty ? `${unsavedEditCount()} unsaved` : "Saved";
    }
  }
}

async function savePendingEdits() {
  if (state.editBusy || !hasUnsavedEdits()) return;
  state.editBusy = true;
  syncSelectionControls();
  syncMaskEditControls();
  setSelectionStatus(`Saving ${unsavedEditCount()} edit${unsavedEditCount() === 1 ? "" : "s"}`);
  let errorMessage = "";
  let finalMessage = "";
  try {
    if (state.pendingEdits.length) {
      const result = await postJson("/api/labels/save", { edits: state.pendingEdits });
      await refreshLabelSummary();
      if (!result.saved) {
        finalMessage = result.message || "No point-label edits saved";
      }
      state.pendingEdits = [];
    }
    state.undoStack = [];
    finalMessage = finalMessage || "Saved edits";
  } catch (error) {
    console.error(error);
    errorMessage = error.message;
  } finally {
    state.editBusy = false;
    syncSelectionControls();
    syncMaskEditControls();
    if (errorMessage) setSelectionStatus(errorMessage);
    else if (finalMessage) setSelectionStatus(finalMessage);
  }
}

function assignSelectionToTarget() {
  const target = targetLabel();
  const indices = selectedFullIndices();
  if (target <= 0 || !indices.length) return;
  pushUndoSnapshot();
  const oldLabels = [];
  for (const index of state.selectedPointIndices) {
    oldLabels.push(state.labels[index]);
    state.labels[index] = target;
  }
  adjustLabelSummaryForAssignments(oldLabels, target);
  state.pendingEdits.push({ type: "assign_points", label: target, indices });
  syncSelectionControls();
  setSelectionStatus(`Queued assign: ${formatCount(indices.length)} points to ID ${target}`);
  render();
}

function extractSelectionToNewLabel() {
  const target = nextAvailableLabel();
  const indices = selectedFullIndices();
  if (!indices.length) return;
  pushUndoSnapshot();
  if (targetLabelInput) targetLabelInput.value = String(target);
  const oldLabels = [];
  for (const index of state.selectedPointIndices) {
    oldLabels.push(state.labels[index]);
    state.labels[index] = target;
  }
  adjustLabelSummaryForAssignments(oldLabels, target);
  state.pendingEdits.push({ type: "assign_points", label: target, indices });
  syncSelectionControls();
  setSelectionStatus(`Queued extract: ${formatCount(indices.length)} points to new ID ${target}`);
  render();
}

function mergeSelectedIdsToTarget() {
  const target = targetLabel();
  const labels = [...selectedLabels()].sort((a, b) => a - b);
  if (target <= 0 || !labels.length || (labels.length === 1 && labels[0] === target)) return;

  pushUndoSnapshot();
  const labelSet = new Set(labels);
  const mergedSelection = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    if (!labelSet.has(state.labels[i])) continue;
    state.labels[i] = target;
    mergedSelection.push(i);
  }
  adjustLabelSummaryForMerge(labels, target);
  state.pendingEdits.push({ type: "merge_labels", target, labels });
  state.selectedPointIndices = new Set(mergedSelection);
  syncSelectionControls();
  setSelectionStatus(`Queued merge: IDs ${labels.join(", ")} into ID ${target}`);
  render();
}
