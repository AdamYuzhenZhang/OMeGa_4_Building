// Temporary local SAM2-video propagation comparison around the active frame.
function propagationNeighborCount() {
  const value = propagationNeighborsInput ? Number(propagationNeighborsInput.value) : 2;
  return Math.max(1, Math.min(Number.isFinite(value) ? Math.round(value) : 2, 8));
}

function syncPropagationControls() {
  const hasFrame = Boolean(state.selectedFrame && state.exactFrameView);
  const ready = proposalReady();
  const hasSource = hasPixelSelection();
  if (runPropagationButton) {
    runPropagationButton.disabled = state.propagationBusy || !ready || !hasFrame || !hasSource;
  }
  if (propagationStatusEl) {
    if (state.propagationBusy) {
      propagationStatusEl.textContent = "Running SAM2 video";
    } else if (!ready) {
      propagationStatusEl.textContent = "Load proposals first";
    } else if (!hasFrame) {
      propagationStatusEl.textContent = "Select a frame";
    } else if (!hasSource) {
      propagationStatusEl.textContent = "Select edited pixels first";
    } else {
      propagationStatusEl.textContent = `Ready | +/-${propagationNeighborCount()} frames`;
    }
  }
}

async function runPropagationTest() {
  if (!state.selectedFrame || !hasPixelSelection() || state.propagationBusy) return;
  const frameId = state.selectedFrame.id;
  const neighborCount = propagationNeighborCount();
  state.propagationBusy = true;
  state.propagationResult = null;
  syncPropagationControls();
  setSelectionStatus(`Running SAM2 video propagation around frame ${frameId}`);
  try {
    const result = await postJson("/api/proposals/sam2/propagate-selection", {
      frameId,
      neighborCount,
      selectionOps: cloneSelectionOps(state.selectionOps),
    });
    state.propagationResult = result;
    renderPropagationModal(result);
    const count = Array.isArray(result.frames) ? result.frames.length : 0;
    setSelectionStatus(`Propagation result ready | ${count.toLocaleString()} frame${count === 1 ? "" : "s"}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
    if (propagationStatusEl) propagationStatusEl.textContent = "Propagation failed";
  } finally {
    state.propagationBusy = false;
    syncPropagationControls();
  }
}

function frameById(frameId) {
  return state.frames.find((frame) => Number(frame.id) === Number(frameId)) || null;
}

function renderPropagationModal(result) {
  if (!propagationModal || !propagationResultsEl) return;
  propagationResultsEl.textContent = "";
  const rows = Array.isArray(result && result.frames) ? result.frames : [];
  if (propagationModalSummary) {
    const sourceArea = Number(result.sourceAreaPixels) || 0;
    propagationModalSummary.textContent = `Frame ${result.frameId} | ${rows.length} views | source ${formatCount(sourceArea)} px`;
  }
  for (const row of rows) {
    const frame = frameById(row.frameId);
    if (!frame) continue;
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.classList.toggle("source", Boolean(row.isSource));

    const title = document.createElement("div");
    title.className = "comparison-card-title";
    const offset = Number(row.offset) || 0;
    const offsetText = offset === 0 ? "source" : offset > 0 ? `+${offset}` : `${offset}`;
    const area = Number(row.areaPixels) || 0;
    title.textContent = `Frame ${row.frameId} | ${offsetText} | ${formatCount(area)} px`;

    const propagated = comparisonImageStack(frame, `data:image/png;base64,${row.maskOverlayPng}`, "Propagated");
    const proposals = comparisonImageStack(frame, proposalOverlayUrl(row.frameId), "Current Proposals");

    card.append(title, propagated, proposals);
    propagationResultsEl.appendChild(card);
  }
  propagationModal.hidden = false;
}

function comparisonImageStack(frame, overlayUrl, label) {
  const wrap = document.createElement("div");
  wrap.className = "comparison-stack rotate-cw";
  const width = Math.max(1, Number(frame.width) || 1);
  const height = Math.max(1, Number(frame.height) || 1);
  wrap.style.aspectRatio = `${height} / ${width}`;
  wrap.style.setProperty("--source-aspect", `${width / height}`);

  const image = document.createElement("img");
  image.className = "comparison-rgb";
  image.loading = "lazy";
  image.src = frame.imageUrl;
  image.alt = "";

  const overlay = document.createElement("img");
  overlay.className = "comparison-overlay";
  overlay.loading = "lazy";
  overlay.src = overlayUrl;
  overlay.alt = "";

  const caption = document.createElement("span");
  caption.textContent = label;
  wrap.append(image, overlay, caption);
  return wrap;
}

function closePropagationModal() {
  if (propagationModal) propagationModal.hidden = true;
}
