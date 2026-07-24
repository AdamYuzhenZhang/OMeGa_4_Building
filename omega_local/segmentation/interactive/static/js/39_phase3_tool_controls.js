// Phase 3 tool panel state: active tool buttons and method-specific sliders.
function syncToolButtons() {
  for (const button of toolButtons) {
    button.classList.toggle("active", button.dataset.tool === state.tool);
  }
  for (const panel of methodControlPanels) {
    panel.hidden = panel.dataset.methodControls !== state.tool;
  }
  canvas.classList.toggle("selecting", state.tool !== "navigate");
  syncSelectionOperationControls();
  syncSamControls();
  syncGrowControls();
  if (typeof syncRgbdCueDebugControls === "function") syncRgbdCueDebugControls();
}

function setTool(tool) {
  state.tool = tool;
  state.lasso = null;
  syncToolButtons();
  render();
}

function syncGrowControls() {
  state.sam2MaskSize = clamp(Number(state.sam2MaskSize) || 0.0, -2.0, 2.0);
  state.normalGrowAngleDeg = clamp(Number(state.normalGrowAngleDeg) || 14, 2, 45);
  state.cueSmoothness = clamp(Number(state.cueSmoothness) || 0.10, 0.02, 0.50);
  state.cueSuperpixels = Math.round(clamp(Number(state.cueSuperpixels) || 800, 400, 2400) / 100) * 100;
  if (sam2SizeInput && Math.abs(Number(sam2SizeInput.value) - state.sam2MaskSize) > 1e-6) {
    sam2SizeInput.value = state.sam2MaskSize.toFixed(2);
  }
  if (normalGrowAngleInput && Number(normalGrowAngleInput.value) !== state.normalGrowAngleDeg) {
    normalGrowAngleInput.value = String(state.normalGrowAngleDeg);
  }
  if (depthGrowEdgeInput && Math.abs(Number(depthGrowEdgeInput.value) - state.cueSmoothness) > 1e-6) {
    depthGrowEdgeInput.value = state.cueSmoothness.toFixed(2);
  }
  if (rgbdCueSuperpixelsInput && Number(rgbdCueSuperpixelsInput.value) !== state.cueSuperpixels) {
    rgbdCueSuperpixelsInput.value = String(state.cueSuperpixels);
  }
  if (sam2SizeValue) sam2SizeValue.textContent = state.sam2MaskSize.toFixed(2);
  if (normalGrowAngleValue) normalGrowAngleValue.textContent = `${Math.round(state.normalGrowAngleDeg)}°`;
  if (depthGrowEdgeValue) depthGrowEdgeValue.textContent = state.cueSmoothness.toFixed(2);
  if (rgbdCueSuperpixelsValue) rgbdCueSuperpixelsValue.textContent = String(state.cueSuperpixels);
}

function setSam2MaskSize(value) {
  state.sam2MaskSize = clamp(Number(value) || 0.0, -2.0, 2.0);
  syncGrowControls();
  if (typeof scheduleSam2Refresh === "function") scheduleSam2Refresh("size");
}

function setNormalGrowAngle(value) {
  state.normalGrowAngleDeg = clamp(Number(value) || 14, 2, 45);
  syncGrowControls();
}

function setDepthGrowEdgeTolerance(value) {
  state.cueSmoothness = clamp(Number(value) || 0.10, 0.02, 0.50);
  syncGrowControls();
  if (typeof resetRgbdCueDebug === "function") resetRgbdCueDebug();
  if (typeof scheduleRgbdCueRefresh === "function") scheduleRgbdCueRefresh("smoothness");
}

function setRgbdCueSuperpixels(value) {
  state.cueSuperpixels = Math.round(clamp(Number(value) || 800, 400, 2400) / 100) * 100;
  syncGrowControls();
  if (typeof resetRgbdCueDebug === "function") resetRgbdCueDebug();
  if (typeof scheduleRgbdCueRefresh === "function") scheduleRgbdCueRefresh("superpixels");
}

function syncSamControls() {
  syncMaskEditControls();
}
