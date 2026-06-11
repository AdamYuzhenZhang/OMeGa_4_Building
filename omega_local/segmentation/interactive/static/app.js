const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d", { alpha: false });
const statusEl = document.getElementById("projectStatus");
const hud = document.getElementById("viewerHud");
const filmstrip = document.getElementById("filmstrip");
const resetButton = document.getElementById("resetView");
const pointSizeInput = document.getElementById("pointSize");
const showFrameImageInput = document.getElementById("showFrameImage");
const rotateFramesInput = document.getElementById("rotateFrames");
const toolButtons = [...document.querySelectorAll("[data-tool]")];
const runSam2Button = document.getElementById("runSam2");
const clearSam2Button = document.getElementById("clearSam2");
const clearSelectionButton = document.getElementById("clearSelection");
const selectionStatusEl = document.getElementById("selectionStatus");
const idPanelSummary = document.getElementById("idPanelSummary");
const idList = document.getElementById("idList");
const targetLabelInput = document.getElementById("targetLabel");
const assignSelectionButton = document.getElementById("assignSelection");
const extractSelectionButton = document.getElementById("extractSelection");
const mergeSelectedIdsButton = document.getElementById("mergeSelectedIds");
const saveEditsButton = document.getElementById("saveEdits");
const pendingEditStatusEl = document.getElementById("pendingEditStatus");

const state = {
  project: null,
  frames: [],
  positions: new Float32Array(),
  labels: new Int32Array(),
  sourceIndices: [],
  labelSummary: null,
  bounds: null,
  camera: null,
  target: [0, 0, 0],
  selectedFrame: null,
  selectedImage: null,
  exactFrameView: false,
  frameView: null,
  transition: null,
  pointSize: 2,
  showFrameImage: true,
  rotateFrames: true,
  dragging: false,
  dragButton: null,
  lastPointer: [0, 0],
  labelColors: new Map(),
  tool: "navigate",
  selectedPointIndices: new Set(),
  lasso: null,
  samPrompts: [],
  samMaskResult: null,
  samMaskImage: null,
  samBusy: false,
  editBusy: false,
  pendingEdits: [],
  undoStack: [],
};

function setStatus(text) {
  statusEl.textContent = text;
}

function setSelectionStatus(text) {
  if (selectionStatusEl) selectionStatusEl.textContent = text;
}

async function loadJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const error = await response.json();
      if (error && error.detail) detail = error.detail;
    } catch (_error) {
      // Keep the HTTP status if the server does not return JSON.
    }
    throw new Error(`${url}: ${detail}`);
  }
  return response.json();
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function formatCount(value) {
  return Number(value).toLocaleString();
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function mul(v, scale) {
  return [v[0] * scale, v[1] * scale, v[2] * scale];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const len = length(v);
  if (!Number.isFinite(len) || len < 1e-9) return [0, 0, 0];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function rotateAroundAxis(vector, axis, angle) {
  const a = normalize(axis);
  if (length(a) < 1e-9) return vector.slice();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const av = dot(a, vector);
  const axv = cross(a, vector);
  return [
    vector[0] * c + axv[0] * s + a[0] * av * (1 - c),
    vector[1] * c + axv[1] * s + a[1] * av * (1 - c),
    vector[2] * c + axv[2] * s + a[2] * av * (1 - c),
  ];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpVec(a, b, t) {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

function cloneCamera(camera) {
  return {
    eye: camera.eye.slice(),
    right: camera.right.slice(),
    down: camera.down.slice(),
    forward: camera.forward.slice(),
    fx: camera.fx,
    fy: camera.fy,
    cx: camera.cx,
    cy: camera.cy,
    width: camera.width,
    height: camera.height,
  };
}

function orthonormalCamera(camera) {
  const forward = normalize(camera.forward);
  let right = sub(camera.right, mul(forward, dot(camera.right, forward)));
  right = normalize(right);
  if (length(right) < 1e-9) right = [1, 0, 0];
  let down = normalize(cross(forward, right));
  if (length(down) < 1e-9) down = [0, 1, 0];
  right = normalize(cross(down, forward));
  return {
    ...camera,
    right,
    down,
    forward,
  };
}

function interpolateCamera(start, end, t) {
  const eased = smoothstep(clamp(t, 0, 1));
  return orthonormalCamera({
    eye: lerpVec(start.eye, end.eye, eased),
    right: lerpVec(start.right, end.right, eased),
    down: lerpVec(start.down, end.down, eased),
    forward: lerpVec(start.forward, end.forward, eased),
    fx: lerp(start.fx, end.fx, eased),
    fy: lerp(start.fy, end.fy, eased),
    cx: lerp(start.cx, end.cx, eased),
    cy: lerp(start.cy, end.cy, eased),
    width: lerp(start.width, end.width, eased),
    height: lerp(start.height, end.height, eased),
  });
}

function palette(label) {
  if (state.labelColors.has(label)) return state.labelColors.get(label);
  if (label <= 0) {
    state.labelColors.set(label, "#6b7178");
    return "#6b7178";
  }
  const hue = (label * 137.508) % 360;
  const color = `hsl(${hue.toFixed(1)} 72% 58%)`;
  state.labelColors.set(label, color);
  return color;
}

function idStats() {
  const stats = new Map();
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      const label = Number(item.label);
      if (label <= 0) continue;
      stats.set(label, {
        label,
        count: Number(item.count) || 0,
        served: 0,
        selected: 0,
      });
    }
  }

  for (let i = 0; i < state.labels.length; i += 1) {
    const label = state.labels[i];
    if (label <= 0) continue;
    const entry = stats.get(label) || { label, count: 0, served: 0, selected: 0 };
    if (!state.labelSummary) entry.count += 1;
    entry.served += 1;
    stats.set(label, entry);
  }
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    const entry = stats.get(label);
    if (entry) entry.selected += 1;
  }
  return [...stats.values()].sort((a, b) => a.label - b.label);
}

function indicesForLabel(label) {
  const indices = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    if (state.labels[i] === label) indices.push(i);
  }
  return indices;
}

function cloneLabelSummary(summary) {
  if (!summary) return null;
  return {
    ...summary,
    labels: Array.isArray(summary.labels)
      ? summary.labels.map((item) => ({ label: Number(item.label), count: Number(item.count) }))
      : [],
  };
}

function labelSummaryMap() {
  const map = new Map();
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      const label = Number(item.label);
      if (label > 0) map.set(label, Number(item.count) || 0);
    }
  }
  return map;
}

function setLabelSummaryFromMap(map) {
  const labels = [...map.entries()]
    .filter(([label, count]) => label > 0 && count > 0)
    .sort((a, b) => a[0] - b[0])
    .map(([label, count]) => ({ label, count }));
  const pointCount = state.labelSummary ? state.labelSummary.pointCount : state.labels.length;
  state.labelSummary = {
    pointCount,
    labelCount: labels.length,
    labels,
  };
}

function adjustLabelSummaryForAssignments(oldLabels, target) {
  const map = labelSummaryMap();
  for (const label of oldLabels) {
    if (label > 0) map.set(label, Math.max(0, (map.get(label) || 0) - 1));
  }
  map.set(target, (map.get(target) || 0) + oldLabels.length);
  setLabelSummaryFromMap(map);
}

function adjustLabelSummaryForMerge(labels, target) {
  const map = labelSummaryMap();
  let moved = 0;
  for (const label of labels) {
    if (label === target) continue;
    moved += map.get(label) || 0;
    map.delete(label);
  }
  map.set(target, (map.get(target) || 0) + moved);
  setLabelSummaryFromMap(map);
}

function pushUndoSnapshot() {
  state.undoStack.push({
    labels: new Int32Array(state.labels),
    labelSummary: cloneLabelSummary(state.labelSummary),
    selectedPointIndices: [...state.selectedPointIndices],
    pendingEditCount: state.pendingEdits.length,
    targetValue: targetLabelInput ? targetLabelInput.value : "",
  });
  if (state.undoStack.length > 60) {
    state.undoStack.shift();
  }
}

function undoLastEdit() {
  if (state.editBusy || !state.undoStack.length) return;
  const snapshot = state.undoStack.pop();
  state.labels = snapshot.labels;
  state.labelSummary = snapshot.labelSummary;
  state.selectedPointIndices = new Set(snapshot.selectedPointIndices);
  state.pendingEdits = state.pendingEdits.slice(0, snapshot.pendingEditCount);
  if (targetLabelInput) targetLabelInput.value = snapshot.targetValue;
  syncSelectionControls();
  syncSaveControls();
  setSelectionStatus(state.pendingEdits.length ? `Undid edit | ${state.pendingEdits.length} unsaved` : "Undid edit | Saved");
  render();
}

function renderIdPanel() {
  if (!idList || !idPanelSummary) return;
  const stats = idStats();
  const totalPoints = state.labelSummary ? state.labelSummary.pointCount : state.labels.length;
  idPanelSummary.textContent = `${formatCount(stats.length)} ids | ${formatCount(totalPoints)} pts`;
  idList.textContent = "";

  if (!stats.length) {
    const empty = document.createElement("div");
    empty.className = "id-empty";
    empty.textContent = "No positive segment IDs";
    idList.appendChild(empty);
    return;
  }

  for (const item of stats) {
    const row = document.createElement("div");
    row.className = "id-row";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.dataset.label = String(item.label);
    row.classList.toggle("active", item.selected > 0);
    row.classList.toggle("full", item.selected > 0 && item.selected === item.served);
    row.title = "Click to select this ID. Shift adds; Alt subtracts.";

    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = palette(item.label);

    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("div");
    name.className = "id-name";
    name.textContent = `ID ${item.label}`;
    const count = document.createElement("div");
    count.className = "id-count";
    count.textContent = `${formatCount(item.count)} pts`;
    if (item.served !== item.count) {
      count.title = `${formatCount(item.served)} loaded in the current point sample`;
    }
    main.append(name, count);

    const selected = document.createElement("span");
    selected.className = "id-selected";
    selected.textContent = item.selected ? formatCount(item.selected) : "-";

    row.append(swatch, main, selected);
    row.addEventListener("click", (event) => {
      selectLabel(item.label, selectionOperationFromEvent(event));
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      selectLabel(item.label, selectionOperationFromEvent(event));
    });
    idList.appendChild(row);
  }
}

function syncCanvasSize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    return true;
  }
  return false;
}

function frameLayout(width, height, camera) {
  const scale = Math.min(width / camera.width, height / camera.height);
  const drawW = camera.width * scale;
  const drawH = camera.height * scale;
  return {
    scale,
    x0: (width - drawW) * 0.5,
    y0: (height - drawH) * 0.5,
    width: drawW,
    height: drawH,
  };
}

function canvasPointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return {
    x: (event.clientX - rect.left) * dpr,
    y: (event.clientY - rect.top) * dpr,
  };
}

function displayToSource(frame, u, v) {
  if (state.rotateFrames) {
    return {
      x: v,
      y: frame.height - u,
    };
  }
  return { x: u, y: v };
}

function sourceToDisplay(frame, x, y) {
  if (state.rotateFrames) {
    return {
      u: frame.height - y,
      v: x,
    };
  }
  return { u: x, v: y };
}

function displayToCanvasPoint(u, v, camera, layout, view) {
  let displayU = u;
  let displayV = v;
  if (view) {
    displayU = (u - view.centerU) * view.zoom + camera.width * 0.5;
    displayV = (v - view.centerV) * view.zoom + camera.height * 0.5;
  }
  return {
    x: layout.x0 + displayU * layout.scale,
    y: layout.y0 + displayV * layout.scale,
  };
}

function canvasToDisplayPoint(canvasX, canvasY, camera, layout, view) {
  let u = (canvasX - layout.x0) / Math.max(layout.scale, 1e-6);
  let v = (canvasY - layout.y0) / Math.max(layout.scale, 1e-6);
  if (view) {
    u = (u - camera.width * 0.5) / Math.max(view.zoom, 1e-6) + view.centerU;
    v = (v - camera.height * 0.5) / Math.max(view.zoom, 1e-6) + view.centerV;
  }
  return { u, v };
}

function defaultFrameView(camera, frameId) {
  return {
    frameId,
    zoom: 1,
    centerU: camera.width * 0.5,
    centerV: camera.height * 0.5,
  };
}

function clampFrameView(view, camera) {
  view.zoom = clamp(view.zoom, 1, 32);
  const halfU = camera.width / (2 * view.zoom);
  const halfV = camera.height / (2 * view.zoom);
  view.centerU = clamp(view.centerU, halfU, camera.width - halfU);
  view.centerV = clamp(view.centerV, halfV, camera.height - halfV);
  return view;
}

function ensureFrameView(camera, frameId) {
  if (!state.frameView || state.frameView.frameId !== frameId) {
    state.frameView = defaultFrameView(camera, frameId);
  }
  return clampFrameView(state.frameView, camera);
}

function applyFrameView(camera, view) {
  const frameView = clampFrameView({ ...view }, camera);
  return {
    ...camera,
    fx: camera.fx * frameView.zoom,
    fy: camera.fy * frameView.zoom,
    cx: (camera.cx - frameView.centerU) * frameView.zoom + camera.width * 0.5,
    cy: (camera.cy - frameView.centerV) * frameView.zoom + camera.height * 0.5,
  };
}

function cameraLookAt(eye, target, width, height) {
  const forward = normalize(sub(target, eye));
  let right = normalize(cross(forward, [0, 1, 0]));
  if (length(right) < 1e-9) right = [1, 0, 0];
  const down = normalize(cross(forward, right));
  const focal = Math.max(1, Math.min(width, height) * 0.9);
  return {
    eye,
    right,
    down,
    forward,
    fx: focal,
    fy: focal,
    cx: width * 0.5,
    cy: height * 0.5,
    width,
    height,
  };
}

function resetView() {
  if (!state.bounds) return;
  cancelTransition();
  syncCanvasSize();
  const center = state.bounds.center.slice();
  const radius = Math.max(Number(state.bounds.radius) || 1, 0.5);
  state.target = center;
  const eye = [
    center[0] + radius * 1.65,
    center[1] + radius * 0.75,
    center[2] + radius * 2.15,
  ];
  state.camera = cameraLookAt(eye, center, canvas.width, canvas.height);
  state.selectedFrame = null;
  state.selectedImage = null;
  state.exactFrameView = false;
  state.frameView = null;
  markActiveFrame();
  render();
}

function frameCamera(frame) {
  const pose = frame.poseWorldFromCamera;
  const cameraRight = normalize([pose[0], pose[4], pose[8]]);
  const cameraDown = normalize([pose[1], pose[5], pose[9]]);
  const cameraForward = normalize([pose[2], pose[6], pose[10]]);
  const eye = [pose[3], pose[7], pose[11]];

  if (state.rotateFrames) {
    return {
      eye,
      right: [-cameraDown[0], -cameraDown[1], -cameraDown[2]],
      down: cameraRight,
      forward: cameraForward,
      fx: frame.fy,
      fy: frame.fx,
      cx: frame.height - frame.cy,
      cy: frame.cx,
      width: frame.height,
      height: frame.width,
    };
  }

  return {
    eye,
    right: cameraRight,
    down: cameraDown,
    forward: cameraForward,
    fx: frame.fx,
    fy: frame.fy,
    cx: frame.cx,
    cy: frame.cy,
    width: frame.width,
    height: frame.height,
  };
}

function frameTarget(camera) {
  const center = state.bounds.center;
  const radius = Math.max(Number(state.bounds.radius) || 1, 0.5);
  const depthToCenter = dot(sub(center, camera.eye), camera.forward);
  const depth = Math.max(depthToCenter, radius * 0.7);
  return add(camera.eye, mul(camera.forward, depth));
}

function focusFrame(frame) {
  if (!state.camera) return;
  cancelTransition();
  if (!state.selectedFrame || state.selectedFrame.id !== frame.id) {
    state.samPrompts = [];
    state.samMaskResult = null;
    state.samMaskImage = null;
    syncSamControls();
  }
  const startCamera = state.exactFrameView && state.selectedFrame && state.frameView
    ? renderCamera()
    : cloneCamera(state.camera);
  const endCamera = frameCamera(frame);
  const endTarget = frameTarget(endCamera);
  state.selectedFrame = frame;
  state.exactFrameView = false;
  state.frameView = defaultFrameView(endCamera, frame.id);

  const image = new Image();
  image.onload = () => {
    if (state.selectedFrame && state.selectedFrame.id === frame.id) {
      state.selectedImage = image;
      render();
    }
  };
  image.src = frame.imageUrl;
  state.selectedImage = image;
  state.transition = {
    frame,
    startCamera,
    endCamera,
    startTarget: state.target.slice(),
    endTarget,
    startTime: performance.now(),
    duration: 700,
    rafId: null,
  };
  markActiveFrame();
  tickTransition(performance.now());
}

function cancelTransition() {
  if (!state.transition) return;
  if (state.transition.rafId) {
    cancelAnimationFrame(state.transition.rafId);
  }
  state.transition = null;
}

function tickTransition(time) {
  const transition = state.transition;
  if (!transition) return;
  const t = (time - transition.startTime) / transition.duration;
  if (t >= 1) {
    state.camera = cloneCamera(transition.endCamera);
    state.target = transition.endTarget.slice();
    state.selectedFrame = transition.frame;
    state.frameView = defaultFrameView(state.camera, transition.frame.id);
    state.exactFrameView = true;
    state.transition = null;
    markActiveFrame();
    render();
    return;
  }
  state.camera = interpolateCamera(transition.startCamera, transition.endCamera, t);
  state.target = lerpVec(transition.startTarget, transition.endTarget, smoothstep(clamp(t, 0, 1)));
  state.exactFrameView = false;
  render();
  if (state.transition) {
    state.transition.rafId = requestAnimationFrame(tickTransition);
  }
}

function project(point, camera) {
  const rel = sub(point, camera.eye);
  const z = dot(rel, camera.forward);
  if (z <= 0.01) return null;
  return {
    u: camera.fx * (dot(rel, camera.right) / z) + camera.cx,
    v: camera.fy * (dot(rel, camera.down) / z) + camera.cy,
    z,
  };
}

function renderCamera() {
  if (state.exactFrameView && state.selectedFrame && state.frameView) {
    return applyFrameView(state.camera, ensureFrameView(state.camera, state.selectedFrame.id));
  }
  return state.camera;
}

function currentBaseFrameView() {
  if (!state.exactFrameView || !state.selectedFrame || !state.frameView) return null;
  return ensureFrameView(state.camera, state.selectedFrame.id);
}

function selectionOperationFromEvent(event) {
  if (event.altKey) return "subtract";
  if (event.shiftKey) return "add";
  return "replace";
}

function selectedLabels() {
  const labels = new Set();
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    if (label > 0) labels.add(label);
  }
  return labels;
}

function selectedFullIndices() {
  const indices = [];
  for (const index of state.selectedPointIndices) {
    const sourceIndex = state.sourceIndices.length ? state.sourceIndices[index] : index;
    if (Number.isInteger(sourceIndex) && sourceIndex >= 0) indices.push(sourceIndex);
  }
  return indices;
}

function majoritySelectedLabel() {
  const counts = new Map();
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    if (label <= 0) continue;
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  let bestLabel = 0;
  let bestCount = -1;
  for (const [label, count] of counts.entries()) {
    if (count > bestCount || (count === bestCount && label < bestLabel)) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

function nextAvailableLabel() {
  let maxLabel = 0;
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      maxLabel = Math.max(maxLabel, Number(item.label) || 0);
    }
  }
  for (const label of state.labels) {
    maxLabel = Math.max(maxLabel, label);
  }
  return maxLabel + 1;
}

function targetLabel() {
  const value = Number(targetLabelInput ? targetLabelInput.value : 0);
  return Number.isInteger(value) && value > 0 ? value : 0;
}

function suggestTargetLabel() {
  if (!targetLabelInput || document.activeElement === targetLabelInput) return;
  if (targetLabel() > 0) return;
  const label = majoritySelectedLabel();
  if (label > 0) {
    targetLabelInput.value = String(label);
  }
}

function applySelection(indices, operation = "replace") {
  const next = operation === "replace" ? new Set() : new Set(state.selectedPointIndices);
  if (operation === "subtract") {
    for (const index of indices) next.delete(index);
  } else {
    for (const index of indices) next.add(index);
  }
  state.selectedPointIndices = next;
  syncSelectionControls();
}

function selectLabel(label, operation = "replace") {
  const indices = indicesForLabel(label);
  applySelection(indices, operation);
  setSelectionStatus(`ID ${label}: ${formatCount(indices.length)} pts | ${formatCount(state.selectedPointIndices.size)} selected`);
  render();
}

function clearSelection() {
  state.selectedPointIndices.clear();
  syncSelectionControls();
  render();
}

function syncSelectionControls() {
  if (clearSelectionButton) {
    clearSelectionButton.disabled = state.selectedPointIndices.size === 0;
  }
  suggestTargetLabel();
  const selectedLabelCount = selectedLabels().size;
  const hasSelection = state.selectedPointIndices.size > 0;
  if (assignSelectionButton) {
    assignSelectionButton.disabled = state.editBusy || !hasSelection || targetLabel() <= 0;
  }
  if (extractSelectionButton) {
    extractSelectionButton.disabled = state.editBusy || !hasSelection;
  }
  if (mergeSelectedIdsButton) {
    const labels = selectedLabels();
    const target = targetLabel();
    const mergesSomething = selectedLabelCount > 1 || (selectedLabelCount === 1 && !labels.has(target));
    mergeSelectedIdsButton.disabled = state.editBusy || !mergesSomething || target <= 0;
  }
  const count = state.selectedPointIndices.size;
  setSelectionStatus(count ? `${count.toLocaleString()} selected` : "No selection");
  renderIdPanel();
  syncSaveControls();
}

function syncToolButtons() {
  for (const button of toolButtons) {
    button.classList.toggle("active", button.dataset.tool === state.tool);
  }
  canvas.classList.toggle("selecting", state.tool !== "navigate");
  syncSamControls();
}

function setTool(tool) {
  state.tool = tool;
  state.lasso = null;
  syncToolButtons();
  render();
}

function syncSamControls() {
  if (runSam2Button) {
    runSam2Button.disabled = state.samBusy || state.samPrompts.length === 0 || !state.selectedFrame;
  }
  if (clearSam2Button) {
    clearSam2Button.disabled = state.samBusy || (state.samPrompts.length === 0 && !state.samMaskResult);
  }
}

async function refreshLabelSummary() {
  state.labelSummary = await loadJson("/api/labels/summary");
  renderIdPanel();
}

function syncSaveControls() {
  const dirty = state.pendingEdits.length > 0;
  if (saveEditsButton) {
    saveEditsButton.disabled = state.editBusy || !dirty;
    const group = saveEditsButton.closest(".save-group");
    if (group) group.classList.toggle("dirty", dirty);
  }
  if (pendingEditStatusEl) {
    if (state.editBusy) {
      pendingEditStatusEl.textContent = "Saving";
    } else {
      pendingEditStatusEl.textContent = dirty ? `${state.pendingEdits.length} unsaved` : "Saved";
    }
  }
}

async function savePendingEdits() {
  if (state.editBusy || !state.pendingEdits.length) return;
  state.editBusy = true;
  syncSelectionControls();
  setSelectionStatus(`Saving ${state.pendingEdits.length} edit${state.pendingEdits.length === 1 ? "" : "s"}`);
  let errorMessage = "";
  let finalMessage = "";
  try {
    const result = await postJson("/api/labels/save", { edits: state.pendingEdits });
    await refreshLabelSummary();
    if (result.saved) {
      state.pendingEdits = [];
      state.undoStack = [];
      finalMessage = "Saved edits";
    } else {
      finalMessage = result.message || "Nothing saved";
    }
  } catch (error) {
    console.error(error);
    errorMessage = error.message;
  } finally {
    state.editBusy = false;
    syncSelectionControls();
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

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const pi = polygon[i];
    const pj = polygon[j];
    const denom = Math.abs(pj.y - pi.y) < 1e-9 ? 1e-9 : pj.y - pi.y;
    const intersects =
      (pi.y > point.y) !== (pj.y > point.y) &&
      point.x < ((pj.x - pi.x) * (point.y - pi.y)) / denom + pi.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function nearestPointAt(canvasX, canvasY) {
  const camera = renderCamera();
  if (!camera) return null;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const threshold = 12 * (window.devicePixelRatio || 1);
  const threshold2 = threshold * threshold;
  let best = null;
  const positions = state.positions;
  for (let i = 0, j = 0; i < state.labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const x = layout.x0 + projected.u * layout.scale;
    const y = layout.y0 + projected.v * layout.scale;
    const dx = x - canvasX;
    const dy = y - canvasY;
    const dist2 = dx * dx + dy * dy;
    if (dist2 > threshold2) continue;
    if (!best || dist2 < best.dist2) {
      best = { index: i, dist2, projected };
    }
  }
  return best;
}

function selectByClickedId(event) {
  const point = canvasPointFromEvent(event);
  const nearest = nearestPointAt(point.x, point.y);
  if (!nearest) {
    setSelectionStatus("No point under cursor");
    return;
  }
  const label = state.labels[nearest.index];
  const indices = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    if (state.labels[i] === label) indices.push(i);
  }
  applySelection(indices, selectionOperationFromEvent(event));
  setSelectionStatus(`Selected label ${label}: ${state.selectedPointIndices.size.toLocaleString()} points`);
  render();
}

function projectedIndicesInsidePolygon(polygon) {
  const camera = renderCamera();
  if (!camera) return [];
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const result = [];
  const positions = state.positions;
  for (let i = 0, j = 0; i < state.labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const point = {
      x: layout.x0 + projected.u * layout.scale,
      y: layout.y0 + projected.v * layout.scale,
    };
    if (pointInPolygon(point, polygon)) result.push(i);
  }
  return result;
}

function beginLasso(event) {
  const point = canvasPointFromEvent(event);
  state.lasso = {
    points: [point],
    operation: selectionOperationFromEvent(event),
  };
  canvas.classList.add("dragging");
}

function updateLasso(event) {
  if (!state.lasso) return;
  const point = canvasPointFromEvent(event);
  const points = state.lasso.points;
  const previous = points[points.length - 1];
  const minStep = 2 * (window.devicePixelRatio || 1);
  if (Math.hypot(point.x - previous.x, point.y - previous.y) >= minStep) {
    points.push(point);
    render();
  }
}

function finishLasso() {
  if (!state.lasso) return;
  const lasso = state.lasso;
  state.lasso = null;
  canvas.classList.remove("dragging");
  if (lasso.points.length >= 3) {
    const indices = projectedIndicesInsidePolygon(lasso.points);
    applySelection(indices, lasso.operation);
    setSelectionStatus(`Lasso selected ${indices.length.toLocaleString()} visible points`);
  }
  render();
}

function clearSamState() {
  state.samPrompts = [];
  state.samMaskResult = null;
  state.samMaskImage = null;
  syncSamControls();
  render();
}

function addSamPrompt(event) {
  if (!state.selectedFrame || !state.exactFrameView) {
    setSelectionStatus("Select a frame before adding SAM2 prompts");
    return;
  }
  const point = canvasPointFromEvent(event);
  const camera = state.camera;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const view = currentBaseFrameView();
  const display = canvasToDisplayPoint(point.x, point.y, camera, layout, view);
  const source = displayToSource(state.selectedFrame, display.u, display.v);
  const inside =
    source.x >= 0 &&
    source.x < state.selectedFrame.width &&
    source.y >= 0 &&
    source.y < state.selectedFrame.height;
  if (!inside) {
    setSelectionStatus("SAM2 prompt is outside the frame image");
    return;
  }
  const label = event.shiftKey || event.altKey ? 0 : 1;
  state.samPrompts.push({
    frameId: state.selectedFrame.id,
    sourceX: source.x,
    sourceY: source.y,
    label,
  });
  state.samMaskResult = null;
  state.samMaskImage = null;
  syncSamControls();
  setSelectionStatus(`${state.samPrompts.length} SAM2 prompt${state.samPrompts.length === 1 ? "" : "s"} (${label ? "positive" : "negative"} added)`);
  render();
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load image"));
    image.src = src;
  });
}

function selectPointsInsideSamMask(result) {
  if (!state.selectedFrame || result.frameId !== state.selectedFrame.id) return [];
  const offscreen = document.createElement("canvas");
  offscreen.width = result.width;
  offscreen.height = result.height;
  const offscreenCtx = offscreen.getContext("2d");
  offscreenCtx.drawImage(state.samMaskImage, 0, 0, result.width, result.height);
  const alpha = offscreenCtx.getImageData(0, 0, result.width, result.height).data;
  const camera = frameCamera(state.selectedFrame);
  const positions = state.positions;
  const indices = [];
  for (let i = 0, j = 0; i < state.labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const source = displayToSource(state.selectedFrame, projected.u, projected.v);
    const x = Math.round(source.x);
    const y = Math.round(source.y);
    if (x < 0 || x >= result.width || y < 0 || y >= result.height) continue;
    if (alpha[(y * result.width + x) * 4 + 3] > 0) indices.push(i);
  }
  return indices;
}

async function runSam2() {
  if (state.samBusy || !state.selectedFrame || !state.samPrompts.length) return;
  const frameId = state.selectedFrame.id;
  const prompts = state.samPrompts.filter((prompt) => prompt.frameId === frameId);
  if (!prompts.length) {
    setSelectionStatus("SAM2 prompts belong to another frame");
    return;
  }
  state.samBusy = true;
  syncSamControls();
  setSelectionStatus("Running SAM2");
  try {
    const result = await postJson("/api/sam2/predict", {
      frameId,
      prompts,
      multimask: true,
    });
    const image = await loadImage(`data:image/png;base64,${result.maskOverlayPng}`);
    state.samMaskResult = result;
    state.samMaskImage = image;
    const indices = selectPointsInsideSamMask(result);
    applySelection(indices, "replace");
    setSelectionStatus(`SAM2 selected ${indices.length.toLocaleString()} points | score ${result.score.toFixed(3)}`);
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.samBusy = false;
    syncSamControls();
    render();
  }
}

function drawSourceImage(image, camera) {
  if (state.rotateFrames) {
    ctx.save();
    ctx.translate(camera.width, 0);
    ctx.rotate(Math.PI / 2);
    ctx.drawImage(image, 0, 0, camera.height, camera.width);
    ctx.restore();
  } else {
    ctx.drawImage(image, 0, 0, camera.width, camera.height);
  }
}

function drawFrameImage(camera, layout, view) {
  if (!state.showFrameImage || !state.exactFrameView || !state.selectedImage || !state.selectedImage.complete) return;
  const frameView = view || defaultFrameView(camera, state.selectedFrame ? state.selectedFrame.id : -1);
  const scale = layout.scale * frameView.zoom;
  ctx.save();
  ctx.translate(
    layout.x0 + layout.width * 0.5 - frameView.centerU * scale,
    layout.y0 + layout.height * 0.5 - frameView.centerV * scale,
  );
  ctx.scale(scale, scale);
  drawSourceImage(state.selectedImage, camera);
  ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
  ctx.fillRect(0, 0, camera.width, camera.height);
  if (
    state.samMaskImage &&
    state.samMaskImage.complete &&
    state.samMaskResult &&
    state.selectedFrame &&
    state.samMaskResult.frameId === state.selectedFrame.id
  ) {
    drawSourceImage(state.samMaskImage, camera);
  }
  ctx.restore();
}

function drawPoints(camera, layout) {
  const size = Math.max(1, state.pointSize * (window.devicePixelRatio || 1));
  const positions = state.positions;
  const labels = state.labels;
  for (let i = 0, j = 0; i < labels.length; i += 1, j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const x = layout.x0 + projected.u * layout.scale;
    const y = layout.y0 + projected.v * layout.scale;
    if (x < -4 || x > canvas.width + 4 || y < -4 || y > canvas.height + 4) continue;
    ctx.fillStyle = palette(labels[i]);
    ctx.fillRect(x, y, size, size);
  }
}

function drawSelectedPoints(camera, layout) {
  if (!state.selectedPointIndices.size) return;
  const radius = Math.max(3, (state.pointSize + 2) * (window.devicePixelRatio || 1));
  const positions = state.positions;
  ctx.save();
  ctx.lineWidth = Math.max(1.5, window.devicePixelRatio || 1);
  for (const index of state.selectedPointIndices) {
    const j = index * 3;
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const x = layout.x0 + projected.u * layout.scale;
    const y = layout.y0 + projected.v * layout.scale;
    if (x < -radius || x > canvas.width + radius || y < -radius || y > canvas.height + radius) continue;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
    ctx.stroke();
  }
  ctx.restore();
}

function drawLassoOverlay() {
  if (!state.lasso || state.lasso.points.length < 2) return;
  ctx.save();
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.strokeStyle = "rgba(124, 199, 255, 0.96)";
  ctx.fillStyle = "rgba(124, 199, 255, 0.12)";
  ctx.beginPath();
  ctx.moveTo(state.lasso.points[0].x, state.lasso.points[0].y);
  for (const point of state.lasso.points.slice(1)) {
    ctx.lineTo(point.x, point.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawSamPrompts(camera, layout, frameView) {
  if (!state.selectedFrame || !state.exactFrameView || !state.samPrompts.length) return;
  ctx.save();
  ctx.lineWidth = Math.max(1.5, window.devicePixelRatio || 1);
  for (const prompt of state.samPrompts) {
    if (prompt.frameId !== state.selectedFrame.id) continue;
    const display = sourceToDisplay(state.selectedFrame, prompt.sourceX, prompt.sourceY);
    const point = displayToCanvasPoint(display.u, display.v, camera, layout, frameView);
    ctx.beginPath();
    ctx.arc(point.x, point.y, 6 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
    ctx.fillStyle = prompt.label > 0 ? "rgba(69, 255, 159, 0.94)" : "rgba(255, 86, 94, 0.94)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
    ctx.stroke();
  }
  ctx.restore();
}

function render() {
  syncCanvasSize();
  ctx.fillStyle = "#070809";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!state.camera || !state.positions.length) {
    hud.textContent = "Loading";
    return;
  }

  const camera = renderCamera();
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const frameView = state.exactFrameView && state.selectedFrame
    ? ensureFrameView(state.camera, state.selectedFrame.id)
    : null;
  drawFrameImage(state.camera, layout, frameView);
  drawPoints(camera, layout);
  drawSelectedPoints(camera, layout);
  drawSamPrompts(state.camera, layout, frameView);
  drawLassoOverlay();

  const frameText = state.selectedFrame
    ? `Frame ${state.selectedFrame.id} | ${state.selectedFrame.imageName}${state.exactFrameView ? "" : " | RGB hidden until aligned"}`
    : "Free orbit";
  const transitionText = state.transition ? " | moving to frame" : "";
  const zoomText = frameView ? ` | view zoom ${frameView.zoom.toFixed(2)}x` : "";
  const samText = state.samPrompts.length ? ` | SAM prompts ${state.samPrompts.length}` : "";
  hud.textContent = `${frameText}${transitionText}${zoomText}${samText} | mode ${state.tool} | left-drag pan/select | right-drag rotate | wheel zoom`;
}

function panCamera(deltaX, deltaY) {
  if (!state.camera) return;
  const camera = state.camera;
  const layout = frameLayout(canvas.width, canvas.height, camera);
  const dpr = window.devicePixelRatio || 1;
  const depth = Math.max(dot(sub(state.target, camera.eye), camera.forward), Math.max(Number(state.bounds.radius) || 1, 1) * 0.25);
  const deltaU = (deltaX * dpr) / Math.max(layout.scale, 1e-6);
  const deltaV = (deltaY * dpr) / Math.max(layout.scale, 1e-6);
  const moveRight = mul(camera.right, -deltaU * depth / Math.max(camera.fx, 1e-6));
  const moveDown = mul(camera.down, -deltaV * depth / Math.max(camera.fy, 1e-6));
  const translation = add(moveRight, moveDown);
  state.camera = {
    ...camera,
    eye: add(camera.eye, translation),
  };
  state.target = add(state.target, translation);
  state.exactFrameView = false;
}

function panFrameView(deltaX, deltaY) {
  if (!state.exactFrameView || !state.selectedFrame || !state.frameView) return false;
  const layout = frameLayout(canvas.width, canvas.height, state.camera);
  const dpr = window.devicePixelRatio || 1;
  const denom = Math.max(layout.scale * state.frameView.zoom, 1e-6);
  state.frameView.centerU -= (deltaX * dpr) / denom;
  state.frameView.centerV -= (deltaY * dpr) / denom;
  clampFrameView(state.frameView, state.camera);
  return true;
}

function rotateCamera(deltaX, deltaY) {
  if (!state.camera) return;
  const speed = 0.006;
  let camera = state.camera;
  let eyeOffset = sub(camera.eye, state.target);
  if (length(eyeOffset) < 1e-6) {
    eyeOffset = mul(camera.forward, -Math.max(Number(state.bounds.radius) || 1, 1));
  }

  const screenUp = mul(camera.down, -1);
  const yaw = deltaX * speed * (state.rotateFrames ? -1 : 1);
  const pitch = -deltaY * speed;

  eyeOffset = rotateAroundAxis(eyeOffset, screenUp, yaw);
  let right = rotateAroundAxis(camera.right, screenUp, yaw);
  let down = rotateAroundAxis(camera.down, screenUp, yaw);

  eyeOffset = rotateAroundAxis(eyeOffset, right, pitch);
  down = rotateAroundAxis(down, right, pitch);

  const eye = add(state.target, eyeOffset);
  const forward = normalize(sub(state.target, eye));
  right = normalize(right);
  down = normalize(cross(forward, right));
  right = normalize(cross(down, forward));

  state.camera = {
    ...camera,
    eye,
    right,
    down,
    forward,
  };
  state.exactFrameView = false;
}

function exitExactFrameViewFor3D() {
  if (!state.exactFrameView || !state.selectedFrame || !state.frameView) return;
  state.camera = renderCamera();
  state.exactFrameView = false;
}

function zoomCamera(deltaY) {
  if (!state.camera) return;
  const factor = Math.exp(deltaY * 0.001);
  const eyeOffset = sub(state.camera.eye, state.target);
  const distance = Math.max(0.05, length(eyeOffset) * factor);
  const direction = normalize(eyeOffset);
  if (length(direction) < 1e-9) return;
  const eye = add(state.target, mul(direction, distance));
  state.camera = {
    ...state.camera,
    eye,
    forward: normalize(sub(state.target, eye)),
  };
  state.exactFrameView = false;
}

function zoomFrameView(event, deltaY) {
  if (!state.exactFrameView || !state.selectedFrame || !state.frameView) return false;
  const layout = frameLayout(canvas.width, canvas.height, state.camera);
  const dpr = window.devicePixelRatio || 1;
  const pointX = (event.clientX - canvas.getBoundingClientRect().left) * dpr;
  const pointY = (event.clientY - canvas.getBoundingClientRect().top) * dpr;
  const offsetX = pointX - layout.x0 - layout.width * 0.5;
  const offsetY = pointY - layout.y0 - layout.height * 0.5;
  const oldScale = Math.max(layout.scale * state.frameView.zoom, 1e-6);
  const anchorU = offsetX / oldScale + state.frameView.centerU;
  const anchorV = offsetY / oldScale + state.frameView.centerV;
  state.frameView.zoom = clamp(state.frameView.zoom * Math.exp(-deltaY * 0.001), 1, 32);
  const newScale = Math.max(layout.scale * state.frameView.zoom, 1e-6);
  state.frameView.centerU = anchorU - offsetX / newScale;
  state.frameView.centerV = anchorV - offsetY / newScale;
  clampFrameView(state.frameView, state.camera);
  return true;
}

function onPointerDown(event) {
  if (event.button !== 0 && event.button !== 2) return;
  event.preventDefault();
  cancelTransition();
  if (event.button === 0 && state.tool === "click-id") {
    selectByClickedId(event);
    return;
  }
  if (event.button === 0 && state.tool === "sam2") {
    addSamPrompt(event);
    return;
  }
  if (event.button === 0 && state.tool === "lasso") {
    beginLasso(event);
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is best-effort across browsers.
    }
    return;
  }
  if (event.button === 2) {
    exitExactFrameViewFor3D();
  }
  state.dragging = true;
  state.dragButton = event.button;
  state.lastPointer = [event.clientX, event.clientY];
  canvas.classList.add("dragging");
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
}

function onPointerMove(event) {
  if (state.lasso) {
    event.preventDefault();
    updateLasso(event);
    return;
  }
  if (!state.dragging) return;
  event.preventDefault();
  const dx = event.clientX - state.lastPointer[0];
  const dy = event.clientY - state.lastPointer[1];
  state.lastPointer = [event.clientX, event.clientY];
  if (state.dragButton === 2) {
    rotateCamera(dx, dy);
  } else if (!panFrameView(dx, dy)) {
    panCamera(dx, dy);
  }
  render();
}

function onPointerUp(event) {
  if (state.lasso) {
    event.preventDefault();
    finishLasso();
    try {
      canvas.releasePointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is best-effort across browsers.
    }
    return;
  }
  if (!state.dragging) return;
  state.dragging = false;
  state.dragButton = null;
  canvas.classList.remove("dragging");
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch (_error) {
    // Pointer capture is best-effort across browsers.
  }
}

function onWheel(event) {
  event.preventDefault();
  cancelTransition();
  if (!zoomFrameView(event, event.deltaY)) {
    zoomCamera(event.deltaY);
  }
  render();
}

function markActiveFrame() {
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    tile.classList.toggle("active", Number(tile.dataset.frameId) === (state.selectedFrame ? state.selectedFrame.id : -1));
    tile.classList.toggle("portrait", state.rotateFrames);
  }
}

function buildFilmstrip() {
  filmstrip.textContent = "";
  for (const frame of state.frames) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "frame-tile";
    tile.dataset.frameId = String(frame.id);
    tile.title = `${frame.id}: ${frame.imageName}`;

    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = frame.imageUrl;
    image.alt = frame.imageName;

    const label = document.createElement("div");
    label.className = "frame-label";
    label.textContent = `Frame ${frame.id}`;

    tile.append(image, label);
    tile.addEventListener("click", () => focusFrame(frame));
    filmstrip.appendChild(tile);
  }
  markActiveFrame();
}

async function initialize() {
  setStatus("Loading project");
  const [project, frames, points, labelSummary] = await Promise.all([
    loadJson("/api/project"),
    loadJson("/api/frames"),
    loadJson("/api/points"),
    loadJson("/api/labels/summary"),
  ]);

  state.project = project;
  state.frames = frames;
  state.positions = Float32Array.from(points.positions);
  state.labels = Int32Array.from(points.labels);
  state.sourceIndices = Array.isArray(points.indices) ? points.indices : [];
  state.labelSummary = labelSummary;
  state.bounds = points.bounds;
  state.pointSize = Number(pointSizeInput.value);
  state.showFrameImage = showFrameImageInput.checked;
  state.rotateFrames = rotateFramesInput.checked;

  setStatus(`${project.baselineName} | ${project.servedPointCount.toLocaleString()} points | ${project.frameCount} frames`);
  syncSelectionControls();
  syncToolButtons();
  buildFilmstrip();
  resetView();
}

for (const button of toolButtons) {
  button.addEventListener("click", () => {
    setTool(button.dataset.tool || "navigate");
  });
}
resetButton.addEventListener("click", resetView);
clearSelectionButton.addEventListener("click", clearSelection);
runSam2Button.addEventListener("click", runSam2);
clearSam2Button.addEventListener("click", clearSamState);
assignSelectionButton.addEventListener("click", assignSelectionToTarget);
extractSelectionButton.addEventListener("click", extractSelectionToNewLabel);
mergeSelectedIdsButton.addEventListener("click", mergeSelectedIdsToTarget);
saveEditsButton.addEventListener("click", savePendingEdits);
targetLabelInput.addEventListener("input", syncSelectionControls);
pointSizeInput.addEventListener("input", () => {
  state.pointSize = Number(pointSizeInput.value);
  render();
});
showFrameImageInput.addEventListener("change", () => {
  state.showFrameImage = showFrameImageInput.checked;
  render();
});
rotateFramesInput.addEventListener("change", () => {
  state.rotateFrames = rotateFramesInput.checked;
  if (state.selectedFrame) {
    focusFrame(state.selectedFrame);
  } else {
    render();
  }
  markActiveFrame();
});

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
canvas.addEventListener("pointerup", onPointerUp);
canvas.addEventListener("pointercancel", onPointerUp);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", onWheel, { passive: false });
window.addEventListener("resize", () => {
  const resized = syncCanvasSize();
  if (resized && !state.selectedFrame) resetView();
  render();
});
window.addEventListener("keydown", (event) => {
  const isUndo = (event.metaKey || event.ctrlKey) && !event.shiftKey && event.key.toLowerCase() === "z";
  if (!isUndo) return;
  event.preventDefault();
  undoLastEdit();
});
window.addEventListener("beforeunload", (event) => {
  if (!state.pendingEdits.length) return;
  event.preventDefault();
  event.returnValue = "";
});

initialize().catch((error) => {
  console.error(error);
  setStatus("Failed to load");
  hud.textContent = error.message;
});
