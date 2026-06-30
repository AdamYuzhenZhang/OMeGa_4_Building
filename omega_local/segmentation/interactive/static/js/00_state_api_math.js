// Shared DOM handles, editor state, API helpers, and vector math.
const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d", { alpha: false });
const statusEl = document.getElementById("projectStatus");
const hud = document.getElementById("viewerHud");
const filmstrip = document.getElementById("filmstrip");
const resetButton = document.getElementById("resetView");
const pointSizeInput = document.getElementById("pointSize");
const showFrameImageInput = document.getElementById("showFrameImage");
const frameBackgroundModeInput = document.getElementById("frameBackgroundMode");
const rotateFramesInput = document.getElementById("rotateFrames");
const showPointCloudInput = document.getElementById("showPointCloud");
const detectKeyframesButton = document.getElementById("detectKeyframes");
const reloadKeyframesButton = document.getElementById("reloadKeyframes");
const keyframeStatusEl = document.getElementById("keyframeStatus");
const loadProposalsButton = document.getElementById("loadProposals");
const runProposalsButton = document.getElementById("runProposals");
const regenerateProposalsButton = document.getElementById("regenerateProposals");
const showProposalsInput = document.getElementById("showProposals");
const proposalStatusEl = document.getElementById("proposalStatus");
const viewEvidenceStatusEl = document.getElementById("viewEvidenceStatus");
const generateNormalEvidenceButton = document.getElementById("generateNormalEvidence");
const generateDepthEvidenceButton = document.getElementById("generateDepthEvidence");
const normalEvidenceStatusEl = document.getElementById("normalEvidenceStatus");
const depthEvidenceStatusEl = document.getElementById("depthEvidenceStatus");
const frameMaskStatusEl = document.getElementById("frameMaskStatus");
const selectedProposalStatusEl = document.getElementById("selectedProposalStatus");
const pendingMaskStatusEl = document.getElementById("pendingMaskStatus");
const clearProposalSelectionButton = document.getElementById("clearProposalSelection");
const updateProposalButton = document.getElementById("updateProposal");
const propagationStatusEl = document.getElementById("propagationStatus");
const propagationNeighborsInput = document.getElementById("propagationNeighbors");
const runPropagationButton = document.getElementById("runPropagation");
const propagationModal = document.getElementById("propagationModal");
const closePropagationModalButton = document.getElementById("closePropagationModal");
const propagationModalSummary = document.getElementById("propagationModalSummary");
const propagationResultsEl = document.getElementById("propagationResults");
const toolButtons = [...document.querySelectorAll("[data-tool]")];
const selectionOperationButtons = [...document.querySelectorAll("[data-selection-operation]")];
const selectionOperationStatusEl = document.getElementById("selectionOperationStatus");
const cursorBadge = document.getElementById("cursorBadge");
const runSam2Button = document.getElementById("runSam2");
const clearSam2Button = document.getElementById("clearSam2");
const clearSelectionButton = document.getElementById("clearSelection");
const selectionStatusEl = document.getElementById("selectionStatus");
const idPanelSummary = document.getElementById("idPanelSummary");
const idList = document.getElementById("idList");
const proposalSortButtons = [...document.querySelectorAll("[data-proposal-sort]")];
const targetLabelInput = document.getElementById("targetLabel");
const assignSelectionButton = document.getElementById("assignSelection");
const extractSelectionButton = document.getElementById("extractSelection");
const mergeSelectedIdsButton = document.getElementById("mergeSelectedIds");
const saveEditsButton = document.getElementById("saveEdits");
const pendingEditStatusEl = document.getElementById("pendingEditStatus");
const pointCloudStatusEl = document.getElementById("pointCloudStatus");

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
  showPointCloud: true,
  showFrameImage: true,
  frameBackgroundMode: "rgb",
  rotateFrames: true,
  showProposals: false,
  keyframeStatus: null,
  keyframeBusy: false,
  proposalStatus: null,
  proposalPollTimer: null,
  viewEvidenceStatus: null,
  viewEvidencePollTimer: null,
  proposalOverlayImages: new Map(),
  proposalOverlayImage: null,
  proposalFrameInfo: null,
  proposalSortMode: "index",
  selectedProposalIds: new Set(),
  activeProposalId: 0,
  selectionOps: [],
  proposalSelectionOverlayImage: null,
  proposalSelectionOverlayStamp: 0,
  selectionPreviewArea: 0,
  selectionPreviewCoverage: 0,
  dragging: false,
  dragButton: null,
  lastPointer: [0, 0],
  labelColors: new Map(),
  tool: "navigate",
  selectionOperationMode: "replace",
  modifierSelectionOperation: "",
  pointerInsideViewer: false,
  lastPointerClient: [0, 0],
  selectedPointIndices: new Set(),
  lasso: null,
  samPrompts: [],
  samBusy: false,
  editBusy: false,
  proposalUpdateBusy: false,
  propagationBusy: false,
  propagationResult: null,
  pendingEdits: [],
  undoStack: [],
};

function setStatus(text) {
  statusEl.textContent = text;
}

function setSelectionStatus(text) {
  if (selectionStatusEl) selectionStatusEl.textContent = text;
}

function eventSelectionOperation(event) {
  if (event && event.altKey) return "subtract";
  if (event && event.shiftKey) return "add";
  return "";
}

function effectiveSelectionOperation(event = null) {
  return eventSelectionOperation(event) || state.modifierSelectionOperation || state.selectionOperationMode || "replace";
}

function selectionOperationLabel(operation) {
  if (operation === "add") return "Add";
  if (operation === "subtract") return "Subtract";
  return "Replace";
}

function selectionOperationSymbol(operation) {
  if (operation === "add") return "+";
  if (operation === "subtract") return "-";
  return "";
}

function isSelectionTool(tool = state.tool) {
  return tool === "click-id" || tool === "lasso" || tool === "mask-pick" || tool === "mask-lasso" || tool === "sam2";
}

function displaySelectionOperation() {
  return effectiveSelectionOperation();
}

function samPromptOperation(event = null) {
  return effectiveSelectionOperation(event) === "subtract" ? "subtract" : "add";
}

function samPromptLabel(operation) {
  return operation === "subtract" ? "Negative" : "Positive";
}

function syncSelectionOperationControls() {
  const effective = displaySelectionOperation();
  const temporary = state.modifierSelectionOperation;
  for (const button of selectionOperationButtons) {
    const operation = button.dataset.selectionOperation || "";
    button.classList.toggle("active", operation === effective && effective !== "replace");
    button.classList.toggle("temporary", operation === temporary);
    button.setAttribute("aria-pressed", String(operation === state.selectionOperationMode && !temporary));
  }
  if (selectionOperationStatusEl) {
    if (state.tool === "sam2") {
      if (temporary) {
        selectionOperationStatusEl.textContent = `${samPromptLabel(temporary)} prompt while held`;
      } else if (state.selectionOperationMode === "add" || state.selectionOperationMode === "subtract") {
        selectionOperationStatusEl.textContent = `${samPromptLabel(state.selectionOperationMode)} prompt pinned`;
      } else {
        selectionOperationStatusEl.textContent = "Positive prompt by default";
      }
    } else if (temporary) {
      selectionOperationStatusEl.textContent = `${selectionOperationLabel(temporary)} while held`;
    } else if (state.selectionOperationMode === "add" || state.selectionOperationMode === "subtract") {
      selectionOperationStatusEl.textContent = `${selectionOperationLabel(state.selectionOperationMode)} pinned`;
    } else {
      selectionOperationStatusEl.textContent = "Replace by default";
    }
  }
  updateCursorBadge();
}

function setSelectionOperationMode(operation) {
  state.selectionOperationMode = state.selectionOperationMode === operation ? "replace" : operation;
  syncSelectionOperationControls();
  syncSamControls();
}

function syncModifierSelectionOperation(event) {
  const next = eventSelectionOperation(event);
  if (state.modifierSelectionOperation === next) return;
  state.modifierSelectionOperation = next;
  syncSelectionOperationControls();
}

function clearModifierSelectionOperation() {
  if (!state.modifierSelectionOperation) return;
  state.modifierSelectionOperation = "";
  syncSelectionOperationControls();
}

function updatePointerPosition(event) {
  state.pointerInsideViewer = true;
  state.lastPointerClient = [event.clientX, event.clientY];
  syncModifierSelectionOperation(event);
  updateCursorBadge();
}

function updateCursorBadge() {
  if (!cursorBadge) return;
  const operation = displaySelectionOperation();
  const editingGesture = Boolean(state.lasso) || !state.dragging;
  const show = state.pointerInsideViewer && editingGesture && isSelectionTool() && (operation === "add" || operation === "subtract");
  cursorBadge.hidden = !show;
  cursorBadge.style.display = show ? "grid" : "none";
  if (!show) {
    cursorBadge.textContent = "";
    cursorBadge.classList.remove("add", "subtract");
    return;
  }
  const rect = canvas.getBoundingClientRect();
  cursorBadge.textContent = selectionOperationSymbol(operation);
  cursorBadge.classList.toggle("add", operation === "add");
  cursorBadge.classList.toggle("subtract", operation === "subtract");
  cursorBadge.style.left = `${state.lastPointerClient[0] - rect.left}px`;
  cursorBadge.style.top = `${state.lastPointerClient[1] - rect.top}px`;
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
