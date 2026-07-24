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
const exportPreviewButton = document.getElementById("exportPreview");
const showPointCloudInput = document.getElementById("showPointCloud");
const showColmapPointCloudInput = document.getElementById("showColmapPointCloud");
const useCleanedColmapInput = document.getElementById("useCleanedColmap");
const useSegmentedColmapInput = document.getElementById("useSegmentedColmap");
const showFeedforwardPointCloudInput = document.getElementById("showFeedforwardPointCloud");
const useSegmentedFeedforwardInput = document.getElementById("useSegmentedFeedforward");
const showOmegaFinalPointCloudInput = document.getElementById("showOmegaFinalPointCloud");
const useHybridOmegaFinalInput = document.getElementById("useHybridOmegaFinal");
const hybridOmegaFinalPointCloudStatusEl = document.getElementById("hybridOmegaFinalPointCloudStatus");
const useSegmentedOmegaFinalInput = document.getElementById("useSegmentedOmegaFinal");
const detectKeyframesButton = document.getElementById("detectKeyframes");
const reloadKeyframesButton = document.getElementById("reloadKeyframes");
const keyframeStatusEl = document.getElementById("keyframeStatus");
const loadProposalsButton = document.getElementById("loadProposals");
const runProposalsButton = document.getElementById("runProposals");
const regenerateProposalsButton = document.getElementById("regenerateProposals");
const proposalLayerOptionsEl = document.getElementById("proposalLayerOptions");
const proposalLayerSummaryEl = document.getElementById("proposalLayerSummary");
const proposalStatusEl = document.getElementById("proposalStatus");
const viewEvidenceStatusEl = document.getElementById("viewEvidenceStatus");
const generateNormalEvidenceButton = document.getElementById("generateNormalEvidence");
const generateDepthEvidenceButton = document.getElementById("generateDepthEvidence");
const generateDinoEvidenceButton = document.getElementById("generateDinoEvidence");
const normalEvidenceStatusEl = document.getElementById("normalEvidenceStatus");
const depthEvidenceStatusEl = document.getElementById("depthEvidenceStatus");
const dinoEvidenceStatusEl = document.getElementById("dinoEvidenceStatus");
const stableNormalDataTypeInput = document.getElementById("stableNormalDataType");
const stableNormalVariantInput = document.getElementById("stableNormalVariant");
const depthAnythingModelInput = document.getElementById("depthAnythingModel");
const frameMaskStatusEl = document.getElementById("frameMaskStatus");
const selectedProposalStatusEl = document.getElementById("selectedProposalStatus");
const clearProposalSelectionButton = document.getElementById("clearProposalSelection");
const regionStatusEl = document.getElementById("regionStatus");
const regionSuggestionCard = document.getElementById("regionSuggestionCard");
const regionSuggestionSwatch = document.getElementById("regionSuggestionSwatch");
const regionSuggestionTitle = document.getElementById("regionSuggestionTitle");
const regionSuggestionMeta = document.getElementById("regionSuggestionMeta");
const regionSuggestionReferenceStrip = document.getElementById("regionSuggestionReferenceStrip");
const useSuggestedRegionButton = document.getElementById("useSuggestedRegion");
const selectedRegionCard = document.getElementById("selectedRegionCard");
const selectedRegionSwatch = document.getElementById("selectedRegionSwatch");
const selectedRegionTitle = document.getElementById("selectedRegionTitle");
const selectedRegionMeta = document.getElementById("selectedRegionMeta");
const closeActiveRegionButton = document.getElementById("closeActiveRegion");
const regionNameInput = document.getElementById("regionName");
const regionNameLabel = document.getElementById("regionNameLabel");
const showRegionsInput = document.getElementById("showRegions");
const frameCompleteInput = document.getElementById("frameComplete");
const createRegionFromSelectionButton = document.getElementById("createRegionFromSelection");
const assignSelectionToRegionButton = document.getElementById("assignSelectionToRegion");
const addSelectionToRegionButton = document.getElementById("addSelectionToRegion");
const clearSelectionFromRegionsButton = document.getElementById("clearSelectionFromRegions");
const clearRegionFrameButton = document.getElementById("clearRegionFrame");
const renameRegionButton = document.getElementById("renameRegion");
const deleteRegionButton = document.getElementById("deleteRegion");
const regionReferenceStrip = document.getElementById("regionReferenceStrip");
const propagationStatusEl = document.getElementById("propagationStatus");
const propagationMethodInput = document.getElementById("propagationMethod");
const runPropagationButton = document.getElementById("runPropagation");
const openPropagationPreviewButton = document.getElementById("openPropagationPreview");
const sourceRefinementMethodInput = document.getElementById("sourceRefinementMethod");
const runSourceRefinementButton = document.getElementById("runSourceRefinement");
const openSourceRefinementPreviewButton = document.getElementById("openSourceRefinementPreview");
const regionPairMethodInput = document.getElementById("regionPairMethod");
const testRegionPairButton = document.getElementById("testRegionPair");
const segmentation3dStatusEl = document.getElementById("segmentation3dStatus");
const segmentation3dResultsEl = document.getElementById("segmentation3dResults");
const segmentation3dResultSummaryEl = document.getElementById("segmentation3dResultSummary");
const segmentation3dMethodInput = document.getElementById("segmentation3dMethod");
const segmentation3dInput = document.getElementById("segmentation3dInput");
const segmentation3dSourceField = document.getElementById("segmentation3dSourceField");
const segmentation3dSourceInput = document.getElementById("segmentation3dSource");
const segmentation3dManualWeightField = document.getElementById("segmentation3dManualWeightField");
const segmentation3dManualWeightInput = document.getElementById("segmentation3dManualWeight");
const segmentation3dPointBudgetInput = document.getElementById("segmentation3dPointBudget");
const segmentation3dSuperpointsInput = document.getElementById("segmentation3dSuperpoints");
const runSegmentation3dButton = document.getElementById("runSegmentation3d");
const propagationModal = document.getElementById("propagationModal");
const closePropagationModalButton = document.getElementById("closePropagationModal");
const propagationModalSummary = document.getElementById("propagationModalSummary");
const propagationResultsEl = document.getElementById("propagationResults");
const propagationLegendTop = document.getElementById("propagationLegendTop");
const propagationLegendBottom = document.getElementById("propagationLegendBottom");
const toolButtons = [...document.querySelectorAll("[data-tool]")];
const selectionOperationButtons = [...document.querySelectorAll("[data-selection-operation]")];
const methodControlPanels = [...document.querySelectorAll("[data-method-controls]")];
const sam2SizeInput = document.getElementById("sam2Size");
const sam2SizeValue = document.getElementById("sam2SizeValue");
const normalGrowAngleInput = document.getElementById("normalGrowAngle");
const normalGrowAngleValue = document.getElementById("normalGrowAngleValue");
const depthGrowEdgeInput = document.getElementById("depthGrowEdge");
const depthGrowEdgeValue = document.getElementById("depthGrowEdgeValue");
const rgbdCueSuperpixelsInput = document.getElementById("rgbdCueSuperpixels");
const rgbdCueSuperpixelsValue = document.getElementById("rgbdCueSuperpixelsValue");
const runRgbdCueDebugButton = document.getElementById("runRgbdCueDebug");
const rgbdCueDebugSelect = document.getElementById("rgbdCueDebugSelect");
const cursorBadge = document.getElementById("cursorBadge");
const showPromptDotsInput = document.getElementById("showPromptDots");
const lockRegionSelectionInput = document.getElementById("lockRegionSelection");
const clearSelectionButton = document.getElementById("clearSelection");
const selectionStatusEl = document.getElementById("selectionStatus");
const regionList = document.getElementById("regionList");
const idList = document.getElementById("idList");
const pointCloudStatusEl = document.getElementById("pointCloudStatus");
const colmapPointCloudStatusEl = document.getElementById("colmapPointCloudStatus");
const cleanedColmapPointCloudStatusEl = document.getElementById("cleanedColmapPointCloudStatus");
const segmentedColmapPointCloudStatusEl = document.getElementById("segmentedColmapPointCloudStatus");
const feedforwardPointCloudStatusEl = document.getElementById("feedforwardPointCloudStatus");
const segmentedFeedforwardPointCloudStatusEl = document.getElementById("segmentedFeedforwardPointCloudStatus");
const omegaFinalPointCloudStatusEl = document.getElementById("omegaFinalPointCloudStatus");
const segmentedOmegaFinalPointCloudStatusEl = document.getElementById("segmentedOmegaFinalPointCloudStatus");

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
  evidencePointClouds: {
    colmap: {
      visible: false,
      mode: "raw",
      cache: { raw: null, cleaned: null, segmented: null },
      positions: new Float32Array(),
      colors: new Uint8Array(),
      busy: false,
    },
    feedforward: {
      visible: false,
      mode: "raw",
      cache: { raw: null, segmented: null },
      positions: new Float32Array(),
      colors: new Uint8Array(),
      busy: false,
    },
    omegaFinal: {
      visible: false,
      mode: "raw",
      cache: { raw: null, hybrid: null, segmented: null },
      positions: new Float32Array(),
      colors: new Uint8Array(),
      busy: false,
    },
  },
  pointCloudSourcesStatus: null,
  showFrameImage: true,
  frameBackgroundMode: "rgb",
  rotateFrames: true,
  showRegions: false,
  keyframeStatus: null,
  keyframeBusy: false,
  proposalStatus: null,
  proposalLayerStatus: null,
  proposalLayers: { sam2: false },
  proposalPollTimer: null,
  viewEvidenceStatus: null,
  viewEvidencePollTimer: null,
  proposalOverlayImages: new Map(),
  proposalOverlayImage: null,
  proposalLayerOverlayImages: new Map(),
  proposalLayerOverlayImage: new Map(),
  proposalFrameInfo: null,
  proposalSortMode: "index",
  regionSortMode: "id",
  regionStatus: null,
  regionFrameInfo: null,
  activeRegionId: 0,
  regionOverlayImages: new Map(),
  regionOverlayImage: null,
  regionBusy: false,
  selectedProposalIds: new Set(),
  activeProposalId: 0,
  activeProposalLayer: "sam2",
  selectionOps: [],
  proposalSelectionOverlayImage: null,
  proposalSelectionOverlayStamp: 0,
  selectionPreviewArea: 0,
  selectionPreviewCoverage: 0,
  selectionPreviewProtectedArea: 0,
  selectionProtectRegions: false,
  dragging: false,
  dragButton: null,
  activePointerId: null,
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
  rgbdCuePrompts: [],
  showPromptDots: true,
  sam2MaskSize: 0.0,
  samBusy: false,
  samRefreshTimer: null,
  samRefreshSerial: 0,
  normalGrowAngleDeg: 14,
  cueSmoothness: 0.10,
  cueSuperpixels: 800,
  growBusy: false,
  rgbdCueRefreshTimer: null,
  rgbdCueRefreshSerial: 0,
  rgbdDebugBusy: false,
  rgbdDebugPanels: [],
  rgbdDebugActiveId: "",
  rgbdDebugOverlayImage: null,
  rgbdDebugFrameId: null,
  rgbdDebugSummary: null,
  selectionBusy: null,
  proposalUpdateBusy: false,
  propagationBusy: false,
  propagationResult: null,
  segmentation3dStatus: null,
  segmentation3dBusy: false,
  segmentation3dJobId: "",
  segmentation3dRunId: "",
  segmentation3dPollTimer: null,
  propagationMethodId: "",
  sourceRefinementMethodId: "",
  regionPairMethodId: "vggts_pair",
  previewExportBusy: false,
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

function selectionOperationSymbol(operation) {
  if (operation === "add") return "+";
  if (operation === "subtract") return "-";
  return "";
}

function isSelectionTool(tool = state.tool) {
  return (
    tool === "click-id" ||
    tool === "lasso" ||
    tool === "mask-pick" ||
    tool === "mask-lasso" ||
    tool === "sam2" ||
    tool === "normal-grow" ||
    tool === "rgbd-cue-select"
  );
}

function displaySelectionOperation() {
  return effectiveSelectionOperation();
}

function samPromptOperation(event = null) {
  return effectiveSelectionOperation(event) === "subtract" ? "subtract" : "add";
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
  const busy = state.selectionBusy;
  const operation = displaySelectionOperation();
  const editingGesture = Boolean(state.lasso) || !state.dragging;
  const showBusy = Boolean(busy && state.pointerInsideViewer && editingGesture);
  const showOperation = state.pointerInsideViewer && editingGesture && isSelectionTool() && (operation === "add" || operation === "subtract");
  const show = showBusy || showOperation;
  cursorBadge.hidden = !show;
  cursorBadge.style.display = show ? "grid" : "none";
  if (!show) {
    cursorBadge.textContent = "";
    cursorBadge.classList.remove("add", "subtract", "busy");
    return;
  }
  const rect = canvas.getBoundingClientRect();
  cursorBadge.textContent = showBusy ? "" : selectionOperationSymbol(operation);
  cursorBadge.title = showBusy ? `${busy.label}: ${busy.detail}` : "";
  cursorBadge.classList.toggle("busy", showBusy);
  cursorBadge.classList.toggle("add", !showBusy && operation === "add");
  cursorBadge.classList.toggle("subtract", !showBusy && operation === "subtract");
  cursorBadge.style.left = `${state.lastPointerClient[0] - rect.left}px`;
  cursorBadge.style.top = `${state.lastPointerClient[1] - rect.top}px`;
}

function selectionBusyTool(label) {
  const key = String(label || "").toLowerCase();
  if (key.includes("sam2")) return "sam2";
  if (key.includes("rgb-d") || key.includes("rgbd")) return "rgbd-cue-select";
  return "";
}

function syncSelectionBusyUi() {
  updateCursorBadge();
}

function setSelectionBusy(label, detail = "Running", tool = "") {
  state.selectionBusy = {
    label: String(label || "Selection"),
    detail: String(detail || "Running"),
    tool: tool || selectionBusyTool(label),
  };
  syncSelectionBusyUi();
}

function clearSelectionBusy(label = "") {
  if (!state.selectionBusy) return;
  if (label && state.selectionBusy.label !== label) return;
  state.selectionBusy = null;
  syncSelectionBusyUi();
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
