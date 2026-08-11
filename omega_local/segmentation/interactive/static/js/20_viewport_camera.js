// 3D camera, exact frame viewport, frame image loading, and projection helpers.
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

async function loadFrameProposalLayers(frameId) {
  await loadProposalFrameInfo(frameId);
  await loadActiveProposalOverlay(frameId);
  if (typeof loadActiveRegionOverlay === "function") {
    await loadActiveRegionOverlay(frameId);
  }
  if (typeof loadActiveGaussianFlatsPlaneMask === "function") {
    await loadActiveGaussianFlatsPlaneMask(frameId);
  }
  await loadProposalSelectionOverlay();
}

function selectedFrameBackgroundUrl(frame) {
  const mode = state.frameBackgroundMode || "rgb";
  if (mode === "rgb") return frame.imageUrl;
  if (mode.startsWith("objectgs:")) {
    const runId = mode.slice("objectgs:".length);
    return datasetUrl(
      `/api/3d-segmentation/runs/${encodeURIComponent(runId)}` +
      `/objectgs-rgb/${Number(frame.id)}.png`,
    );
  }
  if (!viewEvidenceReady() || !viewEvidenceModeReady(mode)) return frame.imageUrl;
  return viewEvidenceImageUrl(frame.id, mode);
}

function loadSelectedFrameImage(frame) {
  const mode = state.frameBackgroundMode || "rgb";
  const image = new Image();
  const src = selectedFrameBackgroundUrl(frame);
  image.onload = () => {
    if (state.selectedFrame && state.selectedFrame.id === frame.id) {
      state.selectedImage = image;
      render();
    }
  };
  image.onerror = () => {
    if (mode !== "rgb" && state.selectedFrame && state.selectedFrame.id === frame.id) {
      const fallback = new Image();
      fallback.onload = () => {
        if (state.selectedFrame && state.selectedFrame.id === frame.id) {
          state.selectedImage = fallback;
          render();
        }
      };
      fallback.src = frame.imageUrl;
      setSelectionStatus(`No ${mode} view evidence for frame ${frame.id}; showing RGB`);
    }
  };
  image.src = src;
  state.selectedImage = image;
}

function focusFrame(frame) {
  if (!state.camera) return;
  cancelTransition();
  if (!state.selectedFrame || state.selectedFrame.id !== frame.id) {
    resetSamState();
    state.proposalOverlayImage = null;
    state.proposalFrameInfo = null;
    state.regionOverlayImage = null;
    state.regionFrameInfo = null;
    state.selectedProposalIds.clear();
    state.activeProposalId = 0;
    state.selectionOps = [];
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
    if (typeof resetRgbdCuePrompts === "function") {
      resetRgbdCuePrompts();
    } else if (typeof resetRgbdCueDebug === "function") {
      resetRgbdCueDebug();
    }
    syncSamControls();
    syncSelectionOperationControls();
    syncMaskEditControls();
  }
  const startCamera = state.exactFrameView && state.selectedFrame && state.frameView
    ? renderCamera()
    : cloneCamera(state.camera);
  const endCamera = frameCamera(frame);
  const endTarget = frameTarget(endCamera);
  state.selectedFrame = frame;
  state.exactFrameView = false;
  state.frameView = defaultFrameView(endCamera, frame.id);

  loadSelectedFrameImage(frame);
  loadFrameProposalLayers(frame.id).catch((error) => {
    console.error(error);
    setSelectionStatus(error.message);
  });
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
    syncMaskEditControls();
    if (typeof syncRgbdCueDebugControls === "function") syncRgbdCueDebugControls();
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
  return effectiveSelectionOperation(event);
}
