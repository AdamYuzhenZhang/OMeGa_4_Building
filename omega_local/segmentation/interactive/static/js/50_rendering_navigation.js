// Canvas drawing plus pan/rotate/zoom navigation.
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

function drawFrameLayers(camera, layout, view) {
  if (!state.exactFrameView) return;
  const frameView = view || defaultFrameView(camera, state.selectedFrame ? state.selectedFrame.id : -1);
  const scale = layout.scale * frameView.zoom;
  ctx.save();
  ctx.translate(
    layout.x0 + layout.width * 0.5 - frameView.centerU * scale,
    layout.y0 + layout.height * 0.5 - frameView.centerV * scale,
  );
  ctx.scale(scale, scale);
  if (state.showFrameImage && state.selectedImage && state.selectedImage.complete) {
    drawSourceImage(state.selectedImage, camera);
    ctx.fillStyle = "rgba(0, 0, 0, 0.22)";
    ctx.fillRect(0, 0, camera.width, camera.height);
  }
  if (
    state.proposalLayerOverlayImage &&
    state.proposalLayerOverlayImage.size
  ) {
    for (const layer of proposalLayerDrawOrder(visibleProposalLayers())) {
      const image = state.proposalLayerOverlayImage.get(layer);
      if (image && image.complete) drawSourceImage(image, camera);
    }
  } else if (
    proposalReady() &&
    state.proposalOverlayImage &&
    state.proposalOverlayImage.complete
  ) {
    drawSourceImage(state.proposalOverlayImage, camera);
  }
  if (
    state.showRegions &&
    state.regionOverlayImage &&
    state.regionOverlayImage.complete
  ) {
    drawSourceImage(state.regionOverlayImage, camera);
  }
  if (
    state.rgbdDebugOverlayImage &&
    state.rgbdDebugOverlayImage.complete &&
    state.selectedFrame &&
    Number(state.rgbdDebugFrameId) === Number(state.selectedFrame.id)
  ) {
    drawSourceImage(state.rgbdDebugOverlayImage, camera);
  }
  if (state.proposalSelectionOverlayImage && state.proposalSelectionOverlayImage.complete) {
    drawSourceImage(state.proposalSelectionOverlayImage, camera);
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

function drawEvidencePoints(layer, camera, layout) {
  const positions = layer.positions;
  const colors = layer.colors;
  if (!positions.length) return;
  const size = Math.max(1, state.pointSize * 0.72 * (window.devicePixelRatio || 1));
  let activeColor = -1;
  ctx.save();
  for (let j = 0; j < positions.length; j += 3) {
    const projected = project([positions[j], positions[j + 1], positions[j + 2]], camera);
    if (!projected) continue;
    const x = layout.x0 + projected.u * layout.scale;
    const y = layout.y0 + projected.v * layout.scale;
    if (x < -4 || x > canvas.width + 4 || y < -4 || y > canvas.height + 4) continue;
    const colorKey = ((colors[j] >> 3) << 10) | ((colors[j + 1] >> 3) << 5) | (colors[j + 2] >> 3);
    if (colorKey !== activeColor) {
      const red = (((colorKey >> 10) & 31) << 3) + 4;
      const green = (((colorKey >> 5) & 31) << 3) + 4;
      const blue = ((colorKey & 31) << 3) + 4;
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      activeColor = colorKey;
    }
    ctx.fillRect(x, y, size, size);
  }
  ctx.restore();
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
    ctx.fillStyle = "rgba(69, 210, 255, 0.88)";
    ctx.fill();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
    ctx.stroke();
  }
  ctx.restore();
}

function drawLassoOverlay() {
  if (!state.lasso || state.lasso.points.length < 2) return;
  const operation = state.lasso.operation || "replace";
  ctx.save();
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  if (operation === "add") {
    ctx.strokeStyle = "rgba(89, 236, 147, 0.96)";
    ctx.fillStyle = "rgba(89, 236, 147, 0.12)";
  } else if (operation === "subtract") {
    ctx.strokeStyle = "rgba(255, 103, 111, 0.96)";
    ctx.fillStyle = "rgba(255, 103, 111, 0.12)";
  } else {
    ctx.strokeStyle = "rgba(124, 199, 255, 0.96)";
    ctx.fillStyle = "rgba(124, 199, 255, 0.12)";
  }
  ctx.beginPath();
  ctx.moveTo(state.lasso.points[0].x, state.lasso.points[0].y);
  for (const point of state.lasso.points.slice(1)) {
    ctx.lineTo(point.x, point.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawSamPrompts(camera, layout, frameView) {
  if (!state.showPromptDots) return;
  drawPromptDots(state.samPrompts, camera, layout, frameView, 6, "sam");
}

function drawRgbdCuePrompts(camera, layout, frameView) {
  if (!state.showPromptDots) return;
  drawPromptDots(state.rgbdCuePrompts, camera, layout, frameView, 7, "rgbd");
}

function drawPromptDots(prompts, camera, layout, frameView, radius, kind) {
  if (!state.selectedFrame || !state.exactFrameView || !prompts.length) return;
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.lineWidth = Math.max(1.5, dpr);
  for (const prompt of prompts) {
    if (prompt.frameId !== state.selectedFrame.id) continue;
    const display = sourceToDisplay(state.selectedFrame, prompt.sourceX, prompt.sourceY);
    const point = displayToCanvasPoint(display.u, display.v, camera, layout, frameView);
    const promptRadius = radius * dpr;

    ctx.beginPath();
    ctx.arc(point.x, point.y, promptRadius, 0, Math.PI * 2);
    ctx.fillStyle = prompt.label > 0 ? "rgba(69, 255, 159, 0.94)" : "rgba(255, 86, 94, 0.94)";
    ctx.fill();
    ctx.strokeStyle = kind === "rgbd" ? "rgba(255, 255, 255, 0.92)" : "rgba(0, 0, 0, 0.9)";
    ctx.lineWidth = Math.max(1.5, dpr);
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
  drawFrameLayers(state.camera, layout, frameView);
  for (const layer of Object.values(state.evidencePointClouds)) {
    if (layer.visible) drawEvidencePoints(layer, camera, layout);
  }
  if (state.showPointCloud) {
    drawPoints(camera, layout);
    drawSelectedPoints(camera, layout);
  }
  drawSamPrompts(state.camera, layout, frameView);
  drawRgbdCuePrompts(state.camera, layout, frameView);
  drawLassoOverlay();

  const activeSamPromptCount = state.selectedFrame
    ? state.samPrompts.filter((prompt) => Number(prompt.frameId) === Number(state.selectedFrame.id)).length
    : 0;
  const activeRgbdClickCount = state.selectedFrame
    ? state.rgbdCuePrompts.filter((prompt) => Number(prompt.frameId) === Number(state.selectedFrame.id)).length
    : 0;
  const shortFrameName = state.selectedFrame
    ? String(state.selectedFrame.imageName || "").replace(/\.[^.]+$/, "")
    : "";
  const viewState = state.transition ? "moving" : state.exactFrameView ? "aligned" : "3D";
  const items = state.selectedFrame
    ? [`F${state.selectedFrame.id}`, shortFrameName, viewState]
    : ["3D orbit"];
  if (frameView) items.push(`${frameView.zoom.toFixed(2)}x`);
  if (state.frameBackgroundMode && state.frameBackgroundMode !== "rgb") items.push(state.frameBackgroundMode);
  if (activeSamPromptCount) items.push(`SAM ${activeSamPromptCount}`);
  if (activeRgbdClickCount) items.push(`RGB-D ${activeRgbdClickCount}`);
  if (state.selectionOps.length) items.push(`${state.selectionOps.length} sel op${state.selectionOps.length === 1 ? "" : "s"}`);
  if (!state.showPointCloud) items.push("points off");
  for (const [sourceId, layer] of Object.entries(state.evidencePointClouds)) {
    if (!layer.visible) continue;
    const sourceName = layer.displayName || (sourceId === "colmap"
      ? "COLMAP"
      : sourceId === "feedforward"
        ? "Init"
        : sourceId === "omegaFinal"
          ? "Optimized"
          : "SAI3D");
    const mode = sourceId.startsWith("segmentation3d:")
      ? ""
      : layer.mode === "cleaned"
        ? "clean"
        : layer.mode;
    items.push(`${sourceName}${mode ? ` ${mode}` : ""} ${formatCount(layer.positions.length / 3)}`);
  }
  hud.textContent = items.filter(Boolean).join(" | ");
  hud.title = state.selectedFrame
    ? `Frame ${state.selectedFrame.id}: ${state.selectedFrame.imageName || ""}`
    : "3D orbit";
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
