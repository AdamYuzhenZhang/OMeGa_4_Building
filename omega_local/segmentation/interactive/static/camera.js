export function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

export function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

export function cloneProjectionCamera(camera) {
  return {
    eye: camera.eye.slice(),
    right: normalize(camera.right),
    yAxis: normalize(camera.yAxis),
    forward: normalize(camera.forward),
    fx: camera.fx,
    fy: camera.fy,
    cx: camera.cx,
    cy: camera.cy,
    width: camera.width,
    height: camera.height,
  };
}

export function rotateVectorAroundAxis(vector, axis, angle) {
  const a = normalize(axis);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const term1 = [vector[0] * c, vector[1] * c, vector[2] * c];
  const crossTerm = cross(a, vector);
  const term2 = [crossTerm[0] * s, crossTerm[1] * s, crossTerm[2] * s];
  const term3Scale = dot(a, vector) * (1 - c);
  const term3 = [a[0] * term3Scale, a[1] * term3Scale, a[2] * term3Scale];
  return [
    term1[0] + term2[0] + term3[0],
    term1[1] + term2[1] + term3[1],
    term1[2] + term2[2] + term3[2],
  ];
}

export function frameLayout(width, height, frame) {
  const viewportWidth = Math.max(1, width);
  const viewportHeight = Math.max(1, height);
  const frameWidth = Math.max(1, frame.width);
  const frameHeight = Math.max(1, frame.height);
  const scale = Math.min(viewportWidth / frameWidth, viewportHeight / frameHeight);
  const drawW = frameWidth * scale;
  const drawH = frameHeight * scale;
  return {
    scale,
    x0: (viewportWidth - drawW) * 0.5,
    y0: (viewportHeight - drawH) * 0.5,
    width: drawW,
    height: drawH,
  };
}

export function defaultFrameView(camera, frameId) {
  return {
    frameId,
    zoom: 1,
    centerU: camera.width * 0.5,
    centerV: camera.height * 0.5,
  };
}

export function clampFrameView(view, camera) {
  view.zoom = clamp(view.zoom, 1, 24);
  view.centerU = clamp(view.centerU, 0, camera.width);
  view.centerV = clamp(view.centerV, 0, camera.height);
  return view;
}

export function applyFrameView(camera, view) {
  const frameView = clampFrameView({ ...view }, camera);
  return {
    ...camera,
    fx: camera.fx * frameView.zoom,
    fy: camera.fy * frameView.zoom,
    cx: (camera.cx - frameView.centerU) * frameView.zoom + camera.width * 0.5,
    cy: (camera.cy - frameView.centerV) * frameView.zoom + camera.height * 0.5,
  };
}

export function baseProjectionCameraFromFrame(frame, rotateFrames) {
  const pose = frame.poseWorldFromCamera;
  const right = normalize([pose[0], pose[4], pose[8]]);
  const yAxis = normalize([pose[1], pose[5], pose[9]]);
  const forward = normalize([pose[2], pose[6], pose[10]]);
  if (rotateFrames) {
    return {
      eye: [pose[3], pose[7], pose[11]],
      right: [-yAxis[0], -yAxis[1], -yAxis[2]],
      yAxis: right,
      forward,
      fx: frame.fy,
      fy: frame.fx,
      cx: frame.height - frame.cy,
      cy: frame.cx,
      width: frame.height,
      height: frame.width,
    };
  }
  return {
    eye: [pose[3], pose[7], pose[11]],
    right,
    yAxis,
    forward,
    fx: frame.fx,
    fy: frame.fy,
    cx: frame.cx,
    cy: frame.cy,
    width: frame.width,
    height: frame.height,
  };
}

export function projectWithCamera(point, camera) {
  const rel = sub(point, camera.eye);
  const z = dot(rel, camera.forward);
  if (z <= 0.01) return null;
  const u = camera.fx * (dot(rel, camera.right) / z) + camera.cx;
  const v = camera.fy * (dot(rel, camera.yAxis) / z) + camera.cy;
  return { u, v, z };
}

export function sourceCoordFromDisplay(frame, camera, displayU, displayV, rotateFrames) {
  if (displayU < 0 || displayU > camera.width || displayV < 0 || displayV > camera.height) {
    return null;
  }
  if (rotateFrames) {
    return {
      displayU,
      displayV,
      sourceX: clamp(displayV, 0, frame.width),
      sourceY: clamp(frame.height - displayU, 0, frame.height),
    };
  }
  return {
    displayU,
    displayV,
    sourceX: clamp(displayU, 0, frame.width),
    sourceY: clamp(displayV, 0, frame.height),
  };
}

export function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = (yi > point.y) !== (yj > point.y)
      && point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

export function smoothstep(t) {
  return t * t * (3 - 2 * t);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function lerpVec(a, b, t) {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
  ];
}

export function orthonormalProjectionBasis(rightSeed, forwardSeed) {
  const forward = normalize(forwardSeed);
  let right = rightSeed;
  const rightDotForward = dot(right, forward);
  right = normalize([
    right[0] - rightDotForward * forward[0],
    right[1] - rightDotForward * forward[1],
    right[2] - rightDotForward * forward[2],
  ]);
  if (!Number.isFinite(right[0])) right = [1, 0, 0];
  let yAxis = normalize(cross(forward, right));
  if (!Number.isFinite(yAxis[0])) yAxis = [0, 1, 0];
  right = normalize(cross(yAxis, forward));
  return { right, yAxis, forward };
}

export function interpolateProjectionCamera(start, end, t) {
  const eased = smoothstep(Math.max(0, Math.min(1, t)));
  const basis = orthonormalProjectionBasis(
    lerpVec(start.right, end.right, eased),
    lerpVec(start.forward, end.forward, eased),
  );
  return {
    eye: lerpVec(start.eye, end.eye, eased),
    right: basis.right,
    yAxis: basis.yAxis,
    forward: basis.forward,
    fx: lerp(start.fx, end.fx, eased),
    fy: lerp(start.fy, end.fy, eased),
    cx: lerp(start.cx, end.cx, eased),
    cy: lerp(start.cy, end.cy, eased),
    width: lerp(start.width, end.width, eased),
    height: lerp(start.height, end.height, eased),
  };
}
