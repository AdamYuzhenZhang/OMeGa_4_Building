const SUPER_SPLAT_MODULE_URL = "/vendor/supersplat/index.js?v=omega-multipart-1";

const BRIDGE_SOURCE = "omega-splat-embed";
const CAMERA_MESSAGE = "omega-splat-camera";
const VISIBILITY_MESSAGE = "omega-splat-visibility";
const COLOR_MODE_MESSAGE = "omega-splat-color-mode";
const LOAD_CONCURRENCY = 4;
const COLOR_SWITCH_POINT_BUDGET = 125000;
const COLOR_SWITCH_MAX_PARTS = 8;
const parentWindow = window.parent;
const params = new URL(window.location.href).searchParams;
const sceneUrl = params.get("scene");
const settingsUrl = params.get("settings");
const sessionId = params.get("session");
const datasetId = params.get("datasetId") || "";

function notify(type, payload = {}) {
  parentWindow.postMessage(
    { source: BRIDGE_SOURCE, sessionId, type, ...payload },
    window.location.origin,
  );
}

async function fetchRequired(url, label, responseType, cache = "force-cache") {
  if (!url) throw new Error(`${label} URL is missing.`);
  const response = await fetch(url, { cache });
  if (!response.ok) throw new Error(`${label} request failed: ${response.status}`);
  return responseType === "json" ? response.json() : response.text();
}

function gaussianContentUrl(scene, part) {
  const base = `/api/3d-segmentation/runs/${encodeURIComponent(scene.runId)}` +
    `/gaussians/${encodeURIComponent(part.variantId)}.ply`;
  const versioned = part.contentVersion
    ? `${base}?v=${encodeURIComponent(part.contentVersion)}`
    : base;
  if (!datasetId) return versioned;
  const parsed = new URL(versioned, window.location.origin);
  parsed.searchParams.set("datasetId", datasetId);
  return `${parsed.pathname}${parsed.search}`;
}

async function fetchGaussian(scene, part) {
  const url = gaussianContentUrl(scene, part);
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Gaussian part ${part.displayName || part.partId} failed: ${response.status}`);
  }
  return { response, url };
}

function installViewerShell(shellHtml) {
  const shell = new DOMParser().parseFromString(shellHtml, "text/html");
  for (const script of shell.body.querySelectorAll("script")) script.remove();
  const children = [...shell.body.children].map((node) => document.importNode(node, true));
  document.body.replaceChildren(...children);
  document.documentElement.style.setProperty("--canvas-opacity", "1");
  const canvas = document.getElementById("application-canvas");
  if (!canvas) throw new Error("SuperSplat viewer shell has no application canvas.");
  canvas.style.visibility = "hidden";
  return canvas;
}

function rotateForSuperSplatImport(vector) {
  // SuperSplat rotates imported PLY scenes 180 degrees around Z.
  return [-vector[0], -vector[1], vector[2]];
}

function waitForViewerReady(state) {
  if (state.readyToRender) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = window.setInterval(() => {
      if (!state.readyToRender) return;
      window.clearInterval(timer);
      resolve();
    }, 60);
  });
}

async function runPool(items, worker, concurrency) {
  let cursor = 0;
  const workers = Array.from(
    { length: Math.min(Math.max(concurrency, 1), items.length) },
    async () => {
      while (cursor < items.length) {
        const index = cursor;
        cursor += 1;
        await worker(items[index]);
      }
    },
  );
  await Promise.all(workers);
}
const REGION_COLOR_MODIFIER = {
  glsl: `
uniform vec3 omegaRegionColor;
uniform float omegaRegionMix;

void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}
void modifySplatColor(vec3 center, inout vec4 color) {
  color.rgb = mix(color.rgb, omegaRegionColor, omegaRegionMix);
}
`,
  wgsl: `
uniform omegaRegionColor: vec3f;
uniform omegaRegionMix: f32;

fn modifySplatCenter(center: ptr<function, vec3f>) {}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
  (*color).rgb = mix((*color).rgb, uniform.omegaRegionColor, uniform.omegaRegionMix);
}
`,
};

function normalizedRegionColor(part) {
  const rgb = Array.isArray(part.regionColor) ? part.regionColor : [];
  if (rgb.length !== 3) return null;
  return new Float32Array(
    rgb.map((value) => Math.max(0, Math.min(255, Number(value) || 0)) / 255),
  );
}

function installPartColorControl(entity, part, mode) {
  if (!entity || !entity.gsplat || part.sourceColorMode !== "rgb") return;
  const color = normalizedRegionColor(part);
  if (!color) return;
  entity.gsplat.setParameter("omegaRegionColor", color);
  entity.gsplat.setParameter("omegaRegionMix", mode === "regions" ? 1 : 0);
  entity.gsplat.setWorkBufferModifier(REGION_COLOR_MODIFIER);
  entity.omegaRegionMode = mode;
}

function setPartColorMode(entity, part, mode) {
  if (
    !entity ||
    !entity.gsplat ||
    part.sourceColorMode !== "rgb" ||
    entity.omegaRegionMode === mode
  ) {
    return false;
  }
  entity.gsplat.setParameter("omegaRegionMix", mode === "regions" ? 1 : 0);
  entity.omegaRegionMode = mode;
  return true;
}

async function start() {
  if (!sessionId) throw new Error("Gaussian bridge session is missing.");
  if (!sceneUrl) throw new Error("Gaussian scene URL is missing.");
  notify("bridge-starting");
  const { loadGsplat, main } = await import(SUPER_SPLAT_MODULE_URL);
  const [scene, settings, shellHtml] = await Promise.all([
    fetchRequired(sceneUrl, "Scene", "json", "no-store"),
    fetchRequired(settingsUrl, "Settings", "json"),
    fetchRequired("/vendor/supersplat/index.html", "Viewer shell", "text"),
  ]);
  if (!Array.isArray(scene.parts) || !scene.parts.length) {
    throw new Error("Gaussian scene contains no parts.");
  }

  const parts = scene.parts.map((part) => ({
    ...part,
    partId: String(part.partId || part.variantId),
  }));
  const partById = new Map(parts.map((part) => [part.partId, part]));
  const bootstrap = [...parts].sort(
    (a, b) => (Number(b.pointCount) || 0) - (Number(a.pointCount) || 0),
  )[0];
  const pending = parts.filter((part) => part.partId !== bootstrap.partId);
  const progress = new Map(parts.map((part) => [part.partId, 0]));
  const entities = new Map();
  const desiredVisibility = new Map(parts.map((part) => [part.partId, true]));
  let desiredColorMode = String(scene.defaultColorMode || "rgb");
  let colorSwitchSerial = 0;
  const totalPoints = Math.max(
    parts.reduce((sum, part) => sum + Math.max(Number(part.pointCount) || 0, 1), 0),
    1,
  );

  function reportPartProgress(part, value) {
    const bounded = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
    progress.set(part.partId, bounded);
    let weighted = 0;
    for (const item of parts) {
      weighted += Math.max(Number(item.pointCount) || 0, 1) * (progress.get(item.partId) || 0);
    }
    notify("part-progress", {
      partId: part.partId,
      progress: bounded,
      sceneProgress: Math.round(weighted / totalPoints),
    });
  }

  const bootstrapRequest = await fetchGaussian(scene, bootstrap);
  const canvas = installViewerShell(shellHtml);
  const viewer = await main(canvas, settings, {
    contentUrl: bootstrapRequest.url,
    contents: Promise.resolve(bootstrapRequest.response),
    entityName: `omega-part:${bootstrap.partId}`,
    noui: true,
    noanim: true,
    // Keep CameraFrame active for high-precision Gaussian accumulation.
    nofx: false,
    renderer: "webgl",
    aa: true,
  });
  const { app, camera, state } = viewer.global;
  let externalCamera = null;
  let projectionInstalled = false;

  camera.camera.clearColor.a = 0;

  function applyExternalCamera() {
    if (!externalCamera) return;
    if (!projectionInstalled) {
      camera.camera.calculateProjection = (matrix) => {
        const { intrinsics, near, far } = externalCamera;
        matrix.setFrustum(
          -intrinsics.cx * near / intrinsics.fx,
          (intrinsics.width - intrinsics.cx) * near / intrinsics.fx,
          -(intrinsics.height - intrinsics.cy) * near / intrinsics.fy,
          intrinsics.cy * near / intrinsics.fy,
          near,
          far,
        );
      };
      projectionInstalled = true;
    }

    const eye = rotateForSuperSplatImport(externalCamera.eye);
    const forward = rotateForSuperSplatImport(externalCamera.forward);
    const down = rotateForSuperSplatImport(externalCamera.down);
    const target = [
      eye[0] + forward[0],
      eye[1] + forward[1],
      eye[2] + forward[2],
    ];
    const up = [-down[0], -down[1], -down[2]];
    camera.setPosition(eye[0], eye[1], eye[2]);
    camera.lookAt(target[0], target[1], target[2], up[0], up[1], up[2]);
    camera.camera.nearClip = externalCamera.near;
    camera.camera.farClip = externalCamera.far;

    const rect = externalCamera.viewport;
    const cameraRect = camera.camera.rect;
    cameraRect.set(rect.x, rect.y, rect.width, rect.height);
    camera.camera.rect = cameraRect;
    canvas.style.visibility = "visible";
    app.renderNextFrame = true;
  }

  app.on("update", applyExternalCamera);

  async function switchColorMode(mode) {
    desiredColorMode = mode;
    const serial = ++colorSwitchSerial;
    const pendingEntities = [...entities].filter(([partId, entity]) => (
      entity.omegaRegionMode !== mode &&
      partById.get(partId)?.sourceColorMode === "rgb"
    ));
    const total = pendingEntities.reduce(
      (sum, [partId]) => sum + Math.max(Number(partById.get(partId)?.pointCount) || 0, 1),
      0,
    );
    let completed = 0;
    let cursor = 0;
    notify("color-progress", { mode, progress: pendingEntities.length ? 0 : 100 });

    while (cursor < pendingEntities.length && serial === colorSwitchSerial) {
      let batchPoints = 0;
      let batchParts = 0;
      while (
        cursor < pendingEntities.length &&
        batchParts < COLOR_SWITCH_MAX_PARTS &&
        (batchPoints < COLOR_SWITCH_POINT_BUDGET || batchParts === 0)
      ) {
        const [partId, entity] = pendingEntities[cursor];
        const part = partById.get(partId);
        const points = Math.max(Number(part?.pointCount) || 0, 1);
        setPartColorMode(entity, part, mode);
        batchPoints += points;
        completed += points;
        batchParts += 1;
        cursor += 1;
      }
      app.renderNextFrame = true;
      notify("color-progress", {
        mode,
        progress: total ? Math.round(100 * completed / total) : 100,
      });
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }
    if (serial !== colorSwitchSerial) return;
    app.renderNextFrame = true;
    notify("color-ready", { mode });
  }
  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || event.source !== parentWindow) return;
    const message = event.data || {};
    if (message.sessionId !== sessionId) return;
    if (message.type === CAMERA_MESSAGE) {
      externalCamera = message.camera || null;
      applyExternalCamera();
    } else if (message.type === VISIBILITY_MESSAGE) {
      const partId = String(message.partId || "");
      desiredVisibility.set(partId, Boolean(message.visible));
      const entity = entities.get(partId);
      if (entity) entity.enabled = Boolean(message.visible);
      app.renderNextFrame = true;
    } else if (message.type === COLOR_MODE_MESSAGE) {
      const mode = String(message.mode || scene.defaultColorMode || "rgb");
      if (mode !== desiredColorMode || [...entities.values()].some(
        (entity) => entity.omegaRegionMode !== mode
      )) {
        void switchColorMode(mode);
      }
    }
  });

  notify("bridge-ready");
  const bootstrapProgressTimer = window.setInterval(() => {
    reportPartProgress(bootstrap, Number(state.progress) || 0);
  }, 120);
  await waitForViewerReady(state);
  window.clearInterval(bootstrapProgressTimer);
  const bootstrapEntity = app.root.findByName(`omega-part:${bootstrap.partId}`);
  if (!bootstrapEntity) throw new Error(`Loaded Gaussian part is missing: ${bootstrap.partId}`);
  installPartColorControl(bootstrapEntity, bootstrap, desiredColorMode);
  bootstrapEntity.enabled = desiredVisibility.get(bootstrap.partId) !== false;
  entities.set(bootstrap.partId, bootstrapEntity);
  reportPartProgress(bootstrap, 100);
  notify("part-ready", { partId: bootstrap.partId });
  applyExternalCamera();

  await runPool(pending, async (part) => {
    const request = await fetchGaussian(scene, part);
    const entity = await loadGsplat(
      app,
      {
        contentUrl: request.url,
        contents: Promise.resolve(request.response),
        entityName: `omega-part:${part.partId}`,
        aa: true,
      },
      (value) => reportPartProgress(part, value),
    );
    installPartColorControl(entity, part, desiredColorMode);
    entity.enabled = desiredVisibility.get(part.partId) !== false;
    entities.set(part.partId, entity);
    reportPartProgress(part, 100);
    notify("part-ready", { partId: part.partId });
    app.renderNextFrame = true;
  }, LOAD_CONCURRENCY);

  notify("ready");
}

start().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  notify("error", { message });
  console.error(error);
});
