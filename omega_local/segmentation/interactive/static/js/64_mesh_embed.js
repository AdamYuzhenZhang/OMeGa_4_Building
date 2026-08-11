import * as pc from "/vendor/playcanvas/playcanvas.mjs";

const BRIDGE_SOURCE = "omega-mesh-embed";
const CAMERA_MESSAGE = "omega-mesh-camera";
const VISIBILITY_MESSAGE = "omega-mesh-visibility";
const COLOR_MODE_MESSAGE = "omega-mesh-color-mode";
const LOAD_CONCURRENCY = 2;
const parentWindow = window.parent;
const params = new URL(window.location.href).searchParams;
const sceneUrl = params.get("scene");
const sessionId = params.get("session");
const datasetId = params.get("datasetId") || "";

function notify(type, payload = {}) {
  parentWindow.postMessage(
    { source: BRIDGE_SOURCE, sessionId, type, ...payload },
    window.location.origin,
  );
}

function datasetContentUrl(path) {
  const parsed = new URL(path, window.location.origin);
  if (datasetId) parsed.searchParams.set("datasetId", datasetId);
  return `${parsed.pathname}${parsed.search}`;
}

function meshContentUrl(scene, part) {
  const base = `/api/3d-segmentation/runs/${encodeURIComponent(scene.runId)}` +
    `/meshes/${encodeURIComponent(scene.sceneId)}/${encodeURIComponent(part.partId)}.glb`;
  const versioned = part.contentVersion
    ? `${base}?v=${encodeURIComponent(part.contentVersion)}`
    : base;
  return datasetContentUrl(versioned);
}

function normalizedRegionColor(part) {
  const rgb = Array.isArray(part.regionColor) ? part.regionColor : [];
  if (rgb.length !== 3) return new pc.Color(0.9, 0.45, 0.2);
  return new pc.Color(...rgb.map(
    (value) => Math.max(0, Math.min(255, Number(value) || 0)) / 255
  ));
}

function rgbMaterial(source) {
  const material = source.clone();
  material.name = `${source.name || "mesh"}:omega-rgb`;
  material.useLighting = false;
  material.useSkybox = false;
  material.emissive.set(1, 1, 1);
  material.emissiveVertexColor = true;
  material.diffuseVertexColor = false;
  material.cull = pc.CULLFACE_NONE;
  material.update();
  return material;
}

function regionMaterial(part) {
  const material = new pc.StandardMaterial();
  const color = normalizedRegionColor(part);
  material.name = `${part.partId}:omega-region`;
  material.useLighting = false;
  material.useSkybox = false;
  material.emissive.copy(color);
  material.diffuse.copy(color);
  material.cull = pc.CULLFACE_NONE;
  material.update();
  return material;
}

function installPartMaterials(entity, part, mode) {
  const solid = regionMaterial(part);
  for (const render of entity.findComponents("render")) {
    for (const meshInstance of render.meshInstances) {
      const materials = {
        rgb: rgbMaterial(meshInstance.material),
        regions: solid,
      };
      meshInstance.omegaMaterials = materials;
      meshInstance.material = materials[mode] || materials.rgb;
    }
  }
}

function setPartColorMode(entity, mode) {
  for (const render of entity.findComponents("render")) {
    for (const meshInstance of render.meshInstances) {
      const target = meshInstance.omegaMaterials?.[mode];
      if (target) meshInstance.material = target;
    }
  }
}

function loadContainer(app, url) {
  return new Promise((resolve, reject) => {
    app.assets.loadFromUrl(url, "container", (error, asset) => {
      if (error) reject(new Error(String(error)));
      else resolve(asset);
    });
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

async function start() {
  if (!sessionId) throw new Error("Mesh bridge session is missing.");
  if (!sceneUrl) throw new Error("Mesh scene URL is missing.");
  notify("bridge-starting");
  const response = await fetch(sceneUrl, { cache: "no-store" });
  if (!response.ok) throw new Error(`Mesh scene request failed: ${response.status}`);
  const scene = await response.json();
  if (!Array.isArray(scene.parts) || !scene.parts.length) {
    throw new Error("Mesh scene contains no parts.");
  }

  const canvas = document.getElementById("meshCanvas");
  canvas.style.visibility = "hidden";
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: {
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    },
  });
  app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.start();

  const camera = new pc.Entity("omega-mesh-camera");
  camera.addComponent("camera", {
    clearColor: new pc.Color(0, 0, 0, 0),
    nearClip: 0.01,
    farClip: 1000,
  });
  app.root.addChild(camera);
  let externalCamera = null;
  camera.camera.calculateProjection = (matrix) => {
    if (!externalCamera) return;
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

  function applyExternalCamera() {
    if (!externalCamera) return;
    const { eye, forward, down, near, far, viewport } = externalCamera;
    const target = [
      eye[0] + forward[0],
      eye[1] + forward[1],
      eye[2] + forward[2],
    ];
    camera.setPosition(...eye);
    camera.lookAt(...target, -down[0], -down[1], -down[2]);
    camera.camera.nearClip = near;
    camera.camera.farClip = far;
    camera.camera.rect = new pc.Vec4(
      viewport.x,
      viewport.y,
      viewport.width,
      viewport.height,
    );
    canvas.style.visibility = "visible";
  }

  const parts = scene.parts.map((part) => ({
    ...part,
    partId: String(part.partId),
  }));
  const entities = new Map();
  const visibility = new Map(parts.map((part) => [part.partId, true]));
  const progress = new Map(parts.map((part) => [part.partId, 0]));
  let colorMode = String(scene.defaultColorMode || "rgb");
  const totalFaces = Math.max(
    parts.reduce((sum, part) => sum + Math.max(Number(part.triangleCount) || 0, 1), 0),
    1,
  );

  function reportPart(part, value) {
    progress.set(part.partId, value);
    const weighted = parts.reduce(
      (sum, item) => sum +
        Math.max(Number(item.triangleCount) || 0, 1) * (progress.get(item.partId) || 0),
      0,
    );
    notify("part-progress", {
      partId: part.partId,
      progress: value,
      sceneProgress: Math.round(weighted / totalFaces),
    });
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
      visibility.set(partId, Boolean(message.visible));
      const entity = entities.get(partId);
      if (entity) entity.enabled = Boolean(message.visible);
    } else if (message.type === COLOR_MODE_MESSAGE) {
      colorMode = String(message.mode || scene.defaultColorMode || "rgb");
      for (const entity of entities.values()) setPartColorMode(entity, colorMode);
    }
  });

  notify("bridge-ready");
  await runPool(parts, async (part) => {
    reportPart(part, 5);
    const asset = await loadContainer(app, meshContentUrl(scene, part));
    reportPart(part, 85);
    const entity = asset.resource.instantiateRenderEntity({
      castShadows: false,
      receiveShadows: false,
    });
    entity.name = `omega-mesh-part:${part.partId}`;
    installPartMaterials(entity, part, colorMode);
    entity.enabled = visibility.get(part.partId) !== false;
    app.root.addChild(entity);
    entities.set(part.partId, entity);
    reportPart(part, 100);
    notify("part-ready", { partId: part.partId });
  }, LOAD_CONCURRENCY);

  applyExternalCamera();
  notify("ready");
}

start().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  notify("error", { message });
  console.error(error);
});
