/* Three.js scene shell: renderer, camera, lights, environment, camera moves. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { clamp, easeInOutCubic } from "./format.js";

const DEG2RAD = Math.PI / 180;
const CAMERA_DIR = new THREE.Vector3(0.78, 0.74, 1).normalize();

function makeRadialTexture(size = 512, inner = "rgba(120, 220, 255, 0.9)") {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const center = size / 2;
  const gradient = ctx.createRadialGradient(center, center, 0, center, center, center);
  gradient.addColorStop(0, inner);
  gradient.addColorStop(0.25, "rgba(120, 200, 255, 0.28)");
  gradient.addColorStop(0.6, "rgba(120, 160, 255, 0.06)");
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function makeFloorTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const gradient = ctx.createRadialGradient(512, 256, 8, 512, 256, 500);
  gradient.addColorStop(0, "rgba(56, 189, 248, 0.2)");
  gradient.addColorStop(0.3, "rgba(59, 130, 246, 0.12)");
  gradient.addColorStop(0.58, "rgba(139, 92, 246, 0.06)");
  gradient.addColorStop(0.85, "rgba(76, 29, 149, 0.02)");
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(103, 232, 249, 0.055)";
  ctx.lineWidth = 1.2;
  for (let i = 1; i <= 4; i += 1) {
    ctx.beginPath();
    ctx.ellipse(512, 256, i * 105, i * 52, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class SceneManager {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x050810, 90, 360);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 4000);
    this.camera.position.set(0, 18, 60);

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.13;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.domElement.setAttribute("aria-label", "3D 模型场景");
    this.container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = "label-layer";
    this.labelRenderer.domElement.style.pointerEvents = "none";
    this.labelRenderer.domElement.setAttribute("aria-hidden", "true");
    this.container.appendChild(this.labelRenderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 820;
    this.controls.maxPolarAngle = Math.PI * 0.94;
    this.controls.minPolarAngle = Math.PI * 0.04;
    this.controls.rotateSpeed = 0.7;
    this.controls.zoomSpeed = 0.85;
    this.controls.panSpeed = 0.8;
    this.controls.target.set(0, 0, 0);
    this.userInteracted = false;
    this.controls.addEventListener("start", () => {
      this.userInteracted = true;
      this.onInteractionStart?.();
    });
    this.controls.addEventListener("end", () => {
      this.onCameraSettled?.();
    });

    this.world = new THREE.Group();
    this.world.name = "world";
    this.scene.add(this.world);
    this.effects = new THREE.Group();
    this.effects.name = "effects";
    this.scene.add(this.effects);

    this.content = null;
    this.layout = null;
    this.graph = null;
    this.cameraTween = null;
    this.intro = null;
    this.introToken = 0;
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.frameCallbacks = new Set();
    this.flowPulse = null;
    this.floorY = -4;
    this.generation = 0;

    this._buildEnvironment();
    this._buildFlowParticle();

    this._lastFrameTime = performance.now();
    this._running = true;
    this._animate = this._animate.bind(this);
    this._onResize = this._onResize.bind(this);

    this.resizeObserver = new ResizeObserver(this._onResize);
    this.resizeObserver.observe(this.container);
    this._onResize();

    this.renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.onContextLost?.();
    });

    requestAnimationFrame(this._animate);
  }

  _buildEnvironment() {
    this.env = new THREE.Group();
    this.env.name = "environment";
    this.scene.add(this.env);

    this.hemi = new THREE.HemisphereLight(0x9fc8ff, 0x080e1a, 1.05);
    this.scene.add(this.hemi);

    this.ambient = new THREE.AmbientLight(0x35486a, 0.55);
    this.scene.add(this.ambient);

    this.keyLight = new THREE.DirectionalLight(0xeaf7ff, 1.75);
    this.keyLight.position.set(24, 44, 28);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0004;
    this.keyLight.shadow.normalBias = 0.035;
    this.keyLight.shadow.camera.near = 0.5;
    this.keyLight.shadow.camera.far = 260;
    this.keyLight.target = new THREE.Object3D();
    this.scene.add(this.keyLight.target);
    this.scene.add(this.keyLight);

    this.rimLight = new THREE.DirectionalLight(0x8b5cf6, 1.15);
    this.rimLight.position.set(-40, 18, -26);
    this.scene.add(this.rimLight);

    this.coolLight = new THREE.PointLight(0x22d3ee, 55, 190, 2);
    this.coolLight.position.set(-6, 16, 30);
    this.scene.add(this.coolLight);

    this.warmLight = new THREE.PointLight(0xf472b6, 34, 160, 2);
    this.warmLight.position.set(40, 12, -24);
    this.scene.add(this.warmLight);

    // Shadow catcher
    this.shadowPlaneMaterial = new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.34 });
    this.shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), this.shadowPlaneMaterial);
    this.shadowPlane.rotation.x = -Math.PI / 2;
    this.shadowPlane.receiveShadow = true;
    this.shadowPlane.position.y = -4;
    this.env.add(this.shadowPlane);

    // Grid — soft technical floor.
    this.grid = new THREE.GridHelper(640, 320, 0x1d3b5e, 0x0f1c30);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.62;
    this.grid.material.depthWrite = false;
    this.grid.position.y = -3.99;
    this.env.add(this.grid);

    const glowMaterial = new THREE.MeshBasicMaterial({
      map: makeFloorTexture(),
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.floorGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), glowMaterial);
    this.floorGlow.rotation.x = -Math.PI / 2;
    this.floorGlow.position.y = -3.97;
    this.floorGlow.scale.set(90, 42, 1);
    this.env.add(this.floorGlow);

    // Star dust
    const count = 460;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      positions[i * 3] = (Math.random() - 0.5) * 260;
      positions[i * 3 + 1] = Math.random() * 46 - 3;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 180 - 20;
    }
    const dustGeometry = new THREE.BufferGeometry();
    dustGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    this.dustMaterial = new THREE.PointsMaterial({
      color: 0x9bd9ff,
      size: 0.13,
      map: makeRadialTexture(128, "rgba(190, 240, 255, 0.95)"),
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    this.dust = new THREE.Points(dustGeometry, this.dustMaterial);
    this.scene.add(this.dust);
  }

  _buildFlowParticle() {
    this.flowGroup = new THREE.Group();
    this.flowGroup.name = "flow";
    this.flowGroup.visible = false;
    this.flowSphereMaterial = new THREE.MeshBasicMaterial({
      color: 0xe6feff,
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.flowSphere = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 12), this.flowSphereMaterial);
    this.flowGroup.add(this.flowSphere);

    this.flowSpriteMaterial = new THREE.SpriteMaterial({
      map: makeRadialTexture(256, "rgba(230, 252, 255, 1)"),
      color: 0x67e8f9,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.flowSprite = new THREE.Sprite(this.flowSpriteMaterial);
    this.flowSprite.scale.set(2.6, 2.6, 1);
    this.flowGroup.add(this.flowSprite);
    this.scene.add(this.flowGroup);

    this.flowRingMaterial = new THREE.MeshBasicMaterial({
      color: 0x67e8f9,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.flowRing = new THREE.Mesh(new THREE.RingGeometry(0.86, 1, 72), this.flowRingMaterial);
    this.flowRing.rotation.x = -Math.PI / 2;
    this.flowRing.visible = false;
    this.effects.add(this.flowRing);
  }

  setContent(content, layout, graph) {
    this.clearContent();
    this.content = content;
    this.layout = layout;
    this.graph = graph;
    this.generation += 1;
    this.world.add(content.nodesRoot, content.kernelsRoot, content.groupsRoot, content.edgesRoot);
    this._updateEnvironment(layout);
    this.fit({ instant: true });
    return this;
  }

  _updateEnvironment(layout) {
    const b = layout.bounds;
    const floorY = b.minY - 0.35;
    this.floorY = floorY;
    this.shadowPlane.position.set(b.centerX, floorY, b.centerZ);
    this.grid.position.set(b.centerX, floorY + 0.01, b.centerZ);
    this.floorGlow.position.set(b.centerX, floorY + 0.02, b.centerZ);
    this.floorGlow.scale.set(Math.max(b.spanX * 1.75, 96), Math.max(b.spanX * 0.6, 44), 1);

    const keyTarget = new THREE.Vector3(b.centerX, 0, b.centerZ);
    this.keyLight.target.position.copy(keyTarget);
    this.keyLight.target.updateMatrixWorld();
    this.keyLight.position.set(b.centerX + 30, 46 + b.spanY, 30);
    const halfX = b.spanX / 2 + 16;
    const halfY = Math.max(b.spanY, b.spanZ) / 2 + 20;
    const shadowCam = this.keyLight.shadow.camera;
    shadowCam.left = -halfX;
    shadowCam.right = halfX;
    shadowCam.top = halfY;
    shadowCam.bottom = -halfY;
    shadowCam.near = 0.5;
    shadowCam.far = 280;
    shadowCam.updateProjectionMatrix();
    const mapSize = layout.nodeCount <= 220 ? 1024 : 768;
    if (this.keyLight.shadow.mapSize.x !== mapSize) {
      this.keyLight.shadow.mapSize.set(mapSize, mapSize);
      if (this.keyLight.shadow.map) {
        this.keyLight.shadow.map.dispose();
        this.keyLight.shadow.map = null;
      }
    }
    this.keyLight.shadow.normalBias = layout.big ? 0.06 : 0.035;
    this.rimLight.position.set(b.centerX - 38, 20, b.centerZ - 30);
    this.coolLight.position.set(b.centerX - 10, 16, b.centerZ + 34);
    this.warmLight.position.set(b.centerX + b.spanX * 0.5 + 20, 12, b.centerZ - 24);

    const fogNear = Math.max(70, b.spanX * 1.4 + 30);
    const fogFar = Math.max(240, b.spanX * 4 + 140);
    this.scene.fog.near = fogNear;
    this.scene.fog.far = fogFar;
  }

  _cameraFitPose(layout) {
    const b = layout ? layout.bounds : null;
    const center = new THREE.Vector3(
      b ? b.centerX : 0,
      b ? b.centerY + 0.15 : 0,
      b ? b.centerZ : 0,
    );
    const spanX = b ? Math.max(b.spanX, 4) : 18;
    const spanY = b ? Math.max(b.spanY, 3) : 10;
    const spanZ = b ? Math.max(b.spanZ, 2) : 8;

    // Project the AABB onto the camera's own right / up axes so the fit is
    // correct for a 3/4 view. Without this a rotated row wastes 20-40% of the
    // viewport (the old behaviour made wide models look like a thin strip).
    const dir = CAMERA_DIR.clone().normalize();
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
    const up = new THREE.Vector3().crossVectors(dir, right).normalize();
    const halfW = (spanX * Math.abs(right.x) + spanY * Math.abs(right.y) + spanZ * Math.abs(right.z)) / 2;
    const halfH = (spanX * Math.abs(up.x) + spanY * Math.abs(up.y) + spanZ * Math.abs(up.z)) / 2;
    const halfDepth = (spanX * Math.abs(dir.x) + spanY * Math.abs(dir.y) + spanZ * Math.abs(dir.z)) / 2;

    const fov = this.camera.fov * DEG2RAD;
    const hFov = 2 * Math.atan(Math.tan(fov / 2) * Math.max(this.camera.aspect, 0.4));
    const pad = 1.02;
    const distH = halfW / Math.max(0.1, Math.tan(hFov / 2)) * pad;
    const distV = halfH / Math.max(0.1, Math.tan(fov / 2)) * pad;
    const distance = Math.max(distH, distV, 8) + halfDepth * 0.7;
    const position = center.clone().addScaledVector(dir, distance);
    position.y = Math.max(position.y, this.floorY + 4.5);
    return { position, target: center, distance };
  }

  fit({ instant = false } = {}) {
    if (!this.layout) return;
    const { position, target, distance } = this._cameraFitPose(this.layout);
    this.fitDistance = distance;
    this._moveCamera(position, target, instant || this.reducedMotion ? 0 : 720);
    this.onCameraMove?.(900);
  }

  /** True when the user has zoomed noticeably closer than the auto-fit pose. */
  isLabelZoomedIn() {
    if (!this.fitDistance) return false;
    const distance = this.camera.position.distanceTo(this.controls.target);
    return distance < this.fitDistance * 0.7;
  }

  focusItem(item, { instant = false } = {}) {
    if (!item) return;
    const center = new THREE.Vector3(item.x, 0, 0);
    const extent = Math.max(item.sx, item.sy, item.sz);
    const distance = clamp(extent * 2.9, 5.8, 38);
    const position = center.clone().addScaledVector(CAMERA_DIR, distance);
    position.y = Math.max(position.y, this.floorY + 3.5);
    this._moveCamera(position, center, instant || this.reducedMotion ? 0 : 680);
    this.onCameraMove?.(820);
  }

  _moveCamera(position, target, duration) {
    if (!duration) {
      this.cameraTween = null;
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    this.cameraTween = {
      fromPos: this.camera.position.clone(),
      toPos: position.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: target.clone(),
      start: performance.now(),
      duration,
    };
  }

  showSelection(item) {
    this.hideSelection();
    if (!item) return;
    const geometry = new THREE.BoxGeometry(
      item.sx * 1.07 + 0.05,
      item.sy * 1.07 + 0.05,
      item.sz * 1.07 + 0.05,
    );
    const material = new THREE.LineBasicMaterial({
      color: 0xdff9ff,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    });
    this.selection = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), material);
    this.selection.position.set(item.x, 0, 0);
    this.selection.renderOrder = 20;
    this.effects.add(this.selection);
  }

  hideSelection() {
    if (!this.selection) return;
    this.effects.remove(this.selection);
    this.selection.geometry.dispose();
    this.selection.material.dispose();
    this.selection = null;
  }

  showFlowPoint(position, visible = true) {
    this.flowGroup.visible = Boolean(visible && position);
    if (position) this.flowGroup.position.copy(position);
  }

  showFlowTarget(item) {
    if (!item) {
      this.flowRing.visible = false;
      this.flowPulse = null;
      return;
    }
    const radius = clamp(Math.max(item.sy, item.sz) * 0.72, 0.9, 4);
    this.flowRing.geometry.dispose();
    this.flowRing.geometry = new THREE.RingGeometry(radius * 0.85, radius, 80);
    this.flowRing.position.set(item.x, this.floorY + 0.04, 0);
    this.flowRing.visible = true;
    this.flowPulse = { start: performance.now(), item };
  }

  startIntro() {
    this.introToken += 1;
    if (!this.content || this.reducedMotion || !this.layout || this.layout.huge) {
      this.intro = null;
      return;
    }
    const token = this.introToken;
    const views = this.content.nodeViews;
    for (const view of views) {
      view.group.position.y = 0;
    }
    this.intro = {
      token,
      start: performance.now(),
      views,
    };
  }

  addFrameCallback(callback) {
    this.frameCallbacks.add(callback);
    return () => this.frameCallbacks.delete(callback);
  }

  setReducedMotion(value) {
    this.reducedMotion = Boolean(value);
  }

  _animate() {
    if (!this._running) return;
    requestAnimationFrame(this._animate);
    const now = performance.now();
    const dt = Math.min((now - this._lastFrameTime) / 1000, 0.05);
    this._lastFrameTime = now;

    this.controls.update();

    if (this.cameraTween) {
      const t = clamp((now - this.cameraTween.start) / this.cameraTween.duration, 0, 1);
      const eased = easeInOutCubic(t);
      this.camera.position.lerpVectors(this.cameraTween.fromPos, this.cameraTween.toPos, eased);
      this.controls.target.lerpVectors(this.cameraTween.fromTarget, this.cameraTween.toTarget, eased);
      this.controls.update();
      if (t >= 1) this.cameraTween = null;
    }

    if (this.intro) {
      const elapsed = now - this.intro.start;
      const views = this.intro.views;
      for (let i = 0; i < views.length; i += 1) {
        const delay = Math.min(i * 22, 380);
        const p = clamp((elapsed - delay) / 620, 0, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        views[i].group.position.y = -(1 - eased) * 1.4;
      }
      if (elapsed > 1400) {
        for (const view of views) view.group.position.y = 0;
        this.intro = null;
      }
    }

    if (!this.reducedMotion) {
      this.dust.rotation.y += dt * 0.006;
      this.dust.rotation.x = Math.sin(now * 0.00002) * 0.02;
    }

    if (this.flowPulse) {
      const t = (now - this.flowPulse.start) / 1000;
      this.flowRingMaterial.opacity = 0.24 + Math.sin(t * 3.2) * 0.12;
      this.flowRing.rotation.z = t * 0.18;
    } else {
      this.flowRingMaterial.opacity = 0;
    }

    for (const callback of this.frameCallbacks) {
      try {
        callback(dt, now);
      } catch (error) {
        // Keep one faulty frame callback from breaking the render loop.
      }
    }

    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }

  _onResize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setSize(width, height, false);
    this.renderer.domElement.style.width = "100%";
    this.renderer.domElement.style.height = "100%";
    this.labelRenderer.setSize(width, height);
    if (!this.userInteracted && this.layout) {
      this.fit({ instant: true });
    }
  }

  clearContent() {
    this.introToken += 1;
    this.intro = null;
    this.cameraTween = null;
    this.showFlowPoint(null, false);
    this.showFlowTarget(null);
    this.hideSelection();
    if (!this.content) return;
    const roots = [
      this.content.nodesRoot,
      this.content.kernelsRoot,
      this.content.groupsRoot,
      this.content.edgesRoot,
    ];
    for (const root of roots) {
      if (!root) continue;
      this.world.remove(root);
      disposeObject3D(root);
    }
    this.content = null;
  }

  dispose() {
    this._running = false;
    this.resizeObserver.disconnect();
    this.clearContent();
    this.scene.traverse((object) => {
      if (object.geometry) object.geometry.dispose?.();
      if (object.material) disposeMaterial(object.material);
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }
}

function disposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) {
    material.forEach((m) => m.dispose?.());
    return;
  }
  if (material.map) material.map.dispose?.();
  material.dispose?.();
}

export function disposeObject3D(root) {
  const labels = [];
  root.traverse((object) => {
    if (object.isCSS2DObject) labels.push(object);
    if (object.geometry) object.geometry.dispose?.();
    if (object.material) disposeMaterial(object.material);
  });
  for (const label of labels) {
    if (label.element && label.element.parentNode) label.element.parentNode.removeChild(label.element);
  }
}
