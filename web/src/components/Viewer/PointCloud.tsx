import { useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useJobStore } from "../../state/jobStore";
import type { ColorMode } from "../../types";

const COLOR_MODE_ENUM: Record<ColorMode, number> = {
  rgb: 0,
  frame: 1,
  confidence: 2,
};

// Categorical palette for "color by frame".
function framePalette(maxFrames: number): THREE.Color[] {
  const out: THREE.Color[] = [];
  for (let i = 0; i < Math.max(1, maxFrames); i++) {
    out.push(new THREE.Color().setHSL((i * 0.618) % 1, 0.6, 0.55));
  }
  return out;
}

const vertexShader = /* glsl */ `
  attribute vec3 color;
  attribute float conf;
  attribute float frameIdx;
  uniform float uPointSize;
  uniform float uConfThreshold;
  uniform int uColorMode;
  uniform vec3 uFramePalette[256];
  varying vec3 vColor;
  varying float vVisible;

  vec3 viridis(float t) {
    // cheap viridis approximation
    return clamp(vec3(
      0.28 + t * (0.1 + t * 1.0),
      0.0 + t * 0.9,
      0.33 + t * (0.6 - t * 0.5)
    ), 0.0, 1.0);
  }

  void main() {
    vVisible = conf >= uConfThreshold ? 1.0 : 0.0;
    if (uColorMode == 0) {
      vColor = color;
    } else if (uColorMode == 1) {
      int idx = int(mod(frameIdx, 256.0));
      vColor = uFramePalette[idx];
    } else {
      vColor = viridis(conf);
    }
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // Perspective size attenuation, but clamp the distance factor so points
    // very close to the camera (small -mv.z) don't blow up to fill the screen.
    float dist = max(-mv.z, 0.1);
    float atten = clamp(8.0 / dist, 0.5, 6.0);
    gl_PointSize = vVisible * uPointSize * atten;
  }
`;

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vVisible;
  void main() {
    if (vVisible < 0.5) discard;
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard; // round points
    gl_FragColor = vec4(vColor, 1.0);
  }
`;

export function PointCloud() {
  const points = useJobStore((s) => s.points);
  const view = useJobStore((s) => s.view);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  const geometry = useMemo(() => {
    if (!points) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
    // normalize uint8 color -> 0..1 in-shader is avoided; convert once here.
    const colorF = new Float32Array(points.count * 3);
    for (let i = 0; i < colorF.length; i++) colorF[i] = points.colors[i] / 255;
    g.setAttribute("color", new THREE.BufferAttribute(colorF, 3));
    g.setAttribute("conf", new THREE.BufferAttribute(points.conf, 1));
    const frameF = new Float32Array(points.count);
    for (let i = 0; i < points.count; i++) frameF[i] = points.frameIdx[i];
    g.setAttribute("frameIdx", new THREE.BufferAttribute(frameF, 1));
    g.computeBoundingSphere();
    return g;
  }, [points]);

  const uniforms = useMemo(() => {
    const palette = framePalette(256).map((c) => new THREE.Vector3(c.r, c.g, c.b));
    return {
      uPointSize: { value: view.pointSize },
      uConfThreshold: { value: view.confThreshold },
      uColorMode: { value: COLOR_MODE_ENUM[view.colorMode] },
      uFramePalette: { value: palette },
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useFrame(() => {
    const m = materialRef.current;
    if (!m) return;
    m.uniforms.uPointSize.value = view.pointSize;
    m.uniforms.uConfThreshold.value = view.confThreshold;
    m.uniforms.uColorMode.value = COLOR_MODE_ENUM[view.colorMode];
  });

  if (!geometry || !points || !view.showPoints) return null;

  return (
    <points geometry={geometry}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
      />
    </points>
  );
}
