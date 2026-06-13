import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { PointCloud } from "./PointCloud";
import { CameraFrustums } from "./CameraFrustums";
import { FitCamera } from "./FitCamera";

export function Viewer() {
  return (
    <Canvas
      camera={{ position: [0, 0, -3], fov: 50, near: 0.01, far: 1000 }}
      style={{ background: "#0e0f13" }}
    >
      <ambientLight intensity={0.8} />
      {/* VGGT uses OpenCV convention (+Y down, +Z forward). Flip Y and Z so the
          scene's up matches three.js (+Y up), avoiding OrbitControls pole lock. */}
      <group scale={[1, -1, -1]}>
        <PointCloud />
        <CameraFrustums />
      </group>
      <FitCamera />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport labelColor="white" axisHeadScale={1} />
      </GizmoHelper>
    </Canvas>
  );
}
