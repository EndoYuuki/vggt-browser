import { useEffect } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { useJobStore } from "../../state/jobStore";

/** Frames the camera to the point cloud whenever new points load, so the view
 * is sensible regardless of the scene's coordinate scale. */
export function FitCamera() {
  const points = useJobStore((s) => s.points);
  const { camera, controls } = useThree();

  useEffect(() => {
    if (!points || points.count === 0) return;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    // Sample for speed on huge clouds.
    const stride = Math.max(1, Math.floor(points.count / 50000));
    for (let i = 0; i < points.count; i += stride) {
      v.set(
        points.positions[i * 3],
        points.positions[i * 3 + 1],
        points.positions[i * 3 + 2],
      );
      box.expandByPoint(v);
    }
    const center = box.getCenter(new THREE.Vector3());
    // Match the scene group's OpenCV->OpenGL flip (scale [1,-1,-1]).
    center.multiply(new THREE.Vector3(1, -1, -1));
    const size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
    const dist = radius * 3;

    const persp = camera as THREE.PerspectiveCamera;
    persp.near = Math.max(radius / 100, 0.001);
    persp.far = radius * 100;
    // Place camera offset along +Z and slightly up for a natural 3/4 view.
    persp.position.set(center.x, center.y + radius * 0.5, center.z + dist);
    persp.updateProjectionMatrix();

    const oc = controls as unknown as OrbitControlsImpl | undefined;
    if (oc && "target" in oc) {
      oc.target.copy(center);
      oc.update();
    } else {
      persp.lookAt(center);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points]);

  return null;
}
