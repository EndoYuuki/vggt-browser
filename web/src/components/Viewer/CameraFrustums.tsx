import { useMemo } from "react";
import * as THREE from "three";
import { useJobStore } from "../../state/jobStore";
import type { CameraInfo } from "../../types";

const SCALE = 0.15; // frustum size in world units

// extrinsic is 3x4 world->cam [R|t]. Camera center in world = -R^T t.
// cam->world rotation = R^T.
function camToWorld(cam: CameraInfo): THREE.Matrix4 {
  const e = cam.extrinsic;
  const R = new THREE.Matrix3().set(
    e[0][0], e[0][1], e[0][2],
    e[1][0], e[1][1], e[1][2],
    e[2][0], e[2][1], e[2][2],
  );
  const t = new THREE.Vector3(e[0][3], e[1][3], e[2][3]);
  const Rt = R.clone().transpose();
  const center = t.clone().applyMatrix3(Rt).multiplyScalar(-1);
  const m = new THREE.Matrix4();
  const el = Rt.elements; // column-major 3x3
  m.set(
    el[0], el[3], el[6], center.x,
    el[1], el[4], el[7], center.y,
    el[2], el[5], el[8], center.z,
    0, 0, 0, 1,
  );
  return m;
}

function frustumGeometry(): THREE.BufferGeometry {
  // pyramid: apex at origin (camera center), base square in +z (OpenCV looks +z)
  const a = SCALE;
  const z = SCALE * 1.5;
  const corners = [
    new THREE.Vector3(-a, -a, z),
    new THREE.Vector3(a, -a, z),
    new THREE.Vector3(a, a, z),
    new THREE.Vector3(-a, a, z),
  ];
  const apex = new THREE.Vector3(0, 0, 0);
  const pts: THREE.Vector3[] = [];
  for (const c of corners) {
    pts.push(apex, c);
  }
  for (let i = 0; i < 4; i++) {
    pts.push(corners[i], corners[(i + 1) % 4]);
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
}

export function CameraFrustums() {
  const cameras = useJobStore((s) => s.cameras);
  const view = useJobStore((s) => s.view);
  const setView = useJobStore((s) => s.setView);

  const geo = useMemo(() => frustumGeometry(), []);
  const matrices = useMemo(
    () => (cameras ? cameras.cameras.map(camToWorld) : []),
    [cameras],
  );

  if (!cameras || !view.showCameras) return null;

  return (
    <group>
      {matrices.map((m, i) => {
        const selected = view.selectedFrame === i;
        return (
          <lineSegments
            key={i}
            geometry={geo}
            matrixAutoUpdate={false}
            matrix={m}
            onClick={(e) => {
              e.stopPropagation();
              setView({ selectedFrame: selected ? null : i });
            }}
          >
            <lineBasicMaterial color={selected ? "#ffcc00" : "#33aaff"} />
          </lineSegments>
        );
      })}
    </group>
  );
}
