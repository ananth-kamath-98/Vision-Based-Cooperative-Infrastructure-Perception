import numpy as np
from . import config


def fuse_detections_across_cameras(detections):
    """
    Fuses detections from multiple cameras. A camera can only contribute
    one detection to a fused object.
    """
    clusters = []
    for det in detections:
        pt = np.array(det["pt"], dtype=np.float32)
        placed = False
        for cluster in clusters:
            # A camera can only contribute one detection to a cluster
            if det["cam"] in {c["cam"] for c in cluster}:
                continue

            pts = np.array([c["pt"] for c in cluster], dtype=np.float32)
            centroid = pts.mean(axis=0)
            if np.linalg.norm(
                    pt - centroid) < config.FUSION_DISTANCE_THRESHOLD:
                cluster.append(det)
                placed = True
                break
        if not placed:
            clusters.append([det])

    fused_results = []
    for cluster in clusters:
        pts = np.array([c["pt"] for c in cluster], dtype=np.float32)
        cx, cy = pts.mean(axis=0).astype(int)
        cams = ",".join(sorted({c["cam"] for c in cluster}))

        # Determine majority direction
        dirs = [c["dir"] for c in cluster if c["dir"]]
        dir_major = max(set(dirs), key=dirs.count) if dirs else None

        fused_results.append({"x": cx, "y": cy, "cam": cams, "dir": dir_major})

    return fused_results
