import numpy as np
import supervision as sv

def filter_detections(detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
    """
    Filter out noisy bounding boxes (too small, too large, or outside the field region).
    """
    if len(detections) == 0:
        return detections

    h, w = frame.shape[:2]
    xyxy = detections.xyxy

    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h
    center_y = (y1 + y2) / 2

    # Exclude boxes that are too small (noise)
    min_area = 40

    # Exclude boxes that are too large (spectators close to camera, banners)
    max_area = h * w * 0.08

    # Field region mask: exclude the top region (usually stands/crowd)
    # Adjust 0.10 if the pitch goes closer to the top border.
    field_region_mask = center_y > h * 0.10

    size_mask = (box_area > min_area) & (box_area < max_area)

    return detections[field_region_mask & size_mask]
