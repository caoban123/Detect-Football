import os
import math
import torch
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


SOURCE_VIDEO_PATH = r"D:\Detech-football\data\test.mp4"
MODEL_PATH = r"D:\Detech-football\models\best.pt"
TARGET_VIDEO_PATH = r"D:\Detech-football\outputs\output_ball_processed_v2.mp4"

os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Nếu có GPU thì tăng imgsz để bắt bóng tốt hơn.
IMGSZ = 1536 if device == "cuda" else 960

# Đặt conf cực thấp cho model dự đoán để không bỏ sót bóng
YOLO_CONF = 0.02

# Lọc bóng bằng 2 ngưỡng tự tin (Hysteresis Thresholding)
BALL_INIT_CONF = 0.22      # Ngưỡng cao để bắt đầu track bóng (tránh gán nhầm giày/vạch sân)
BALL_MIN_CONF = 0.05       # Ngưỡng thấp để tiếp tục bám theo bóng khi đã có track

# Giới hạn kích thước bóng trong video 1080p
BALL_MAX_SIZE = 22.0       # Bóng không thể to hơn 22x22 pixel
BALL_MIN_Y_RATIO = 0.08    # Bỏ qua các vật thể phía trên khán đài (8% mép trên)

# Giới hạn khoảng cách bóng di chuyển tối đa giữa 2 frame liên tiếp (pixel)
MAX_BALL_JUMP = 70.0       

# Nếu mất bóng, dùng Constant Velocity Model để dự đoán hướng đi tiếp theo
MAX_MISSING_FRAMES_TO_KEEP = 12
VELOCITY_DAMPING = 0.92    # Hệ số giảm ma sát để bóng chậm lại tự nhiên khi mất dấu

# Làm mượt vị trí bằng Exponential Moving Average
SMOOTHING_ALPHA = 0.70

# Vẽ đường đuôi bóng
BALL_TRAIL_LENGTH = 25


def get_class_id_by_name(model_names, target_name: str):
    target_name = target_name.lower().strip()
    if isinstance(model_names, dict):
        for class_id, class_name in model_names.items():
            if str(class_name).lower().strip() == target_name:
                return int(class_id)
    if isinstance(model_names, list):
        for class_id, class_name in enumerate(model_names):
            if str(class_name).lower().strip() == target_name:
                return int(class_id)
    return None


def get_box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def select_best_ball_candidate(
    ball_detections: sv.Detections,
    previous_center,
    frame_shape
):
    """
    Chọn candidate là bóng tốt nhất dựa vào:
    1. Kiểm tra kích thước hộp (max_size) và tỉ lệ khung hình (aspect ratio).
    2. Chỉ cho phép khởi tạo track mới với độ tự tin cao (BALL_INIT_CONF).
    3. Khi đã có track, duy trì với độ tự tin thấp hơn (BALL_MIN_CONF) nhưng khống chế khoảng cách nhảy (MAX_BALL_JUMP).
    """
    if len(ball_detections) == 0:
        return None, None, None

    h, w = frame_shape[:2]
    
    valid_indices = []
    for idx, (xyxy, conf) in enumerate(zip(ball_detections.xyxy, ball_detections.confidence)):
        x1, y1, x2, y2 = xyxy
        box_w = x2 - x1
        box_h = y2 - y1
        center_y = (y1 + y2) / 2

        # Loại bỏ nếu nằm ngoài sân bóng (khán đài)
        if center_y < h * BALL_MIN_Y_RATIO:
            continue

        # Loại bỏ nếu box quá to (không phải bóng)
        if box_w > BALL_MAX_SIZE or box_h > BALL_MAX_SIZE:
            continue

        # Loại bỏ nếu tỉ lệ box bị méo mó (bóng phải tương đối vuông)
        aspect_ratio = box_w / box_h
        if not (0.4 <= aspect_ratio <= 2.2):
            continue

        # Kiểm tra ngưỡng tự tin tương ứng
        if previous_center is None:
            if conf < BALL_INIT_CONF:
                continue
        else:
            if conf < BALL_MIN_CONF:
                continue

        valid_indices.append(idx)

    if not valid_indices:
        return None, None, None

    # Lấy thông tin các candidate hợp lệ
    centers = np.array([get_box_center(ball_detections.xyxy[i]) for i in valid_indices])
    confidences = np.array([ball_detections.confidence[i] for i in valid_indices])
    boxes = np.array([ball_detections.xyxy[i] for i in valid_indices])

    # Nếu chưa có previous_center: chọn candidate có conf cao nhất
    if previous_center is None:
        best_idx = np.argmax(confidences)
        return centers[best_idx], boxes[best_idx], float(confidences[best_idx])

    # Nếu đã có previous_center: tính khoảng cách
    previous_center = np.array(previous_center, dtype=np.float32)
    distances = np.linalg.norm(centers - previous_center, axis=1)

    # Chỉ giữ các candidate nằm trong bán kính cho phép di chuyển (MAX_BALL_JUMP)
    in_range_indices = np.where(distances <= MAX_BALL_JUMP)[0]
    if len(in_range_indices) == 0:
        return None, None, None

    # Tìm candidate tốt nhất dựa trên score kết hợp (ưu tiên conf cao + khoảng cách gần)
    frame_diag = math.sqrt(w * w + h * h)
    best_score = -float('inf')
    best_match_idx = -1

    for idx in in_range_indices:
        dist = distances[idx]
        conf = confidences[idx]
        # Score phạt khoảng cách di chuyển xa
        score = conf - 0.6 * (dist / frame_diag)
        if score > best_score:
            best_score = score
            best_match_idx = idx

    if best_match_idx != -1:
        return centers[best_match_idx], boxes[best_match_idx], float(confidences[best_match_idx])

    return None, None, None


def smooth_center(current_center, previous_smoothed_center):
    if current_center is None:
        return previous_smoothed_center
    if previous_smoothed_center is None:
        return current_center
    return (
        SMOOTHING_ALPHA * current_center
        + (1 - SMOOTHING_ALPHA) * previous_smoothed_center
    )


def draw_ball(
    frame,
    center,
    confidence=None,
    is_estimated=False,
    trail_points=None
):
    annotated = frame.copy()

    # Vẽ trail chuyển động bóng
    if trail_points is not None and len(trail_points) >= 2:
        valid_points = [p for p in trail_points if p is not None]
        for i in range(1, len(valid_points)):
            p1 = tuple(valid_points[i - 1].astype(int))
            p2 = tuple(valid_points[i].astype(int))
            # Vẽ nét mờ dần về phía đuôi
            alpha = i / len(valid_points)
            color = (int(0 * alpha), int(255 * alpha), int(255 * alpha)) # Màu vàng/cyan sáng
            cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)

    # Vẽ vòng tròn bóng
    if center is not None:
        x, y = center.astype(int)
        radius = 7 if not is_estimated else 5
        color = (0, 0, 255) if not is_estimated else (0, 165, 255) # Đỏ nếu detect được, Cam nếu dự đoán

        # Vẽ tâm và vòng ngoài
        cv2.circle(annotated, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), radius + 4, color, 1, cv2.LINE_AA)

        if confidence is not None:
            label = f"ball {confidence:.2f}"
        else:
            label = "ball predicted"

        cv2.putText(
            annotated,
            label,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated


BALL_ID = get_class_id_by_name(model.names, "ball")
if BALL_ID is None:
    raise ValueError(f"Không tìm thấy class 'ball'. Các class của model: {model.names}")

print(f"BALL_ID = {BALL_ID}")

previous_raw_center = None
previous_smoothed_center = None
velocity = np.zeros(2, dtype=np.float32)
missing_frames = 0
ball_trail = []

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame_index, frame in enumerate(frame_generator, start=1):
        result = model(
            frame,
            device=device,
            conf=YOLO_CONF,
            iou=0.50,
            imgsz=IMGSZ,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        # Lọc class ball
        ball_detections = detections[detections.class_id == BALL_ID]

        # Tìm candidate tốt nhất
        current_center, current_box, current_conf = select_best_ball_candidate(
            ball_detections=ball_detections,
            previous_center=previous_raw_center,
            frame_shape=frame.shape
        )

        if current_center is not None:
            # Nếu có detection thật: tính vector vận tốc mới
            if previous_smoothed_center is not None:
                velocity = current_center - previous_smoothed_center
            else:
                velocity = np.zeros(2, dtype=np.float32)

            previous_raw_center = current_center
            previous_smoothed_center = smooth_center(current_center, previous_smoothed_center)
            missing_frames = 0
            is_estimated = False
            draw_center = previous_smoothed_center
            draw_conf = current_conf
        else:
            missing_frames += 1

            # Nếu mất bóng nhưng chưa quá giới hạn: Dự đoán vị trí bằng Constant Velocity Model
            if previous_smoothed_center is not None and missing_frames <= MAX_MISSING_FRAMES_TO_KEEP:
                # Vị trí mới = Vị trí cũ + Vận tốc * Độ suy giảm
                predicted_center = previous_smoothed_center + velocity * VELOCITY_DAMPING
                
                # Giảm dần vận tốc do ma sát không khí/mặt sân
                velocity = velocity * VELOCITY_DAMPING
                
                previous_smoothed_center = predicted_center
                draw_center = predicted_center
                draw_conf = None
                is_estimated = True
            else:
                # Quá số frame mất bóng -> Hủy vết, đợi kích hoạt lại
                previous_raw_center = None
                previous_smoothed_center = None
                velocity = np.zeros(2, dtype=np.float32)
                draw_center = None
                draw_conf = None
                is_estimated = False

        ball_trail.append(draw_center)
        if len(ball_trail) > BALL_TRAIL_LENGTH:
            ball_trail.pop(0)

        annotated_frame = draw_ball(
            frame=frame,
            center=draw_center,
            confidence=draw_conf,
            is_estimated=is_estimated,
            trail_points=ball_trail
        )

        sink.write_frame(annotated_frame)

        if frame_index % 50 == 0:
            status = "detected" if current_center is not None else ("predicted" if draw_center is not None else "missing")
            print(
                f"Frame {frame_index}: ball_status={status}, "
                f"missing_frames={missing_frames}, "
                f"vel_mag={np.linalg.norm(velocity):.2f}"
            )

print(f"Done. Saved to {TARGET_VIDEO_PATH}")
