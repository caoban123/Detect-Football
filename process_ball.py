import os
import math
import torch
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm

SOURCE_VIDEO_PATH = r"D:\Detech-football\data\test2.mp4"
MODEL_PATH = r"D:\Detech-football\models\best.pt"
TARGET_VIDEO_PATH = r"D:\Detech-football\outputs\output_ball_processed.mp4"

os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)
print("Model classes:", model.names)

# --- CONFIGURATION ---

# Slicing (chia nhỏ ảnh) để phát hiện bóng siêu nhỏ tốt hơn (yêu cầu cấu hình mạnh/GPU)
USE_SLICING = False

# Độ phân giải đầu vào khi chạy chế độ thường
IMGSZ = 1536 if device == "cuda" else 960

# Ngưỡng phát hiện của YOLO (Đặt rất thấp để tránh bỏ sót bóng, sau đó ta tự lọc bằng heuristic)
YOLO_CONF = 0.02

# Lọc bóng bằng 2 ngưỡng tự tin (Hysteresis Thresholding)
BALL_INIT_CONF = 0.22      # Ngưỡng cao để bắt đầu track bóng (tránh false positive từ vạch kẻ/giày trắng)
BALL_MIN_CONF = 0.05       # Ngưỡng thấp để tiếp tục bám đuổi khi đã có vết bóng

# Giới hạn kích thước bóng trong video 1080p
BALL_MAX_SIZE = 22.0       # Bóng không thể to hơn 22x22 pixel
BALL_MIN_Y_RATIO = 0.08    # Bỏ qua các vật thể phía trên khán đài (8% mép trên)

# Khoảng cách nhảy tối đa giữa 2 frame (pixel)
MAX_BALL_JUMP = 75.0       

# [NEW] Nội suy tuyến tính (Linear Interpolation) cho các khoảng mất dấu ngắn
# Nếu bóng bị mất dưới 15 frames (~0.6 giây), hệ thống sẽ tự động vẽ đường nội suy
MAX_INTERPOLATION_GAP = 15

# Làm mượt vị trí bằng Exponential Moving Average
SMOOTHING_ALPHA = 0.70

# Độ dài của đuôi bóng vẽ trên màn hình
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
    rolling_centroid,
    frame_shape
):
    """
    Chọn ứng cử viên bóng tốt nhất dựa vào:
    1. Kiểm tra kích thước và tỉ lệ khung hình (Aspect Ratio).
    2. Nếu chưa có vết bóng (rolling_centroid = None): Yêu cầu conf >= BALL_INIT_CONF.
    3. Nếu đã có vết bóng: Cho phép conf >= BALL_MIN_CONF, nhưng phải nằm trong bán kính MAX_BALL_JUMP của rolling_centroid.
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

        # Lọc vùng sân (tránh khán đài)
        if center_y < h * BALL_MIN_Y_RATIO:
            continue

        # Lọc kích thước tối đa của bóng
        if box_w > BALL_MAX_SIZE or box_h > BALL_MAX_SIZE:
            continue

        # Lọc tỉ lệ box (bóng phải gần vuông)
        aspect_ratio = box_w / box_h
        if not (0.4 <= aspect_ratio <= 2.2):
            continue

        # Lọc theo ngưỡng confidence thích ứng
        if rolling_centroid is None:
            if conf < BALL_INIT_CONF:
                continue
        else:
            if conf < BALL_MIN_CONF:
                continue

        valid_indices.append(idx)

    if not valid_indices:
        return None, None, None

    centers = np.array([get_box_center(ball_detections.xyxy[i]) for i in valid_indices])
    confidences = np.array([ball_detections.confidence[i] for i in valid_indices])
    boxes = np.array([ball_detections.xyxy[i] for i in valid_indices])

    # Nếu chưa có quỹ đạo trước đó: chọn box có confidence cao nhất
    if rolling_centroid is None:
        best_idx = np.argmax(confidences)
        return centers[best_idx], boxes[best_idx], float(confidences[best_idx])

    # Nếu đã có quỹ đạo: tính khoảng cách từ các ứng cử viên tới rolling_centroid
    rolling_centroid = np.array(rolling_centroid, dtype=np.float32)
    distances = np.linalg.norm(centers - rolling_centroid, axis=1)

    # Loại bỏ các candidate nhảy quá xa quỹ đạo thực tế
    in_range_indices = np.where(distances <= MAX_BALL_JUMP)[0]
    if len(in_range_indices) == 0:
        return None, None, None

    # Chọn match tốt nhất dựa trên điểm số kết hợp (confidence cao và gần centroid quỹ đạo)
    frame_diag = math.sqrt(w * w + h * h)
    best_score = -float('inf')
    best_match_idx = -1

    for idx in in_range_indices:
        dist = distances[idx]
        conf = confidences[idx]
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


# --- CALLBACK CHO INFERENCE SLICER ---
def yolov8_inference_callback(slice_image: np.ndarray) -> sv.Detections:
    result = model(
        slice_image,
        device=device,
        conf=YOLO_CONF,
        imgsz=640,
        verbose=False
    )[0]
    return sv.Detections.from_ultralytics(result)


# Khởi tạo InferenceSlicer
slicer = sv.InferenceSlicer(
    callback=yolov8_inference_callback,
    slice_wh=(640, 640),
    overlap_wh=(128, 128),
    iou_threshold=0.3
)

# Khởi tạo TriangleAnnotator vẽ hình tam giác chỉ vào bóng
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFFF00"), # Tam giác màu vàng sáng
    base=12,
    height=12,
    position=sv.Position.TOP_CENTER
)

BALL_ID = get_class_id_by_name(model.names, "ball")
if BALL_ID is None:
    raise ValueError(f"Không tìm thấy class 'ball'. Các class của model: {model.names}")

print(f"BALL_ID = {BALL_ID}")


# =========================================================================
# [PASS 1]: CHẠY INFERENCE VÀ THEO DÕI BÓNG (LỌC ANOMALY BẰNG ROLLING CENTROID)
# =========================================================================
print("=== [PASS 1]: Running detection and raw ball tracking ===")
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

raw_ball_centers = []     # Lưu tâm bóng phát hiện được ở từng frame (chứa None nếu mất bóng)
raw_ball_confidences = []   # Lưu confidence tương ứng
ball_history = []         # Lưu lịch sử centroid thật để làm rolling centroid

for frame in tqdm(frame_generator, total=video_info.total_frames):
    if USE_SLICING:
        detections = slicer(frame)
    else:
        result = model(
            frame,
            device=device,
            conf=YOLO_CONF,
            iou=0.50,
            imgsz=IMGSZ,
            verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)

    # Tách class ball
    ball_detections = detections[detections.class_id == BALL_ID]

    # Tính rolling centroid từ 5 frames thực tế gần nhất
    real_detections = [p for p in ball_history[-5:] if p is not None]
    if len(real_detections) > 0:
        rolling_centroid = np.mean(real_detections, axis=0)
    else:
        rolling_centroid = None

    # Lựa chọn candidate bóng tốt nhất
    current_center, current_box, current_conf = select_best_ball_candidate(
        ball_detections=ball_detections,
        rolling_centroid=rolling_centroid,
        frame_shape=frame.shape
    )

    # Lưu kết quả
    raw_ball_centers.append(current_center)
    raw_ball_confidences.append(current_conf)
    
    # Cập nhật lịch sử
    ball_history.append(current_center)
    if len(ball_history) > 30:
        ball_history.pop(0)


# =========================================================================
# [PASS 2]: NỘI SUY TUYẾN TÍNH CÁC KHOẢNG MẤT DẤU NGẮN (LINEAR INTERPOLATION)
# =========================================================================
print("\n=== [PASS 2]: Interpolating missing frames ===")
interpolated_ball_centers = list(raw_ball_centers)
n_frames = len(interpolated_ball_centers)

i = 0
interpolated_count = 0
while i < n_frames:
    if interpolated_ball_centers[i] is not None:
        i += 1
        continue
    
    # Phát hiện điểm bắt đầu của một khoảng trống (None)
    start_idx = i - 1
    
    # Tìm điểm kết thúc của khoảng trống
    while i < n_frames and interpolated_ball_centers[i] is None:
        i += 1
    end_idx = i
    
    # Không nội suy nếu khoảng trống nằm ở đầu hoặc cuối video
    if start_idx < 0 or end_idx >= n_frames:
        continue
        
    gap_len = end_idx - start_idx - 1
    # Chỉ nội suy nếu khoảng trống ngắn hơn MAX_INTERPOLATION_GAP
    if gap_len <= MAX_INTERPOLATION_GAP:
        start_val = interpolated_ball_centers[start_idx]
        end_val = interpolated_ball_centers[end_idx]
        for j in range(1, gap_len + 1):
            alpha = j / (gap_len + 1)
            interpolated_ball_centers[start_idx + j] = (1 - alpha) * start_val + alpha * end_val
            interpolated_count += 1

print(f"Interpolated {interpolated_count} missing frames.")


# =========================================================================
# [PASS 3]: LÀM MƯỢT VÀ GHI VIDEO KẾT QUẢ
# =========================================================================
print("\n=== [PASS 3]: Smoothing and rendering output video ===")
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

previous_smoothed_center = None
ball_trail = []

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame_index, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
        
        interpolated_center = interpolated_ball_centers[frame_index]
        raw_center = raw_ball_centers[frame_index]
        raw_conf = raw_ball_confidences[frame_index]

        # Kiểm tra trạng thái bóng ở frame này
        if interpolated_center is not None:
            is_estimated = (raw_center is None) # Là bóng nội suy nếu không có detection thật
            
            # Làm mượt tọa độ
            previous_smoothed_center = smooth_center(interpolated_center, previous_smoothed_center)
            draw_center = previous_smoothed_center
        else:
            is_estimated = False
            previous_smoothed_center = None
            draw_center = None

        # Cập nhật đuôi bóng
        ball_trail.append(draw_center)
        if len(ball_trail) > BALL_TRAIL_LENGTH:
            ball_trail.pop(0)

        # Vẽ lên khung hình
        annotated_frame = frame.copy()

        # A. Vẽ đuôi chuyển động của bóng
        if len(ball_trail) >= 2:
            valid_trail = [p for p in ball_trail if p is not None]
            for i in range(1, len(valid_trail)):
                p1 = tuple(valid_trail[i - 1].astype(int))
                p2 = tuple(valid_trail[i].astype(int))
                alpha = i / len(valid_trail)
                color = (int(0 * alpha), int(255 * alpha), int(255 * alpha))
                cv2.line(annotated_frame, p1, p2, color, 2, cv2.LINE_AA)

        # B. Vẽ bóng (Tam giác chỉ thị + hình tròn tâm)
        if draw_center is not None:
            x, y = draw_center
            
            # Giả lập detections cho TriangleAnnotator
            fake_box = np.array([[x - 5, y - 5, x + 5, y + 5]], dtype=np.float32)
            ball_det = sv.Detections(
                xyxy=fake_box,
                confidence=np.array([raw_conf if raw_conf is not None else 1.0]),
                class_id=np.array([BALL_ID])
            )
            
            # Vẽ hình tam giác chỉ vào bóng ở phía trên
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_det
            )

            # Vẽ tâm bóng
            radius = 5 if not is_estimated else 4
            # Đỏ nếu detect thật, xanh dương nhạt (cyan) nếu là bóng nội suy
            color = (0, 0, 255) if not is_estimated else (255, 255, 0)
            cv2.circle(annotated_frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

            # Ghi text nhãn
            if not is_estimated:
                label = f"ball {raw_conf:.2f}"
            else:
                label = "ball interpolated"
                
            cv2.putText(
                annotated_frame,
                label,
                (int(x) + 12, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )

        sink.write_frame(annotated_frame)

print(f"\nDone. Saved to {TARGET_VIDEO_PATH}")
