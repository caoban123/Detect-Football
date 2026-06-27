import os
import math
import json
import torch
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from src.detect_football import FootballDetector
from src.utils import filter_detections
from src.track_objects import FootballTracker
from src.team_assignment import (
    TeamClassifier,
    get_torso_crop,
    is_valid_player_crop,
    assign_goalkeeper_team
)

# Cấu hình đường dẫn video và mô hình
SOURCE_VIDEO_PATH = r"D:\Detech-football\data\test2.mp4"
TARGET_VIDEO_PATH = r"D:\Detech-football\outputs\output_tracked.mp4"
MODEL_PATH = r"D:\Detech-football\models\best.pt"

# --- CONFIGURATION BÓNG ---
YOLO_CONF_BALL = 0.02       # Ngưỡng thấp để tránh bỏ sót bóng
BALL_INIT_CONF = 0.22       # Ngưỡng bắt đầu bám đuổi bóng
BALL_MIN_CONF = 0.05        # Ngưỡng duy trì vết bóng
MAX_INTERPOLATION_GAP = 15  # Số frame mất bóng tối đa để nội suy (~0.6s)
SMOOTHING_ALPHA = 0.70      # Hệ số làm mượt Exponential Moving Average (EMA)
BALL_TRAIL_LENGTH = 25      # Chiều dài của dung sai đuôi bóng vẽ trên sân

# --- CONFIGURATION PHÂN ĐỘI ---
MIN_TEAM_CROPS = 80         # Số lượng crop áo đấu tối thiểu để fit KMeans

def get_box_bottom_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)

def get_box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

def draw_text_label(frame, xyxy, text, color):
    """
    Vẽ nhãn text nhỏ gọn, trong suốt và sắc nét trực tiếp phía trên bounding box.
    Sử dụng hiệu ứng viền đen để không bị che khuất trên nền cỏ hoặc áo đấu.
    """
    x1, y1, x2, y2 = xyxy
    (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cx = int((x1 + x2) / 2 - w_text / 2)
    cy = int(y1 - 6)
    
    # Đảm bảo chữ nằm hoàn toàn trong khung hình
    cy = max(cy, h_text + 5)
    cx = max(0, min(cx, frame.shape[1] - w_text))
    
    # Vẽ viền đen (shadow) bao quanh để chữ nổi bật
    cv2.putText(frame, text, (cx - 1, cy + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    # Vẽ chữ màu chính tương ứng với đội bóng
    cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def select_best_ball_candidate(ball_detections, rolling_centroid, frame_shape, max_ball_size, max_ball_jump):
    """
    Chọn ứng cử viên bóng tốt nhất dựa vào:
    1. Kích thước thích ứng và tỉ lệ khung hình (Aspect Ratio).
    2. Khoảng cách địa lý thích ứng tới rolling_centroid.
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

        # Bảo vệ: Tránh trường hợp chia cho 0 hoặc box quá nhỏ/âm
        if box_h <= 1.0 or box_w <= 1.0:
            continue

        # Lọc vùng khán đài (8% mép trên)
        if center_y < h * 0.08:
            continue

        # Lọc kích thước bóng dựa theo độ phân giải ảnh thích ứng
        if box_w > max_ball_size or box_h > max_ball_size:
            continue

        # Lọc tỉ lệ box bóng
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

    if rolling_centroid is None:
        best_idx = np.argmax(confidences)
        return centers[best_idx], boxes[best_idx], float(confidences[best_idx])

    rolling_centroid = np.array(rolling_centroid, dtype=np.float32)
    distances = np.linalg.norm(centers - rolling_centroid, axis=1)

    # Lọc khoảng nhảy thích ứng
    in_range_indices = np.where(distances <= max_ball_jump)[0]
    if len(in_range_indices) == 0:
        return None, None, None

    # Chọn match tốt nhất dựa trên điểm số kết hợp (confidence cao và gần centroid)
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

def main():
    os.makedirs(os.path.dirname(TARGET_VIDEO_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Khởi tạo detector
    detector = FootballDetector(model_path=MODEL_PATH, device=device)
    
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video Info: {video_info}")

    # Tính toán các ngưỡng thích ứng theo kích thước ảnh
    h, w = video_info.height, video_info.width
    frame_diag = math.sqrt(w * w + h * h)
    max_ball_jump = frame_diag * 0.035   # Khoảng cách bóng nhảy tối đa thích ứng
    max_ball_size = h * 0.021            # Kích thước bóng tối đa thích ứng
    possession_dist_threshold = frame_diag * 0.03  # Ngưỡng khoảng cách sở hữu bóng thích ứng

    # Đặt tên file cache động theo tên video để tránh lẫn lộn màu áo giữa các trận
    video_basename = os.path.splitext(os.path.basename(SOURCE_VIDEO_PATH))[0]
    classifier_cache_path = os.path.join(
        os.path.dirname(TARGET_VIDEO_PATH),
        f"team_classifier_{video_basename}.pkl"
    )

    # 2. Khởi tạo tracker
    tracker = FootballTracker(
        fps=video_info.fps,
        lost_track_buffer=60,
        track_activation_threshold=0.15,
        high_conf_det_threshold=0.45,
        minimum_iou_threshold=0.20,
        minimum_consecutive_frames=2
    )
    
    # 3. Khởi tạo TeamClassifier (SigLIP + UMAP + KMeans)
    team_classifier = TeamClassifier(device=device)

    # Điều chỉnh độ phân giải khi chạy YOLO tùy thuộc vào GPU/CPU
    imgsz = 1536 if device == "cuda" else 960

    # Khôi phục cache của team classifier nếu đúng trận
    has_classifier_cache = team_classifier.load(classifier_cache_path)
    if has_classifier_cache:
        print(f"Loaded fitted TeamClassifier from cache: {classifier_cache_path}")
    else:
        print("TeamClassifier cache not found. Will fit during Pass 1.")

    # =========================================================================
    # [PASS 1]: YOLO INFERENCE, BALL TRAJECTORY & PLAYER CROP COLLECTION
    # =========================================================================
    print("\n=== [PASS 1]: YOLO Inference, Ball Trajectory & Player Crop Collection ===")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    
    # Caching toàn bộ detections của các frame để tránh chạy YOLO 2 lần
    cached_detections = []
    
    raw_ball_centers = []       # Tâm bóng phát hiện được ở từng frame (chứa None nếu mất bóng)
    raw_ball_confidences = []     # Confidence tương ứng của bóng
    ball_history = []           # Lịch sử centroid thật để làm rolling centroid
    collected_player_crops = [] # Thu thập torso player để fit phân đội

    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing frames")):
        # Chạy YOLO với conf thấp để bắt bóng nhỏ nhạy nhất
        detections = detector.detect(frame, conf=YOLO_CONF_BALL, imgsz=imgsz)
        cached_detections.append(detections)
        
        # --- A. LỌC VÀ CHỌN BÓNG TỐT NHẤT ---
        if detector.ball_id is not None:
            ball_dets = detections[detections.class_id == detector.ball_id]
        else:
            ball_dets = sv.Detections.empty()

        # Tính rolling centroid từ tối đa 5 frame thật gần nhất
        real_detections = [p for p in ball_history[-5:] if p is not None]
        rolling_centroid = np.mean(real_detections, axis=0) if len(real_detections) > 0 else None

        current_center, _, current_conf = select_best_ball_candidate(
            ball_detections=ball_dets,
            rolling_centroid=rolling_centroid,
            frame_shape=frame.shape,
            max_ball_size=max_ball_size,
            max_ball_jump=max_ball_jump
        )

        raw_ball_centers.append(current_center)
        raw_ball_confidences.append(current_conf)
        ball_history.append(current_center)
        if len(ball_history) > 30:
            ball_history.pop(0)

        # --- B. THU THẬP CROPS HỌC ÁO ĐẤU (NẾU CHƯA CÓ CACHE) ---
        if not has_classifier_cache and (frame_idx % 15 == 0) and (len(collected_player_crops) < 300):
            player_dets = detections[detections.class_id == detector.player_id]
            for xyxy, conf in zip(player_dets.xyxy, player_dets.confidence):
                if is_valid_player_crop(xyxy, frame.shape, conf, min_conf=0.35):
                    crop = get_torso_crop(frame, xyxy)
                    collected_player_crops.append(crop)

    # =========================================================================
    # [PASS 2]: TEAM CLASSIFIER FITTING & BALL LINEAR INTERPOLATION
    # =========================================================================
    print("\n=== [PASS 2]: Team Classifier Fitting & Ball Linear Interpolation ===")
    
    # 2.1 Huấn luyện Team Classifier nếu chưa có cache và đủ crops
    if not has_classifier_cache:
        if len(collected_player_crops) >= MIN_TEAM_CROPS:
            team_classifier.fit(collected_player_crops)
            team_classifier.save(classifier_cache_path)
        else:
            print(
                f"WARNING: Only {len(collected_player_crops)} valid player crops found. "
                f"Need at least {MIN_TEAM_CROPS} to fit. Team classification is disabled (fallback to Team 0)."
            )

    # 2.2 Nội suy tuyến tính quỹ đạo bóng
    interpolated_ball_centers = list(raw_ball_centers)
    n_frames = len(interpolated_ball_centers)
    interpolated_count = 0
    
    i = 0
    while i < n_frames:
        if interpolated_ball_centers[i] is not None:
            i += 1
            continue
        
        start_idx = i - 1
        while i < n_frames and interpolated_ball_centers[i] is None:
            i += 1
        end_idx = i
        
        if start_idx < 0 or end_idx >= n_frames:
            continue
            
        gap_len = end_idx - start_idx - 1
        if gap_len <= MAX_INTERPOLATION_GAP:
            start_val = interpolated_ball_centers[start_idx]
            end_val = interpolated_ball_centers[end_idx]
            for j in range(1, gap_len + 1):
                alpha = j / (gap_len + 1)
                interpolated_ball_centers[start_idx + j] = (1 - alpha) * start_val + alpha * end_val
                interpolated_count += 1

    print(f"Interpolated {interpolated_count} missing ball frames.")

    # =========================================================================
    # [PASS 3]: OBJECT TRACKING, TEAM ASSIGNMENT & RENDERING VIDEO
    # =========================================================================
    print("\n=== [PASS 3]: Object Tracking, Team Assignment & Rendering Video ===")
    frame_generator_render = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    # Định nghĩa các màu sắc chỉ báo trực quan
    COLOR_TEAM_0 = sv.Color.from_hex("#e6194B")       # Đỏ
    COLOR_TEAM_1 = sv.Color.from_hex("#4363d8")       # Xanh dương
    COLOR_GOALKEEPER = sv.Color.from_hex("#3cb44b")   # Xanh lá
    COLOR_REFEREE = sv.Color.from_hex("#ffe119")      # Vàng sáng

    # Khởi tạo các annotator riêng biệt cho từng đội
    ellipse_team_0 = sv.EllipseAnnotator(color=COLOR_TEAM_0, thickness=2)
    ellipse_team_1 = sv.EllipseAnnotator(color=COLOR_TEAM_1, thickness=2)
    ellipse_gk = sv.EllipseAnnotator(color=COLOR_GOALKEEPER, thickness=2)
    ellipse_ref = sv.EllipseAnnotator(color=COLOR_REFEREE, thickness=2)
    
    # Annotator đặc trị cho quả bóng (vẽ hình tam giác chỉ thị)
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFFF00"),
        base=12,
        height=12,
        position=sv.Position.TOP_CENTER
    )

    previous_smoothed_center = None
    ball_trail = []

    # Quản lý Possession (Kiểm soát bóng)
    team_0_possession_frames = 0
    team_1_possession_frames = 0
    last_possession_team = None  # 0 hoặc 1 (Lưu giữ đội chạm bóng cuối cùng khi bóng bay tự do)

    # Danh sách lưu trữ dữ liệu chi tiết per-frame phục vụ phân tích
    possession_frames = []

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(frame_generator_render, total=video_info.total_frames, desc="Rendering frames")):
            
            # Lấy detections đã cache từ Pass 1
            detections = cached_detections[frame_idx]

            # --- A. XỬ LÝ THEO DÕI NGƯỜI (PLAYERS, GK, REFEREE) ---
            # Chỉ lấy các detections có conf >= 0.08 để đưa vào tracker (tránh nhiễu)
            person_mask = np.isin(detections.class_id, detector.tracked_class_ids) & (detections.confidence >= 0.08)
            person_detections = detections[person_mask]

            # Lọc nhiễu vùng ngoài sân đấu
            person_detections = filter_detections(person_detections, frame)

            # Cập nhật ByteTrack để gán ID
            tracked_person_detections = tracker.update(person_detections)

            # Team 0
            xyxy_t0, conf_t0, class_t0, track_t0 = [], [], [], []
            # Team 1
            xyxy_t1, conf_t1, class_t1, track_t1 = [], [], [], []
            # Referee
            xyxy_ref, conf_ref, class_ref, track_ref = [], [], [], []
            # Goalkeeper
            xyxy_gk, conf_gk, class_gk, track_gk = [], [], [], []

            team_0_feet_positions = []
            team_1_feet_positions = []
            gk_pending_list = []

            # Lưu trữ thông tin chi tiết của người chơi phục vụ xuất JSON
            players_data_list = []
            gks_data_list = []

            # Gom tất cả player crops trong frame này để dự đoán theo batch (SigLIP batch inference)
            player_torsos = []
            player_mapping_indices = [] # Lưu index trong danh sách duyệt tracked detections

            for tracked_idx, (xyxy, confidence, class_id) in enumerate(zip(
                tracked_person_detections.xyxy,
                tracked_person_detections.confidence,
                tracked_person_detections.class_id
            )):
                if class_id == detector.player_id:
                    torso = get_torso_crop(frame, xyxy)
                    player_torsos.append(torso)
                    player_mapping_indices.append(tracked_idx)

            # Dự đoán team theo batch
            if len(player_torsos) > 0 and team_classifier.is_fitted():
                predicted_teams = team_classifier.predict(player_torsos)
            else:
                predicted_teams = np.zeros(len(player_torsos), dtype=np.int32)

            # Tạo map: tracked_idx -> team_id
            player_team_map = {
                player_mapping_indices[k]: predicted_teams[k]
                for k in range(len(player_torsos))
            }

            # Phân bổ đối tượng vào từng nhóm
            for tracked_idx, (xyxy, confidence, class_id, tracker_id) in enumerate(zip(
                tracked_person_detections.xyxy,
                tracked_person_detections.confidence,
                tracked_person_detections.class_id,
                tracked_person_detections.tracker_id
            )):
                feet_pos = get_box_bottom_center(xyxy)

                if class_id == detector.player_id:
                    team_id = player_team_map.get(tracked_idx, 0)
                    
                    # Lưu trữ thông tin cho JSON
                    players_data_list.append({
                        "tracker_id": int(tracker_id),
                        "team_id": int(team_id),
                        "feet_pos": feet_pos.tolist(),
                        "confidence": float(confidence)
                    })

                    if team_id == 0:
                        xyxy_t0.append(xyxy)
                        conf_t0.append(confidence)
                        class_t0.append(class_id)
                        track_t0.append(tracker_id)
                        team_0_feet_positions.append(feet_pos)
                    else:
                        xyxy_t1.append(xyxy)
                        conf_t1.append(confidence)
                        class_t1.append(class_id)
                        track_t1.append(tracker_id)
                        team_1_feet_positions.append(feet_pos)

                elif class_id == detector.referee_id:
                    xyxy_ref.append(xyxy)
                    conf_ref.append(confidence)
                    class_ref.append(class_id)
                    track_ref.append(tracker_id)

                elif class_id == detector.goalkeeper_id:
                    gk_pending_list.append((xyxy, confidence, class_id, tracker_id, feet_pos))

            # Gán goalkeeper vào team tương ứng dựa trên centroid đội hình và đưa vào danh sách feet tương ứng
            for xyxy, confidence, class_id, tracker_id, feet_pos in gk_pending_list:
                gk_team_id = assign_goalkeeper_team(feet_pos, team_0_feet_positions, team_1_feet_positions)
                
                xyxy_gk.append(xyxy)
                conf_gk.append(confidence)
                class_gk.append(class_id)
                track_gk.append(tracker_id)

                gks_data_list.append({
                    "tracker_id": int(tracker_id),
                    "team_id": int(gk_team_id),
                    "feet_pos": feet_pos.tolist(),
                    "confidence": float(confidence)
                })

                # Gộp thủ môn vào danh sách feet của đội đó để tính kiểm soát bóng
                if gk_team_id == 0:
                    team_0_feet_positions.append(feet_pos)
                else:
                    team_1_feet_positions.append(feet_pos)

            # --- B. XỬ LÝ LÀM MƯỢT VÀ QUỸ ĐẠO BÓNG ---
            interpolated_center = interpolated_ball_centers[frame_idx]
            raw_center = raw_ball_centers[frame_idx]
            raw_conf = raw_ball_confidences[frame_idx]

            if interpolated_center is not None:
                is_estimated = (raw_center is None) # Bóng nội suy nếu detection gốc là None
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

            # --- C. TÍNH TOÁN KIỂM SOÁT BÓNG (BALL POSSESSION) ---
            if draw_center is not None:
                min_dist = float('inf')
                closest_team = None

                # Quét khoảng cách từ bóng đến các cầu thủ + thủ môn của Team 0
                for feet in team_0_feet_positions:
                    dist = np.linalg.norm(draw_center - feet)
                    if dist < min_dist:
                        min_dist = dist
                        closest_team = 0

                # Quét khoảng cách đến Team 1
                for feet in team_1_feet_positions:
                    dist = np.linalg.norm(draw_center - feet)
                    if dist < min_dist:
                        min_dist = dist
                        closest_team = 1

                # Nếu khoảng cách gần nhất nhỏ hơn ngưỡng thích ứng, gán quyền sở hữu
                if min_dist < possession_dist_threshold:
                    last_possession_team = closest_team

            # Tăng số lượng frame sở hữu của đội tương ứng
            if last_possession_team == 0:
                team_0_possession_frames += 1
            elif last_possession_team == 1:
                team_1_possession_frames += 1

            # Tính tỷ lệ phần trăm kiểm soát bóng
            total_poss = team_0_possession_frames + team_1_possession_frames
            if total_poss > 0:
                team_0_pct = int(round((team_0_possession_frames / total_poss) * 100))
                team_1_pct = 100 - team_0_pct
            else:
                team_0_pct, team_1_pct = 50, 50

            # Lưu lại dữ liệu phân tích chi tiết của frame này
            possession_frames.append({
                "frame_idx": int(frame_idx),
                "ball_center": draw_center.tolist() if draw_center is not None else None,
                "ball_detected": raw_center is not None,
                "ball_interpolated": is_estimated,
                "players": players_data_list,
                "goalkeepers": gks_data_list,
                "possession_team": last_possession_team
            })

            annotated_frame = frame.copy()

            # 1. Vẽ đuôi chuyển động của bóng (Cyan chuyển màu mượt, cũ -> mới đậm dần)
            if len(ball_trail) >= 2:
                valid_trail = [p for p in ball_trail if p is not None]
                for k in range(1, len(valid_trail)):
                    p1 = tuple(valid_trail[k - 1].astype(int))
                    p2 = tuple(valid_trail[k].astype(int))
                    alpha = k / len(valid_trail)
                    color_trail = (int(255 * alpha), int(255 * alpha), 0) # Màu Cyan (BGR)
                    cv2.line(annotated_frame, p1, p2, color_trail, 2, cv2.LINE_AA)

            # 2. Vẽ Team 0 (Đỏ) bằng annotator và nhãn trong suốt của riêng nó
            if len(xyxy_t0) > 0:
                det_t0 = sv.Detections(
                    xyxy=np.array(xyxy_t0, dtype=np.float32),
                    confidence=np.array(conf_t0, dtype=np.float32),
                    class_id=np.array(class_t0, dtype=np.int32),
                    tracker_id=np.array(track_t0, dtype=np.int32)
                )
                annotated_frame = ellipse_team_0.annotate(scene=annotated_frame, detections=det_t0)
                for box, tid in zip(xyxy_t0, track_t0):
                    draw_text_label(annotated_frame, box, str(tid), COLOR_TEAM_0.as_bgr())

            # 3. Vẽ Team 1 (Xanh dương) bằng annotator và nhãn trong suốt của riêng nó
            if len(xyxy_t1) > 0:
                det_t1 = sv.Detections(
                    xyxy=np.array(xyxy_t1, dtype=np.float32),
                    confidence=np.array(conf_t1, dtype=np.float32),
                    class_id=np.array(class_t1, dtype=np.int32),
                    tracker_id=np.array(track_t1, dtype=np.int32)
                )
                annotated_frame = ellipse_team_1.annotate(scene=annotated_frame, detections=det_t1)
                for box, tid in zip(xyxy_t1, track_t1):
                    draw_text_label(annotated_frame, box, str(tid), COLOR_TEAM_1.as_bgr())

            # 4. Vẽ Goalkeepers (Xanh lá) bằng annotator và nhãn trong suốt của riêng nó
            if len(xyxy_gk) > 0:
                det_gk = sv.Detections(
                    xyxy=np.array(xyxy_gk, dtype=np.float32),
                    confidence=np.array(conf_gk, dtype=np.float32),
                    class_id=np.array(class_gk, dtype=np.int32),
                    tracker_id=np.array(track_gk, dtype=np.int32)
                )
                annotated_frame = ellipse_gk.annotate(scene=annotated_frame, detections=det_gk)
                for box, tid in zip(xyxy_gk, track_gk):
                    draw_text_label(annotated_frame, box, f"GK{tid}", COLOR_GOALKEEPER.as_bgr())

            # 5. Vẽ Referees (Vàng sáng) bằng annotator và nhãn trong suốt của riêng nó
            if len(xyxy_ref) > 0:
                det_ref = sv.Detections(
                    xyxy=np.array(xyxy_ref, dtype=np.float32),
                    confidence=np.array(conf_ref, dtype=np.float32),
                    class_id=np.array(class_ref, dtype=np.int32),
                    tracker_id=np.array(track_ref, dtype=np.int32)
                )
                annotated_frame = ellipse_ref.annotate(scene=annotated_frame, detections=det_ref)
                for box, tid in zip(xyxy_ref, track_ref):
                    draw_text_label(annotated_frame, box, f"R{tid}", COLOR_REFEREE.as_bgr())

            # 6. Vẽ quả bóng (Dùng tam giác chỉ thị vàng + tâm tròn đỏ/cyan)
            if draw_center is not None and detector.ball_id is not None:
                x, y = draw_center
                fake_box = np.array([[x - 5, y - 5, x + 5, y + 5]], dtype=np.float32)
                ball_det = sv.Detections(
                    xyxy=fake_box,
                    confidence=np.array([raw_conf if raw_conf is not None else 1.0]),
                    class_id=np.array([detector.ball_id])
                )
                
                # Vẽ hình tam giác chỉ vào bóng ở phía trên
                annotated_frame = triangle_annotator.annotate(
                    scene=annotated_frame, detections=ball_det
                )

                # Vẽ tâm bóng: Đỏ nếu detect thật, màu vàng/cyan nếu là bóng nội suy
                radius = 5 if not is_estimated else 4
                color_ball = (0, 0, 255) if not is_estimated else (255, 255, 0)
                cv2.circle(annotated_frame, (int(x), int(y)), radius, color_ball, -1, cv2.LINE_AA)

                # Ghi nhãn text bên cạnh quả bóng
                ball_text = f"ball {raw_conf:.2f}" if not is_estimated else "ball interpolated"
                cv2.putText(
                    annotated_frame,
                    ball_text,
                    (int(x) + 12, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_ball,
                    2,
                    cv2.LINE_AA
                )

            # 7. Vẽ giao diện (HUD) tỉ lệ kiểm soát bóng Possession phía trên khung hình
            banner_w = 400
            banner_h = 45
            banner_x = int((w - banner_w) / 2)
            banner_y = 15

            # Vẽ thanh nền bán trong suốt (semi-transparent background)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # Vẽ thanh tiến trình sở hữu bóng (Possession Bar Chart)
            bar_x = banner_x + 20
            bar_y = banner_y + 30
            bar_w = banner_w - 40
            bar_h = 6

            t0_w = int(bar_w * (team_0_pct / 100))
            # Vẽ phần Team 0 (Đỏ) - BGR
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + t0_w, bar_y + bar_h), COLOR_TEAM_0.as_bgr(), -1)
            # Vẽ phần Team 1 (Xanh dương) - BGR
            cv2.rectangle(annotated_frame, (bar_x + t0_w, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_TEAM_1.as_bgr(), -1)

            # Ghi text tỷ lệ phần trăm
            cv2.putText(annotated_frame, f"TEAM 0: {team_0_pct}%", (banner_x + 20, banner_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEAM_0.as_bgr(), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, "POSSESSION", (banner_x + int(banner_w/2) - 40, banner_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"{team_1_pct}% :TEAM 1", (banner_x + banner_w - 120, banner_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEAM_1.as_bgr(), 2, cv2.LINE_AA)

            sink.write_frame(annotated_frame)

    # =========================================================================
    # GHI DỮ LIỆU POSSESSION CHI TIẾT RA FILE JSON
    # =========================================================================
    possession_data_path = os.path.join(
        os.path.dirname(TARGET_VIDEO_PATH),
        f"possession_data_{video_basename}.json"
    )
    with open(possession_data_path, "w") as jf:
        json.dump({
            "video_source": SOURCE_VIDEO_PATH,
            "resolution": {"width": w, "height": h},
            "total_frames": video_info.total_frames,
            "final_possession_percentage": {
                "team_0": team_0_pct,
                "team_1": team_1_pct
            },
            "frames": possession_frames
        }, jf, indent=2)

    print(f"\nPipeline complete! Tracked video saved to: {TARGET_VIDEO_PATH}")
    print(f"Possession stats JSON saved to: {possession_data_path}")

if __name__ == "__main__":
    main()
