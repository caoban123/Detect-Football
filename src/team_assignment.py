import os
import pickle
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def get_torso_crop(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
    Cắt vùng áo đấu (torso - từ 20% đến 65% chiều cao của bounding box)
    để loại bỏ cỏ sân bóng, tất và giày của cầu thủ.
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = map(int, xyxy)
    
    # Kẹp tọa độ nằm trong biên ảnh
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    box_h = y2 - y1
    torso_y1 = y1 + int(box_h * 0.20)
    torso_y2 = y1 + int(box_h * 0.65)
    
    # Kiểm tra tính hợp lệ của crop torso
    if torso_y2 > torso_y1 and x2 > x1:
        return image[torso_y1:torso_y2, x1:x2]
    return image[y1:y2, x1:x2]  # Fallback nếu box lỗi


def is_valid_player_crop(xyxy: np.ndarray, frame_shape: tuple, confidence: float, min_conf: float = 0.35) -> bool:
    """
    Lọc bỏ các crop chất lượng thấp để tránh làm nhiễu thuật toán phân cụm KMeans:
    - Bỏ qua vật thể quá nhỏ hoặc quá xa
    - Bỏ qua tỉ lệ khung hình bất thường
    - Bỏ qua các đối tượng quá gần biên ảnh hoặc nằm trên khán đài
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = xyxy
    box_w = x2 - x1
    box_h = y2 - y1
    
    # 1. Loại bỏ nếu độ tự tin thấp
    if confidence < min_conf:
        return False
        
    # 2. Loại bỏ các box quá nhỏ (xa camera)
    if box_w < 15 or box_h < 35:
        return False
        
    # 3. Kiểm tra tỉ lệ khung hình (cầu thủ thường cao và hẹp, tỉ lệ w/h từ 0.15 đến 0.85)
    aspect_ratio = box_w / box_h
    if not (0.15 <= aspect_ratio <= 0.85):
        return False
        
    # 4. Loại bỏ các box bị cắt sát biên (trong khoảng 10px của biên ảnh)
    if x1 < 10 or y1 < 10 or x2 > w - 10 or y2 > h - 10:
        return False
        
    # 5. Loại bỏ đối tượng ngoài sân đấu (ví dụ: nằm ở 8% phía trên màn hình - vùng khán đài)
    center_y = (y1 + y2) / 2
    if center_y < h * 0.08:
        return False
        
    return True


def create_batches(sequence, batch_size: int):
    """ Chia dữ liệu thành các batch nhỏ """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    Bộ phân loại đội bóng tự động sử dụng:
    - SigLIP Vision Model làm bộ trích xuất đặc trưng (embedding).
    - UMAP giảm chiều dữ liệu từ 768D xuống 3D để tập trung vào màu áo đấu.
    - KMeans phân cụm 3D thành 2 đội (Team 0 và Team 1).
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        
        print(f"Initializing SigLIP model on {device}...")
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        
        # Cố định random_state=42 để kết quả không bị thay đổi giữa các lần chạy
        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self._is_fitted = False

    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        """ Trích xuất đặc trưng 768-D từ các crop torso cầu thủ """
        if len(crops) == 0:
            return np.empty((0, 768), dtype=np.float32)

        # Chuyển đổi BGR OpenCV sang RGB PIL cho HuggingFace Processor
        pil_crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = list(create_batches(pil_crops, self.batch_size))
        
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Extracting SigLIP embeddings'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                # Lấy vector trung bình trên chiều không gian để ra embedding 768D
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        all_embeddings = np.concatenate(data, axis=0)
        # Chuẩn hóa L2 embeddings để ổn định khoảng cách cosine
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        # Tránh chia cho 0
        norms = np.where(norms == 0, 1.0, norms)
        return all_embeddings / norms

    def fit(self, crops: list[np.ndarray]) -> None:
        """ Huấn luyện bộ giảm chiều UMAP và thuật toán KMeans trên tập crops cầu thủ thu thập được """
        if len(crops) < 10:
            print("WARNING: Too few crops to fit TeamClassifier! Need at least 10 crops.")
            self._is_fitted = False
            return
            
        print(f"Fitting TeamClassifier with {len(crops)} crops...")
        embeddings = self.extract_features(crops)
        projections = self.reducer.fit_transform(embeddings)
        self.cluster_model.fit(projections)
        self._is_fitted = True
        print("TeamClassifier fitted successfully.")

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """ Dự đoán team (0 hoặc 1) cho các crops cầu thủ mới """
        if not self._is_fitted:
            print("WARNING: TeamClassifier is not fitted yet!")
            return np.zeros(len(crops), dtype=np.int32)
            
        if len(crops) == 0:
            return np.array([], dtype=np.int32)

        embeddings = self.extract_features(crops)
        projections = self.reducer.transform(embeddings)
        return self.cluster_model.predict(projections)

    def is_fitted(self) -> bool:
        return self._is_fitted

    def save(self, filepath: str) -> None:
        """ Lưu trữ mô hình umap reducer và KMeans đã fit xuống ổ cứng """
        if not self._is_fitted:
            print("Cannot save: TeamClassifier is not fitted.")
            return
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'reducer': self.reducer,
                'cluster_model': self.cluster_model,
                'is_fitted': self._is_fitted
            }, f)
        print(f"Saved TeamClassifier models to {filepath}")

    def load(self, filepath: str) -> bool:
        """ Khôi phục mô hình umap reducer và KMeans từ ổ cứng """
        if not os.path.exists(filepath):
            print(f"No cached classifier found at {filepath}")
            return False
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.reducer = data['reducer']
                self.cluster_model = data['cluster_model']
                self._is_fitted = data['is_fitted']
            print(f"Loaded TeamClassifier models from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading cached classifier: {e}")
            return False


def assign_goalkeeper_team(gk_position: np.ndarray, team_0_positions: list[np.ndarray], team_1_positions: list[np.ndarray]) -> int:
    """
    Gán goalkeeper vào Team 0 hoặc Team 1 dựa trên khoảng cách
    đến tâm đội hình (centroid) của các cầu thủ mỗi team trong frame đó.
    """
    if len(team_0_positions) == 0 and len(team_1_positions) == 0:
        return 0 # Mặc định
        
    if len(team_0_positions) == 0:
        return 1
    if len(team_1_positions) == 0:
        return 0
        
    centroid_0 = np.mean(team_0_positions, axis=0)
    centroid_1 = np.mean(team_1_positions, axis=0)
    
    dist_0 = np.linalg.norm(gk_position - centroid_0)
    dist_1 = np.linalg.norm(gk_position - centroid_1)
    
    return 0 if dist_0 < dist_1 else 1
