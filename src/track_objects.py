import supervision as sv
from trackers import ByteTrackTracker

class FootballTracker:
    def __init__(self, fps: float, lost_track_buffer: int = 60,
                 track_activation_threshold: float = 0.15,
                 high_conf_det_threshold: float = 0.45,
                 minimum_iou_threshold: float = 0.20,
                 minimum_consecutive_frames: int = 2):
        """
        Initialize the ByteTrack-based Football Tracker with optimal defaults.
        """
        self.tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=fps,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
            minimum_iou_threshold=minimum_iou_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the tracker state with the filtered player/referee/goalkeeper detections.
        """
        return self.tracker.update(detections)
