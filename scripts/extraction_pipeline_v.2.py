import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Any
from tqdm import tqdm
from utils import get_anchor_and_scale, normalize_spatial_points, uniform_subsampling

def process_video_to_landmarks(
    video_path: str, 
    holistic_model: Any, 
    target_frames: int = 16
) -> np.ndarray:
    """
    Opens a video file, extracts, spatially normalizes each frame, 
    and applies uniform subsampling to the resulting vector.

    Args:
        video_path (str): Path to the MP4 file.
        holistic_model (Any): Contextual instance of MediaPipe Holistic.
        target_frames (int): Exact number of output frames.

    Returns:
        np.ndarray: Structured tensor of shape (target_frames, 225).
    """
    capture = cv2.VideoCapture(video_path)
    sequence_history: List[np.ndarray] = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Mandatory color space conversion for MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(rgb_image)

        # 1. Determine the local coordinate system for this specific frame
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)

        # 2. Extract base coordinates or initialize with zeros if no detection
        raw_pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 3)
            
        raw_left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
            
        raw_right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)

        # 3. Frame-by-frame spatial normalization (Scale and translation invariance)
        normalized_pose = normalize_spatial_points(raw_pose, anchor, scale)
        normalized_left_hand = normalize_spatial_points(raw_left_hand, anchor, scale)
        normalized_right_hand = normalize_spatial_points(raw_right_hand, anchor, scale)

        # 4. Concatenation into the unified feature space (225 dimensions)
        frame_vector = np.concatenate([normalized_pose, normalized_left_hand, normalized_right_hand])
        sequence_history.append(frame_vector)

    capture.release()

    # 5. Deterministic temporal compression using equally spaced sampling
    return uniform_subsampling(sequence_history, target_frames=target_frames)


def run_extraction_pipeline(source_dir: str, dest_dir: str) -> None:
    """
    Orchestrates the massive dataset conversion by iterating through class folders,
    processing video files, and storing clean NumPy arrays.

    Args:
        source_dir (str): Root path of the new MP4 video dataset.
        dest_dir (str): Path where normalized .npy files will be saved.
    """
    os.makedirs(dest_dir, exist_ok=True)
    sign_classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    mp_holistic = mp.solutions.holistic

    # Initialize the MediaPipe context optimized for batch processing
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for sign_class in sign_classes:
            print(f"\n[*] Processing kinematic category: {sign_class}")
            class_input_path = os.path.join(source_dir, sign_class)
            class_output_path = os.path.join(dest_dir, sign_class)
            os.makedirs(class_output_path, exist_ok=True)

            available_videos = glob.glob(os.path.join(class_input_path, "*.mp4"))

            for video_path in tqdm(available_videos, desc="Video Progress"):
                npy_filename = os.path.basename(video_path).replace('.mp4', '.npy')
                final_save_path = os.path.join(class_output_path, npy_filename)

                # Optimization to avoid reprocessing if the script is interrupted
                if os.path.exists(final_save_path):
                    continue

                try:
                    landmarks_tensor = process_video_to_landmarks(video_path, holistic, target_frames=16)
                    if landmarks_tensor.shape[0] == 16:
                        np.save(final_save_path, landmarks_tensor)
                except Exception as process_error:
                    print(f"\n[!] Critical error in file {npy_filename}: {process_error}")


if __name__ == "__main__":
    # Local absolute path configuration for the workspace
    NEW_VIDEOS_DIR = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\entrenamiento\dataset"
    OUTPUT_NPY_DIR = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\entrenamiento\dataset_procesado_npy"
    
    run_extraction_pipeline(NEW_VIDEOS_DIR, OUTPUT_NPY_DIR)