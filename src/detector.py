"""
detector.py

Face detection utilities for FaceFinderAI.

Responsibilities:
- Load the InsightFace detector
- Detect faces
- Draw bounding boxes and landmarks
- Crop detected faces
"""

from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# --------------------------------------------------
# Singleton Detector
# --------------------------------------------------

_detector = None


def get_detector() -> FaceAnalysis:
    """
    Load the InsightFace detector only once.

    Returns:
        FaceAnalysis: Initialized detector.
    """

    global _detector

    if _detector is None:

        print("Loading InsightFace model...")

        _detector = FaceAnalysis(name="buffalo_l")

        # CPU
        _detector.prepare(
            ctx_id=-1,
            det_size=(640, 640)
        )

        print("Detector ready.\n")

    return _detector


# --------------------------------------------------
# Face Detection
# --------------------------------------------------

def detect_faces(image: np.ndarray, verbose: bool = False) -> List:
    """
    Detect faces in an image.

    Args:
        image: BGR image.
        verbose: Print detection information.

    Returns:
        List of detected Face objects.
    """

    detector = get_detector()

    faces = detector.get(image)

    if verbose:

        print("=" * 40)
        print("FACE DETECTION")
        print("=" * 40)

        print(f"Faces Detected : {len(faces)}")

        for i, face in enumerate(faces):

            print(f"\nFace {i}")

            print(f"Confidence : {face.det_score:.4f}")

            print(f"BBox       : {face.bbox.astype(int)}")

            print(f"Embedding  : {face.embedding.shape}")

    return faces


# --------------------------------------------------
# Draw Bounding Boxes
# --------------------------------------------------

def draw_detections(image: np.ndarray, faces: List) -> np.ndarray:
    """
    Draw bounding boxes, face IDs and landmarks.

    Args:
        image: Input BGR image.
        faces: List of Face objects.

    Returns:
        Annotated image.
    """

    output = image.copy()

    for i, face in enumerate(faces):

        x1, y1, x2, y2 = face.bbox.astype(int)

        # Bounding Box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Face ID
        cv2.putText(
            output,
            f"Face {i}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Landmarks
        for point in face.kps:

            x, y = point.astype(int)

            cv2.circle(
                output,
                (x, y),
                2,
                (255, 255, 0),
                -1,
            )

    return output


# --------------------------------------------------
# Crop Faces
# --------------------------------------------------

def crop_faces(image: np.ndarray, faces: List) -> List[np.ndarray]:
    """
    Crop all detected faces.

    Args:
        image: Original BGR image.
        faces: List of Face objects.

    Returns:
        List of cropped face images.
    """

    crops = []

    height, width = image.shape[:2]

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        # Clamp coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        crop = image[y1:y2, x1:x2]

        crops.append(crop)

    return crops


# --------------------------------------------------
# Demo
# --------------------------------------------------

if __name__ == "__main__":

    from src.utils import (
        load_image,
        resize_image,
        display_image,
    )

    image = load_image("dataset/wedding_images/test.jpg")

    if image is None:
        raise RuntimeError("Could not load image.")

    image = resize_image(image, 800)

    faces = detect_faces(image, verbose=True)

    annotated = draw_detections(image, faces)

    display_image(annotated, "Detected Faces")

    print(f"\nDetected {len(faces)} face(s).")