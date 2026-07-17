"""
search.py

Search similar faces using FAISS.
"""

import json
import os
from typing import List, Dict

import cv2
import numpy as np

from src.detector import detect_faces
from src.embedder import get_embedding
from src.indexer import load_faiss_index
from src.utils import load_image, resize_image, save_image


# =====================================================
# Metadata
# =====================================================

def load_metadata() -> List[dict]:
    """Load metadata for indexed faces."""

    metadata_path = "embeddings/metadata.json"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.json not found.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================================
# Search
# =====================================================

def search_faces(
    query_image_path: str,
    top_k: int = 10,
    threshold: float = 0.50,
) -> List[Dict]:
    """
    Search for faces similar to the query image.

    Args:
        query_image_path: Path to uploaded image.
        top_k: Maximum neighbours.
        threshold: Similarity threshold.

    Returns:
        List of matching faces.
    """

    # -----------------------------
    # Load Query Image
    # -----------------------------

    image = load_image(query_image_path)

    if image is None:
        raise RuntimeError("Unable to load query image.")

    image = resize_image(image, 800)

    # -----------------------------
    # Detect Face
    # -----------------------------

    faces = detect_faces(image)

    if len(faces) == 0:
        print("No face detected.")
        return []

    # For now we always search using the first face.
    # Later the frontend will allow selecting a face.
    query_embedding = get_embedding(faces[0])

    query = query_embedding.reshape(1, -1).astype(np.float32)

    # -----------------------------
    # Load Index
    # -----------------------------

    index = load_faiss_index()

    metadata = load_metadata()

    distances, indices = index.search(query, top_k)

    results = []
    seen_images = set()

    os.makedirs("output", exist_ok=True)

    # -----------------------------
    # Process Results
    # -----------------------------

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        if score < threshold:
            continue

        meta = metadata[idx]

        image_path = meta["image_path"]

        # Skip duplicate images
        if image_path in seen_images:
            continue

        seen_images.add(image_path)

        matched = load_image(image_path)

        if matched is None:
            continue

        x1, y1, x2, y2 = map(int, meta["bbox"])

        cv2.rectangle(
            matched,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            matched,
            f"{score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        output_path = os.path.join(
            "output",
            f"match_{os.path.basename(image_path)}",
        )

        save_image(
            matched,
            output_path,
        )

        results.append(
            {
                "image_path": image_path,
                "score": float(score),
                "bbox": meta["bbox"],
                "output_image": output_path,
            }
        )

    return results


# =====================================================
# Demo
# =====================================================

if __name__ == "__main__":

    query = input("Query image: ")

    matches = search_faces(query)

    print("\n========== RESULTS ==========\n")

    for match in matches:

        print(match)

    print(f"\nTotal Matches: {len(matches)}")