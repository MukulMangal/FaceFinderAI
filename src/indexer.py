"""
indexer.py

Indexes all faces in a dataset and builds the FAISS index.
"""

import json
import os
from typing import List

import faiss
import numpy as np

from src.detector import detect_faces
from src.embedder import get_embedding
from src.utils import load_image


# ==========================================================
# Build FAISS Index
# ==========================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build and save a FAISS index.

    Args:
        embeddings: Matrix of face embeddings.

    Returns:
        FAISS index.
    """

    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, "embeddings/faiss.index")

    print("\n========== FAISS ==========")
    print(f"Vectors Indexed : {index.ntotal}")
    print("Saved           : embeddings/faiss.index")

    return index


# ==========================================================
# Load FAISS Index
# ==========================================================

def load_faiss_index() -> faiss.Index:
    """
    Load the saved FAISS index.
    """

    if not os.path.exists("embeddings/faiss.index"):
        raise FileNotFoundError("FAISS index not found.")

    return faiss.read_index("embeddings/faiss.index")


# ==========================================================
# Dataset Indexing
# ==========================================================

def index_dataset(image_dir: str) -> None:
    """
    Process all images in a folder and create an embedding database.

    Args:
        image_dir: Folder containing wedding images.
    """

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"{image_dir} does not exist.")

    os.makedirs("embeddings", exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    image_files = sorted(
        [
            file
            for file in os.listdir(image_dir)
            if file.endswith(valid_extensions)
        ]
    )

    all_embeddings: List[np.ndarray] = []
    metadata = []

    face_id = 0
    images_processed = 0
    faces_found = 0

    print("\n========== INDEXING DATASET ==========\n")

    for index, filename in enumerate(image_files, start=1):

        image_path = os.path.join(image_dir, filename)

        print(f"[{index}/{len(image_files)}] {filename}")

        try:

            image = load_image(image_path)

            if image is None:
                print("   Could not load image.\n")
                continue

            images_processed += 1

            faces = detect_faces(image)

            print(f"   Faces Detected : {len(faces)}")

            for face in faces:

                embedding = get_embedding(face)

                all_embeddings.append(embedding)

                metadata.append(
                    {
                        "face_id": face_id,
                        "image_path": image_path,
                        "bbox": face.bbox.astype(float).tolist(),
                        "score": float(face.det_score),
                    }
                )

                face_id += 1
                faces_found += 1

            print("   Indexed Successfully\n")

        except Exception as e:

            print(f"   ERROR : {e}\n")

            continue

    if len(all_embeddings) == 0:
        raise RuntimeError("No faces were indexed.")

    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)

    # Save embeddings
    np.save(
        "embeddings/embeddings.npy",
        embeddings_matrix,
    )

    # Save metadata
    with open(
        "embeddings/metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # Build FAISS
    build_faiss_index(embeddings_matrix)

    print("\n========== INDEX SUMMARY ==========")
    print(f"Images Processed : {images_processed}")
    print(f"Faces Indexed    : {faces_found}")
    print(f"Embedding Shape  : {embeddings_matrix.shape}")
    print("Saved            : embeddings/embeddings.npy")
    print("Saved            : embeddings/metadata.json")


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    index_dataset("dataset/wedding_images")