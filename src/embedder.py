"""
embedder.py

Generate and compare face embeddings.
"""

from typing import Optional

import numpy as np


def get_embedding(face) -> np.ndarray:
    """
    Extract and normalize the embedding from an InsightFace Face object.

    Args:
        face: InsightFace Face object.

    Returns:
        Normalized 512-dimensional embedding.
    """

    embedding: Optional[np.ndarray] = face.embedding

    if embedding is None:
        raise ValueError("No embedding was generated for this face.")

    embedding = embedding.astype(np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Embedding norm is zero.")

    return embedding / norm


def cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two normalized embeddings.

    Returns:
        Similarity score between -1 and 1.
    """

    return float(np.dot(embedding1, embedding2))


def compare_faces(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = 0.50,
) -> bool:
    """
    Compare two embeddings.

    Args:
        embedding1: First embedding.
        embedding2: Second embedding.
        threshold: Similarity threshold.

    Returns:
        True if the faces are considered the same person.
    """

    similarity = cosine_similarity(
        embedding1,
        embedding2,
    )

    print("\n========== FACE COMPARISON ==========")
    print(f"Cosine Similarity : {similarity:.4f}")

    if similarity >= threshold:
        print("Result            : SAME PERSON")
        return True

    print("Result            : DIFFERENT PERSON")
    return False


if __name__ == "__main__":

    from src.detector import detect_faces
    from src.utils import load_image, resize_image

    image1 = load_image("dataset/wedding_images/test.jpg")
    image2 = load_image("dataset/wedding_images/test.jpg")

    image1 = resize_image(image1, 800)
    image2 = resize_image(image2, 800)

    faces1 = detect_faces(image1)
    faces2 = detect_faces(image2)

    if not faces1 or not faces2:
        raise RuntimeError("Face detection failed.")

    emb1 = get_embedding(faces1[0])
    emb2 = get_embedding(faces2[0])

    compare_faces(emb1, emb2)