"""Construct the fixed affine saddle family from the small-network checkpoint."""

import numpy as np
from scipy.linalg import null_space

from model import SMALL_WIDTHS, WIDE_WIDTHS, layout

# Layer indices are zero-based: widen the first and third hidden layers.
SPLIT_COUNTS = {0: [5, 5], 2: [4, 3]}


def embed(theta, widths, counts, coefficients=None):
    """Split neurons while preserving their total outgoing weights."""
    weights, biases = ([], [])
    for a, b, ws, bs in layout(widths)[0]:
        weights.append(theta[ws].reshape(b, a).copy())
        biases.append(None if bs is None else theta[bs].copy())
    new_widths = list(widths)
    metadata = []
    for layer, copies in sorted(counts.items()):
        indices = np.repeat(np.arange(len(copies)), copies)
        coeff = (
            np.concatenate([np.full(c, 1 / c) for c in copies])
            if coefficients is None
            else np.asarray(coefficients[layer])
        )
        for j in range(len(copies)):
            assert abs(coeff[indices == j].sum() - 1) < 1e-12
        weights[layer] = weights[layer][indices]
        biases[layer] = biases[layer][indices]
        weights[layer + 1] = weights[layer + 1][:, indices] * coeff[None, :]
        new_widths[layer + 1] = len(indices)
        metadata.append(
            dict(layer=layer + 1, parents=indices.tolist(), coefficients=coeff.tolist())
        )
    flat = []
    for w, b in zip(weights, biases):
        flat.append(w.ravel())
        if b is not None:
            flat.append(b)
    return (np.concatenate(flat), tuple(new_widths), metadata)


def tangent(theta, widths, counts):
    """Affine split coefficients for NONADJACENT widened layers only."""
    assert all((l + 1 not in counts for l in counts))
    coefficients = {
        l: np.concatenate([np.full(c, 1 / c) for c in cs]) for l, cs in counts.items()
    }
    q, wide, _ = embed(theta, widths, counts, coefficients)
    columns = []
    for l, cs in sorted(counts.items()):
        start = 0
        for c in cs:
            for j in range(c - 1):
                changed = {a: b.copy() for a, b in coefficients.items()}
                changed[l][start + j] += 1
                changed[l][start + c - 1] -= 1
                columns.append(embed(theta, widths, counts, changed)[0] - q)
            start += c
    return np.column_stack(columns)


def build_reference(theta_small):
    """Return (reference_parameters, widths, orthonormal_normal_basis)."""
    if theta_small.shape != (30,):
        raise ValueError("theta_small must have shape (30,)")
    center, widths, _ = embed(theta_small, SMALL_WIDTHS, SPLIT_COUNTS)
    directions = tangent(theta_small, SMALL_WIDTHS, SPLIT_COUNTS)
    normal = null_space(directions.T)
    if (
        widths != WIDE_WIDTHS
        or directions.shape != (100, 13)
        or normal.shape != (100, 87)
    ):
        raise ValueError("Unexpected embedding or manifold dimension")
    return center, widths, normal
