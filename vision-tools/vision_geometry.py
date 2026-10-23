"""Patch ordering and dynamic geometry for the compiled vision graph."""
import numpy as np
PATCH=16
MERGE=2
ROPE_THETA=10000.0

def block_major_positions(grid_h: int, grid_w: int):
    """(row, col) per patch token in spatial-merge-block-major order."""
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    shape = (grid_h // MERGE, MERGE, grid_w // MERGE, MERGE)
    rows = rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    cols = cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    return rows, cols


def patchify(pixels: np.ndarray) -> tuple[np.ndarray, int, int]:
    """pixels: (H, W, 3) uint8 with sides divisible by 32 -> (tokens, 1536)."""
    height, width, channels = pixels.shape
    if channels != 3 or height % (PATCH * MERGE) or width % (PATCH * MERGE):
        raise ValueError("image must be RGB with sides divisible by 32")
    grid_h, grid_w = height // PATCH, width // PATCH
    normalized = pixels.astype(np.float32)
    normalized /= 127.5
    normalized -= 1.0
    normalized = normalized.transpose(2, 0, 1)  # (C, H, W)
    patches = normalized.reshape(
        3, grid_h // MERGE, MERGE, PATCH, grid_w // MERGE, MERGE, PATCH
    )
    # (block_row, block_col, in_row, in_col, channel, patch_row, patch_col)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
    patches = patches.reshape(grid_h * grid_w, 3, 1, PATCH, PATCH)
    patches = np.broadcast_to(patches, (grid_h * grid_w, 3, 2, PATCH, PATCH))
    return np.ascontiguousarray(patches).reshape(grid_h * grid_w, -1), grid_h, grid_w


def patchify_into(pixels: np.ndarray, out: np.ndarray):
    """Write one merge-block row at a time, preserving FP32 normalization."""
    height, width, channels = pixels.shape
    if channels != 3 or height % 32 or width % 32:
        raise ValueError("image must be RGB with sides divisible by 32")
    if out.shape != (height // PATCH * (width // PATCH), 1536):
        raise ValueError("unexpected patch destination shape")
    rows_per_block = MERGE * (width // PATCH)
    for top in range(0, height, PATCH * MERGE):
        offset = top // (PATCH * MERGE) * rows_per_block
        out[offset:offset + rows_per_block] = patchify(pixels[top:top + 32])[0]


def interpolated_positions(table: np.ndarray, grid_h: int, grid_w: int, side: int,
                           *, out=None):
    rows, cols = block_major_positions(grid_h, grid_w)

    def taps(index, size):
        src = index * (side - 1) / max(size - 1, 1)
        lower = np.floor(src)
        offsets = np.arange(2)
        tap = np.clip(lower[:, None] + offsets, 0, side - 1).astype(np.int64)
        weight = np.clip(1.0 - np.abs(src[:, None] - lower[:, None] - offsets), 0, None)
        return tap, weight

    h_taps, h_weights = taps(rows, grid_h)
    w_taps, w_weights = taps(cols, grid_w)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    if out is None:
        out = np.empty((grid_h * grid_w, table.shape[1]), dtype=table.dtype)
    if out.shape != (grid_h * grid_w, table.shape[1]):
        raise ValueError("unexpected position destination shape")
    # Keep the same four-tap reduction order without two full-grid expansions.
    for start in range(0, len(indices), 128):
        end = min(start + 128, len(indices))
        gathered = table[indices[start:end]]
        gathered *= weights[start:end, :, None].astype(np.float32)
        out[start:end] = gathered.sum(axis=1)
    return out


def rope_tables(grid_h: int, grid_w: int, head_dim: int):
    rows, cols = block_major_positions(grid_h, grid_w)
    frequencies = 1.0 / (
        ROPE_THETA
        ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2))
    )
    half = np.concatenate(
        [rows[:, None] * frequencies[None, :], cols[:, None] * frequencies[None, :]],
        axis=1,
    ).astype(np.float32)
    angles = np.concatenate([half, half], axis=1)
    return np.cos(angles), np.sin(angles)
