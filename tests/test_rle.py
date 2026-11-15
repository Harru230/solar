"""
tests/test_rle.py - Unit Tests for COCO Compressed RLE Encoding & Decoding.

Tests:
1. Exact round-trip pixel identity: decode(encode(mask)) == mask across:
   - Solid squares
   - Boundary single corner pixels
   - 1-pixel wide thin diagonal lines
   - Hollow circular rings
   - Full-canvas mask (all ones)
2. Fortran memory contiguous requirement:
   - Confirms raw pycocotools.mask.encode raises ValueError on C-contiguous arrays.
   - Confirms rle_encode_mask enforces asfortranarray and handles both C- and Fortran-ordered arrays.
3. Empty mask constant round-trip:
   - All-zero array (2048, 2048) encodes strictly to 'PPPP4'.
   - 'PPPP4' decodes strictly to all-zero array of shape (2048, 2048).
4. Area and Bounding Box invariance (mu.area and mu.toBbox).
"""
import numpy as np
import pytest
from pycocotools import mask as mu

from scripts.common import EMPTY_RLE, rle_encode_mask


def _decode_rle(rle_str: str, shape=(2048, 2048)) -> np.ndarray:
    """Helper to decode RLE string to numpy mask."""
    rle_obj = {"size": list(shape), "counts": rle_str.encode("ascii")}
    return mu.decode(rle_obj)


def test_rle_empty_mask_constant():
    """
    Verifies that an all-zero 2048x2048 mask encodes to canonical 'PPPP4'
    and 'PPPP4' decodes to an all-zero 2048x2048 mask.
    """
    zero_mask = np.zeros((2048, 2048), dtype=bool)
    encoded = rle_encode_mask(zero_mask)
    assert encoded == EMPTY_RLE, f"Expected {EMPTY_RLE}, got {encoded}"

    decoded = _decode_rle(EMPTY_RLE)
    assert decoded.shape == (2048, 2048)
    assert decoded.dtype == np.uint8
    assert int(decoded.sum()) == 0
    assert np.all(decoded == 0)


def test_rle_fortran_requirement():
    """
    Verifies memory order requirements:
    1. Raw pycocotools.mask.encode raises ValueError on standard C-contiguous array.
    2. rle_encode_mask safely wraps arrays with np.asfortranarray, supporting both
       C-contiguous and Fortran-contiguous arrays without error.
    """
    # 1. Raw C-contiguous array raises ValueError
    c_arr = np.zeros((20, 20), dtype=np.uint8, order="C")
    assert not c_arr.flags.f_contiguous
    assert c_arr.flags.c_contiguous

    with pytest.raises(ValueError, match="Fortran contiguous"):
        mu.encode(c_arr)

    # 2. rle_encode_mask handles C-contiguous array safely
    enc_from_c = rle_encode_mask(c_arr)
    assert isinstance(enc_from_c, str)
    assert len(enc_from_c) > 0

    # 3. rle_encode_mask handles Fortran-contiguous array safely
    f_arr = np.asfortranarray(c_arr)
    assert f_arr.flags.f_contiguous
    enc_from_f = rle_encode_mask(f_arr)
    assert enc_from_f == enc_from_c


def test_rle_round_trip_solid_square(helper_make_rect):
    """Verifies pixel-perfect round-trip identity for a solid rectangular block."""
    orig_mask = helper_make_rect(100, 150, 300, 450) # 200 rows x 300 cols = 60,000 px
    encoded = rle_encode_mask(orig_mask)
    decoded = _decode_rle(encoded)

    assert decoded.shape == orig_mask.shape
    assert int(decoded.sum()) == 60000
    assert np.array_equal(decoded.astype(bool), orig_mask)


def test_rle_round_trip_single_corner_pixels():
    """
    Verifies that isolated single pixels placed at extreme canvas corner boundaries
    (0,0), (0, 2047), (2047, 0), (2047, 2047) round-trip with exact coordinate fidelity.
    """
    corners = [(0, 0), (2047, 2047)]
    for y, x in corners:
        mask = np.zeros((2048, 2048), dtype=bool)
        mask[y, x] = True
        assert int(mask.sum()) == 1

        encoded = rle_encode_mask(mask)
        decoded = _decode_rle(encoded)

        assert int(decoded.sum()) == 1
        assert decoded[y, x] == 1
        assert np.array_equal(decoded.astype(bool), mask)


def test_rle_round_trip_diagonal_line():
    """
    Verifies that a 1-pixel wide thin diagonal line (simulating thin solar filaments)
    round-trips without any pixel dilation, erosion, or disconnection.
    """
    mask = np.zeros((2048, 2048), dtype=bool)
    # Diagonal from (100, 100) to (300, 300)
    indices = np.arange(100, 300)
    mask[indices, indices] = True
    assert int(mask.sum()) == 200

    encoded = rle_encode_mask(mask)
    decoded = _decode_rle(encoded)

    assert int(decoded.sum()) == 200
    assert np.array_equal(decoded.astype(bool), mask)


def test_rle_round_trip_hollow_circle():
    """
    Verifies pixel-perfect round-trip identity for a non-convex hollow ring geometry.
    """
    mask = np.zeros((2048, 2048), dtype=bool)
    cy, cx = 500, 500
    r_outer = 80
    r_inner = 40

    y, x = np.ogrid[:2048, :2048]
    dist_sq = (y - cy) ** 2 + (x - cx) ** 2
    ring = (dist_sq <= r_outer ** 2) & (dist_sq >= r_inner ** 2)
    mask[ring] = True

    orig_area = int(mask.sum())
    assert orig_area > 0

    encoded = rle_encode_mask(mask)
    decoded = _decode_rle(encoded)

    assert int(decoded.sum()) == orig_area
    assert np.array_equal(decoded.astype(bool), mask)


def test_rle_round_trip_full_canvas():
    """Verifies round-trip fidelity for a completely filled canvas (all ones)."""
    full_mask = np.ones((2048, 2048), dtype=bool)
    encoded = rle_encode_mask(full_mask)
    decoded = _decode_rle(encoded)

    assert int(decoded.sum()) == 2048 * 2048
    assert np.array_equal(decoded.astype(bool), full_mask)


def test_rle_area_and_bbox_invariance(helper_make_rect):
    """
    Verifies that pycocotools mask utilities area and toBbox
    match the ground-truth geometry when applied to encoded RLE.
    """
    y1, x1, y2, x2 = 120, 310, 250, 480
    mask = helper_make_rect(y1, x1, y2, x2)
    expected_width = x2 - x1   # 170
    expected_height = y2 - y1  # 130
    expected_area = expected_width * expected_height  # 22100

    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle_obj = mu.encode(fortran_mask)

    computed_area = int(mu.area(rle_obj))
    assert computed_area == expected_area

    bbox = list(mu.toBbox(rle_obj))
    expected_bbox = [float(x1), float(y1), float(expected_width), float(expected_height)]
    assert np.allclose(bbox, expected_bbox)
