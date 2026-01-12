# tests/test_resource_grid_extractor.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------
# PATH FIX (da importi rade kad pytest krene iz root-a)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.resource_grid_extractor import (
    pbch_symbol_indices_for_subframes,
    PBCHConfig,
    PBCHExtractor,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_grid(ndlrb: int = 6, num_subframes: int = 4, normal_cp: bool = True, *, seed: int = 0) -> np.ndarray:
    """
    Kreira "dummy" grid shape (12*ndlrb, num_subframes*symbols_per_subframe)
    popunjen kompleksnim brojevima (deterministički).
    """
    rng = np.random.default_rng(seed)
    n_sc = 12 * ndlrb
    n_sym = num_subframes * (14 if normal_cp else 12)
    re = rng.standard_normal((n_sc, n_sym))
    im = rng.standard_normal((n_sc, n_sym))
    return (re + 1j * im).astype(np.complex64)


def central_k0(num_subcarriers: int) -> int:
    return (num_subcarriers - 72) // 2


def expected_scan_positions_for_one_symbol(
    num_subcarriers: int,
    symbol_index: int,
    reserved_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """
    Redoslijed (sc, sym) koji extractor prolazi kroz centralnih 6 RB × 12.
    """
    k0 = central_k0(num_subcarriers)
    pos: list[tuple[int, int]] = []
    for rb in range(6):
        base = k0 + rb * 12
        for off in range(12):
            sc = base + off
            if reserved_mask is not None and reserved_mask[sc, symbol_index]:
                continue
            pos.append((sc, symbol_index))
    return pos


# ============================================================
# 1) pbch_symbol_indices_for_subframes helper
# ============================================================

def test_pbch_symbol_indices_for_subframes_normal_cp_one_sf():
    idx = pbch_symbol_indices_for_subframes(num_subframes=1, normal_cp=True, start_subframe=0)
    assert idx == [7, 8, 9, 10]


def test_pbch_symbol_indices_for_subframes_extended_cp_one_sf():
    idx = pbch_symbol_indices_for_subframes(num_subframes=1, normal_cp=False, start_subframe=0)
    assert idx == [6, 7, 8, 9]


def test_pbch_symbol_indices_for_subframes_normal_cp_two_sf_start0():
    idx = pbch_symbol_indices_for_subframes(num_subframes=2, normal_cp=True, start_subframe=0)
    assert idx == [7, 8, 9, 10, 21, 22, 23, 24]  # 14 offset


def test_pbch_symbol_indices_for_subframes_extended_cp_two_sf_start0():
    idx = pbch_symbol_indices_for_subframes(num_subframes=2, normal_cp=False, start_subframe=0)
    assert idx == [6, 7, 8, 9, 18, 19, 20, 21]  # 12 offset


def test_pbch_symbol_indices_for_subframes_start_subframe_offset():
    idx = pbch_symbol_indices_for_subframes(num_subframes=1, normal_cp=True, start_subframe=3)
    assert idx == [3 * 14 + 7, 3 * 14 + 8, 3 * 14 + 9, 3 * 14 + 10]


def test_pbch_symbol_indices_for_subframes_invalid_num_subframes_raises():
    with pytest.raises(ValueError):
        pbch_symbol_indices_for_subframes(num_subframes=0, normal_cp=True, start_subframe=0)


# ============================================================
# 2) PBCHConfig post_init
# ============================================================

def test_pbch_config_default_indices_normal_cp():
    cfg = PBCHConfig(normal_cp=True, pbch_symbol_indices=None, pbch_symbols_to_extract=240)
    assert cfg.pbch_symbol_indices == [7, 8, 9, 10]


def test_pbch_config_default_indices_extended_cp():
    cfg = PBCHConfig(normal_cp=False, pbch_symbol_indices=None, pbch_symbols_to_extract=240)
    assert cfg.pbch_symbol_indices == [6, 7, 8, 9]


def test_pbch_config_copies_indices_list():
    lst = [1, 2, 3]
    cfg = PBCHConfig(normal_cp=True, pbch_symbol_indices=lst, pbch_symbols_to_extract=240)
    lst.append(99)
    assert cfg.pbch_symbol_indices == [1, 2, 3]  # nije se promijenilo


def test_pbch_config_invalid_symbols_to_extract_raises():
    with pytest.raises(ValueError):
        PBCHConfig(pbch_symbols_to_extract=0)


# ============================================================
# 3) PBCHExtractor.extract unhappy paths
# ============================================================

def test_extract_grid_not_2d_raises():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = np.zeros((72,), dtype=np.complex64)
    with pytest.raises(ValueError):
        ex.extract(grid)


def test_extract_ndlrb_mismatch_raises():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=7, num_subframes=1, normal_cp=True)  # 84 sc, očekuje 72
    with pytest.raises(ValueError):
        ex.extract(grid)


def test_extract_reserved_mask_wrong_shape_raises():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    bad_mask = np.zeros((1, 1), dtype=bool)
    with pytest.raises(ValueError):
        ex.extract(grid, reserved_re_mask=bad_mask)


def test_extract_grid_too_few_subcarriers_raises():
    # num_subcarriers < 72 -> greška
    cfg = PBCHConfig(ndlrb=None, normal_cp=True, pbch_symbols_to_extract=10)
    ex = PBCHExtractor(cfg)
    grid = np.zeros((60, 20), dtype=np.complex64)
    with pytest.raises(ValueError):
        ex.extract(grid)


def test_extract_symbol_index_out_of_range_raises():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[999], pbch_symbols_to_extract=10)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    with pytest.raises(ValueError):
        ex.extract(grid)


def test_extract_not_enough_symbols_due_to_too_short_indices_raises():
    # samo 1 simbol * 72 RE = 72 < 240
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7], pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    with pytest.raises(ValueError):
        ex.extract(grid)


def test_extract_not_enough_symbols_due_to_mask_skipping_raises():
    # ima dovoljno RE u teoriji, ali maska preskoči sve -> nema dovoljno
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True)

    mask = np.ones(grid.shape, dtype=bool)  # sve rezervisano
    with pytest.raises(ValueError):
        ex.extract(grid, reserved_re_mask=mask)


# ============================================================
# 4) PBCHExtractor.extract happy paths + tačnost redoslijeda
# ============================================================

def test_extract_returns_exact_number_of_symbols_240_no_mask():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True)
    out = ex.extract(grid)
    assert out.shape == (240,)
    assert out.dtype == grid.dtype


def test_extract_returns_exact_number_of_symbols_960_no_mask_four_subframes():
    idx = pbch_symbol_indices_for_subframes(num_subframes=4, normal_cp=True, start_subframe=0)
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=idx, pbch_symbols_to_extract=960)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=4, normal_cp=True)
    out = ex.extract(grid)
    assert out.shape == (960,)


def test_extract_order_matches_expected_scan_for_first_72_in_symbol7():
    # kad tražimo 72 simbola, to treba biti tačno svih 72 iz prvog PBCH simbola (npr. 7)
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=72)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True, seed=123)

    out = ex.extract(grid)

    # očekivane pozicije (sc,7) redoslijedom RB0..RB5, offset0..11
    positions = expected_scan_positions_for_one_symbol(grid.shape[0], symbol_index=7, reserved_mask=None)
    expected = np.array([grid[sc, sym] for (sc, sym) in positions], dtype=grid.dtype)

    assert np.allclose(out, expected)


def test_extract_stops_early_when_needed_less_than_full_symbol():
    # tražimo 10 simbola => prvih 10 skeniranih u symbol=7
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=10)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True, seed=5)

    out = ex.extract(grid)
    assert out.shape == (10,)

    positions = expected_scan_positions_for_one_symbol(grid.shape[0], symbol_index=7, reserved_mask=None)
    expected = np.array([grid[sc, sym] for (sc, sym) in positions[:10]], dtype=grid.dtype)

    assert np.allclose(out, expected)


def test_extract_skips_reserved_re_and_still_returns_exact_count():
    # maskiraj npr. svaku drugu poziciju u symbol=7, ali imamo 4 simbola => i dalje skupimo 240
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True, seed=7)

    mask = np.zeros(grid.shape, dtype=bool)

    # rezervisi ~36 RE u symbol=7 (svaki drugi u centralnih 72)
    n_sc = grid.shape[0]
    k0 = central_k0(n_sc)
    toggles = []
    for rb in range(6):
        base = k0 + rb * 12
        for off in range(12):
            sc = base + off
            toggles.append(sc)
    toggles = np.array(toggles, dtype=int)
    mask[toggles[::2], 7] = True

    out = ex.extract(grid, reserved_re_mask=mask)
    assert out.shape == (240,)

    # provjeri da nijedan element iz "rezervisanih" nije uzet za symbol=7
    # (nije strogo dokazivo bez rekonstrukcije, ali možemo bar provjeriti prvih par koji bi bili rezervisani)
    # očekivano: prvi uzorak u out je grid[prvi_ne_rezervisani,7]
    positions = expected_scan_positions_for_one_symbol(n_sc, 7, reserved_mask=mask)
    assert np.allclose(out[0], grid[positions[0][0], positions[0][1]])


def test_extract_reserved_mask_dtype_int_is_accepted():
    cfg = PBCHConfig(ndlrb=6, normal_cp=True, pbch_symbol_indices=[7, 8, 9, 10], pbch_symbols_to_extract=240)
    ex = PBCHExtractor(cfg)
    grid = make_grid(ndlrb=6, num_subframes=1, normal_cp=True, seed=8)

    mask_int = np.zeros(grid.shape, dtype=np.uint8)  # biće cast na bool
    out = ex.extract(grid, reserved_re_mask=mask_int)
    assert out.shape == (240,)
