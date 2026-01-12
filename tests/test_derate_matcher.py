import os
import sys
import numpy as np
import pytest

# ------------------------------------------------------------
# Ensure project root is on sys.path (radi kad pytest pokreće iz root-a ili drugih foldera)
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from receiver.de_rate_matching import DeRateMatcherPBCH


def _repeat_like_tx(bits_coded: np.ndarray, E: int, n_coded: int) -> np.ndarray:
    """
    Simulira ponavljanje iz TX (po indeksu i % n_coded), za proizvoljan E:
      bits_E[i] = bits_coded[i % n_coded]
    """
    bits_coded = np.asarray(bits_coded, dtype=np.uint8).ravel()
    idx = np.arange(E, dtype=np.int64) % int(n_coded)
    return bits_coded[idx].astype(np.uint8)


# ============================================================
# 1) INIT / VALIDACIJE
# ============================================================

def test_init_default_ok():
    drm = DeRateMatcherPBCH()
    assert drm.n_coded == 120


def test_init_invalid_n_coded_raises():
    with pytest.raises(ValueError):
        DeRateMatcherPBCH(n_coded=0)
    with pytest.raises(ValueError):
        DeRateMatcherPBCH(n_coded=-7)


def test_derate_match_requires_min_length():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.zeros(119, dtype=np.uint8)
    with pytest.raises(ValueError):
        drm.derate_match(x)


# ============================================================
# 2) OSNOVNA SVOJSTVA IZLAZA
# ============================================================

def test_output_length_is_120_hard():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.random.randint(0, 2, size=1920, dtype=np.uint8)
    y = drm.derate_match(x, return_soft=False)
    assert y.shape == (120,)
    assert y.dtype == np.uint8


def test_output_length_is_120_soft():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.random.randint(0, 2, size=1920, dtype=np.uint8)
    y = drm.derate_match(x, return_soft=True)
    assert y.shape == (120,)
    assert np.issubdtype(y.dtype, np.floating)


def test_output_bits_are_binary_for_hard():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.random.randint(0, 2, size=1920, dtype=np.uint8)
    y = drm.derate_match(x, return_soft=False)
    assert set(np.unique(y)).issubset({0, 1})


# ============================================================
# 3) ROUNDTRIP / IDENTITET
# ============================================================

def test_identity_when_E_equals_120():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    out = drm.derate_match(bits120)
    np.testing.assert_array_equal(out, bits120)


def test_roundtrip_normal_cp_1920_exact_tile_recovers_bits():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16)
    out = drm.derate_match(bits1920)
    np.testing.assert_array_equal(out, bits120)


def test_roundtrip_extended_cp_1728_mod_repetition_recovers_bits():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1728 = _repeat_like_tx(bits120, E=1728, n_coded=120)
    out = drm.derate_match(bits1728)
    np.testing.assert_array_equal(out, bits120)


# ============================================================
# 4) PONAVLJANJE / COUNTS PROVJERE
# ============================================================

def test_counts_distribution_for_1728_first_48_have_extra_repeat():
    E = 1728
    idx = np.arange(E) % 120
    counts = np.bincount(idx, minlength=120)
    assert np.all(counts[:48] == 15)
    assert np.all(counts[48:] == 14)


def test_soft_magnitude_for_clean_1920_is_16():
    # Za idealno ponavljanje: abs(soft)=16 na svim pozicijama
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16)
    soft = drm.derate_match(bits1920, return_soft=True)
    np.testing.assert_array_equal(np.abs(soft), np.full(120, 16.0))


def test_soft_magnitude_for_clean_1728_is_15_for_first48_else_14():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1728 = _repeat_like_tx(bits120, E=1728, n_coded=120)
    soft = drm.derate_match(bits1728, return_soft=True)
    assert np.all(np.abs(soft[:48]) == 15.0)
    assert np.all(np.abs(soft[48:]) == 14.0)


# ============================================================
# 5) MAJORITY VOTE / ROBUSTNOST NA GREŠKE
# ============================================================

def test_majority_vote_corrects_sparse_errors_normal_cp():
    rng = np.random.default_rng(0)
    drm = DeRateMatcherPBCH(n_coded=120)

    bits120 = rng.integers(0, 2, size=120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16)

    # Flipuj po 1 kopiju za prvih 30 pozicija (16 kopija -> i dalje većina tačna)
    for j in range(30):
        k = int(j + 120 * rng.integers(0, 16))
        bits1920[k] ^= 1

    out = drm.derate_match(bits1920)
    np.testing.assert_array_equal(out, bits120)


def test_hard_decision_matches_soft_sign_rule():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16)

    soft = drm.derate_match(bits1920, return_soft=True)
    hard = drm.derate_match(bits1920, return_soft=False)

    # soft < 0 -> hard=1, soft > 0 -> hard=0 (na idealnom slučaju nikad 0)
    np.testing.assert_array_equal(hard, (soft < 0).astype(np.uint8))


def test_tie_break_when_summed_zero_outputs_zero():
    # Napravi E=240 tako da je za svaku poziciju jednom 0 (+1) i jednom 1 (-1) => suma 0
    drm = DeRateMatcherPBCH(n_coded=120)
    first = np.zeros(120, dtype=np.uint8)
    second = np.ones(120, dtype=np.uint8)
    x = np.concatenate([first, second])  # E=240

    out = drm.derate_match(x, return_soft=False)
    np.testing.assert_array_equal(out, np.zeros(120, dtype=np.uint8))


# ============================================================
# 6) TIPOVI ULAZA / SHAPE ROBUST
# ============================================================

def test_accepts_python_list_and_tuple_inputs():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    x = np.tile(bits120, 16)

    out_list = drm.derate_match(x.tolist())
    out_tuple = drm.derate_match(tuple(x.tolist()))
    np.testing.assert_array_equal(out_list, bits120)
    np.testing.assert_array_equal(out_tuple, bits120)


def test_accepts_2d_input_is_flattened():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    x = np.tile(bits120, 16).reshape(64, 30)  # 2D shape, ukupno 1920
    out = drm.derate_match(x)
    np.testing.assert_array_equal(out, bits120)


def test_nonbinary_int_inputs_use_lsb_only():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)
    bits1920 = np.tile(bits120, 16).astype(np.int32)

    # 0->2, 1->3 (LSB ostaje 0/1)
    x = np.where(bits1920 == 0, 2, 3).astype(np.int32)
    out = drm.derate_match(x, return_soft=False)
    np.testing.assert_array_equal(out, bits120)


# ============================================================
# 7) FLOAT “SOFT” ULAZI
# ============================================================

def test_float_inputs_thresholding_behavior_clean():
    drm = DeRateMatcherPBCH(n_coded=120)
    bits120 = np.random.randint(0, 2, size=120, dtype=np.uint8)

    x = np.tile(bits120, 16).astype(np.float64)
    # 0->0.1, 1->0.9 (bez šuma)
    x = np.where(x == 0.0, 0.1, 0.9)

    out = drm.derate_match(x, return_soft=False)
    np.testing.assert_array_equal(out, bits120)


def test_float_all_0p5_produces_zero_evidence_and_hard_zero():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.full(1920, 0.5, dtype=np.float64)  # evidence=0 svuda -> soft=0 -> hard=0
    out = drm.derate_match(x, return_soft=False)
    np.testing.assert_array_equal(out, np.zeros(120, dtype=np.uint8))


# ============================================================
# 8) EKSTREMI / SANITY
# ============================================================

def test_all_zeros_input_returns_zeros():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.zeros(1920, dtype=np.uint8)
    out = drm.derate_match(x)
    np.testing.assert_array_equal(out, np.zeros(120, dtype=np.uint8))


def test_all_ones_input_returns_ones():
    drm = DeRateMatcherPBCH(n_coded=120)
    x = np.ones(1920, dtype=np.uint8)
    out = drm.derate_match(x)
    np.testing.assert_array_equal(out, np.ones(120, dtype=np.uint8))


def test_custom_n_coded_roundtrip_general_case():
    # Provjera da radi i za druge n_coded (nije samo hardcode na 120)
    n_coded = 10
    drm = DeRateMatcherPBCH(n_coded=n_coded)

    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    E = 25
    x = _repeat_like_tx(bits, E=E, n_coded=n_coded)
    out = drm.derate_match(x)
    np.testing.assert_array_equal(out, bits)
