# tests/test_lte_rx_chain.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# ------------------------------------------------------------
# PATH FIX: da import radi kad pytest starta iz root-a projekta
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.LTERxChain import LTERxChain, RxResult


# ============================================================
# Helpers
# ============================================================

def rand_cpx(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return x.astype(np.complex64)


def make_min_rx(normal_cp: bool = True, **kw) -> LTERxChain:
    return LTERxChain(
        sample_rate_hz=1.92e6,
        ndlrb=6,
        normal_cp=normal_cp,
        **kw,
    )


# ============================================================
# 1) Init / validation tests
# ============================================================

def test_init_uses_expected_fs_when_sample_rate_none():
    rx = LTERxChain(sample_rate_hz=None, ndlrb=6, normal_cp=True)
    # should match ofdm_demod.sample_rate
    assert np.isclose(rx.sample_rate_hz, float(rx.ofdm_demod.sample_rate), atol=1e-3)


def test_init_sample_rate_mismatch_raises():
    # wrong Fs for given OFDM config (expected 1.92e6 for NDLRB=6 in your design)
    with pytest.raises(ValueError):
        LTERxChain(sample_rate_hz=2.0e6, ndlrb=6, normal_cp=True)




def test_offset_samples_to_pss_cp_start_normal_cp_positive():
    rx = make_min_rx(normal_cp=True)
    off = rx._offset_samples_to_pss_cp_start()
    assert isinstance(off, int)
    assert off > 0


def test_offset_samples_to_pss_cp_start_extended_cp_positive():
    rx = make_min_rx(normal_cp=False)
    off = rx._offset_samples_to_pss_cp_start()
    assert isinstance(off, int)
    assert off > 0


def test_samples_per_subframe_positive():
    rx = make_min_rx()
    spsf = rx._samples_per_subframe()
    assert isinstance(spsf, int)
    assert spsf > 0


# ============================================================
# 2) Gold / descramble tests
# ============================================================

def test_gold_sequence_invalid_pci_raises():
    with pytest.raises(ValueError):
        LTERxChain._gold_sequence_pbch(-1, 10)
    with pytest.raises(ValueError):
        LTERxChain._gold_sequence_pbch(504, 10)


def test_gold_sequence_length_and_binary():
    seq = LTERxChain._gold_sequence_pbch(0, 256)
    assert seq.shape == (256,)
    assert seq.dtype == np.uint8
    u = set(np.unique(seq).tolist())
    assert u.issubset({0, 1})


def test_descramble_is_xor_with_gold_and_is_involutive():
    rx = make_min_rx(pci=7)
    bits = np.random.default_rng(0).integers(0, 2, size=100, dtype=np.uint8)
    once = rx._descramble_pbch_bits(bits)
    twice = rx._descramble_pbch_bits(once)
    # XOR with same sequence twice => original
    assert np.array_equal(bits, twice)


# ============================================================
# 3) PBCH deinterleave tests
# ============================================================

def test_pbch_deinterleave_raises_on_wrong_len():
    rx = make_min_rx()
    with pytest.raises(ValueError):
        rx._pbch_deinterleave_120(np.zeros(119, dtype=np.uint8))


def test_pbch_deinterleave_output_length_and_binary():
    rx = make_min_rx()
    b = np.random.default_rng(1).integers(0, 2, size=120, dtype=np.uint8)
    out = rx._pbch_deinterleave_120(b)
    assert out.shape == (120,)
    assert out.dtype == np.uint8
    u = set(np.unique(out).tolist())
    assert u.issubset({0, 1})


def test_pbch_deinterleave_is_deterministic():
    rx = make_min_rx()
    b = np.random.default_rng(2).integers(0, 2, size=120, dtype=np.uint8)
    o1 = rx._pbch_deinterleave_120(b)
    o2 = rx._pbch_deinterleave_120(b.copy())
    assert np.array_equal(o1, o2)


# ============================================================
# 4) decode() unhappy paths via monkeypatch
#    (ovdje ciljamo grane gdje code vraća crc_ok=False i debug["error"])
# ============================================================

def test_decode_rejects_too_short_waveform():
    rx = make_min_rx()
    x = rand_cpx(10)  # min_num_samples=256
    with pytest.raises(Exception):
        # validate_rx_samples vjerovatno baca ValueError (ili sl.)
        rx.decode(x)


def test_decode_handles_ofdm_demod_exception(monkeypatch):
    rx = make_min_rx()

    # stub PSS sync outputs (minimalno validan)
    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    # make OFDM demod raise
    def boom(_):
        raise RuntimeError("demod boom")
    monkeypatch.setattr(rx.ofdm_demod, "demodulate", boom)

    # waveform long enough
    x = rand_cpx(5000, seed=1)
    res = rx.decode(x)
    assert isinstance(res, RxResult)
    assert res.crc_ok is False
    assert res.mib_bits is None
    assert "error" in res.debug
    assert "OFDM demod failed" in res.debug["error"]


def test_decode_handles_active_subcarrier_exception(monkeypatch):
    rx = make_min_rx()

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    # OFDM demod returns dummy grid_full
    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))

    # active extraction fails
    def boom(_):
        raise RuntimeError("active boom")
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", boom)

    x = rand_cpx(5000, seed=2)
    res = rx.decode(x)
    assert res.crc_ok is False
    assert res.mib_bits is None
    assert "Active subcarrier extraction failed" in res.debug.get("error", "")




def test_decode_handles_derate_match_exception(monkeypatch):
    rx = make_min_rx(pbch_spread_subframes=4)

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    # PBCH per-subframe extractor used in loop -> easiest: patch PBCHExtractor.extract class-wide via monkeypatching module attribute
    # but we can instead patch QPSK demapper to give E bits anyway, then force de_rm derate_match to fail.
    monkeypatch.setattr(rx.qpsk_demapper, "demap", lambda syms: np.zeros((1920,), dtype=np.uint8))
    monkeypatch.setattr(rx.de_rm, "derate_match", lambda bits, return_soft=False: (_ for _ in ()).throw(RuntimeError("derm boom")))

    x = rand_cpx(5000, seed=4)
    res = rx.decode(x)
    assert res.crc_ok is False
    assert res.mib_bits is None
    assert "De-rate-matching failed" in res.debug.get("error", "")


def test_decode_handles_deinterleave_exception(monkeypatch):
    rx = make_min_rx(pbch_spread_subframes=4)

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    monkeypatch.setattr(rx.qpsk_demapper, "demap", lambda syms: np.zeros((1920,), dtype=np.uint8))
    monkeypatch.setattr(rx.de_rm, "derate_match", lambda bits, return_soft=False: np.zeros((120,), dtype=np.uint8))

    # force deinterleave to fail
    monkeypatch.setattr(rx, "_pbch_deinterleave_120", lambda b: (_ for _ in ()).throw(RuntimeError("deint boom")))

    x = rand_cpx(5000, seed=5)
    res = rx.decode(x)
    assert res.crc_ok is False
    assert res.mib_bits is None
    assert "PBCH deinterleave failed" in res.debug.get("error", "")


def test_decode_handles_viterbi_returns_too_short(monkeypatch):
    rx = make_min_rx(pbch_spread_subframes=4)

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.qpsk_demapper, "demap", lambda syms: np.zeros((1920,), dtype=np.uint8))
    monkeypatch.setattr(rx.de_rm, "derate_match", lambda bits, return_soft=False: np.zeros((120,), dtype=np.uint8))
    monkeypatch.setattr(rx, "_pbch_deinterleave_120", lambda b: np.zeros((120,), dtype=np.uint8))

    # viterbi returns < 40 bits
    monkeypatch.setattr(rx.viterbi, "decode", lambda coded, tail_biting=True: np.zeros((10,), dtype=np.uint8))

    x = rand_cpx(5000, seed=6)
    res = rx.decode(x)
    assert res.crc_ok is False
    assert res.mib_bits is None
    assert "Viterbi nije vratio 40 bitova" in res.debug.get("error", "")


# ============================================================
# 5) decode() behavior / debug keys (mostly monkeypatched happy-ish)
# ============================================================

def test_decode_returns_RxResult_and_debug_contains_tau_expected(monkeypatch):
    rx = make_min_rx()

    # simple correlation with a clear peak near expected tau
    # expected tau depends on CP pattern; we place peak at rx._offset_samples_to_pss_cp_start()
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 300, 3000)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[1, tau_exp] = 10 + 0j  # make N_ID_2_hat = candidate index 1

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    # stub downstream to fail safely late (pbch extract) so we still get debug with tau_expected
    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    # Force PBCH fail (so we don't need full chain)
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=7)
    res = rx.decode(x)

    assert isinstance(res, RxResult)
    assert "tau_expected" in res.debug
    assert "tau_win" in res.debug
    assert "tau_search_mode" in res.debug
    assert res.debug["tau_expected"] == int(tau_exp)


def test_decode_sets_pci_used_for_pbch_equal_detected_nid(monkeypatch):
    rx = make_min_rx()
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)

    corr = np.zeros((3, L), dtype=np.complex64)
    corr[2, tau_exp] = 9 + 0j  # pick N_ID_2 candidate index 2

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=8)
    res = rx.decode(x)

    assert res.n_id_2_hat in (0, 1, 2)
    assert res.debug.get("pci_used_for_pbch") == res.n_id_2_hat
    assert rx.pci == res.n_id_2_hat


def test_decode_cfo_clamps_when_estimator_too_large(monkeypatch):
    rx = make_min_rx(cfo_limit_hz=100.0)  # tight clamp
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)

    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 9999.0)  # too large
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=9)
    res = rx.decode(x)

    assert res.debug.get("cfo_clamped") is True
    assert res.debug.get("cfo_hat_hz") == 0.0


def test_decode_does_not_clamp_when_cfo_within_limit(monkeypatch):
    rx = make_min_rx(cfo_limit_hz=5000.0)
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)

    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 200.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=10)
    res = rx.decode(x)

    assert res.debug.get("cfo_clamped") is False
    assert np.isclose(res.debug.get("cfo_hat_hz"), 200.0)


def test_decode_rms_norm_scale_present_when_enabled(monkeypatch):
    rx = make_min_rx(normalize_before_pss=True)

    # simple correlation peak
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=11)
    res = rx.decode(x)

    assert "rms_norm_scale" in res.debug
    # scale can be any float; just ensure it's number
    assert (res.debug["rms_norm_scale"] is None) or isinstance(res.debug["rms_norm_scale"], float)


def test_decode_rms_norm_scale_none_when_disabled(monkeypatch):
    rx = make_min_rx(normalize_before_pss=False)

    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=12)
    res = rx.decode(x)

    assert res.debug.get("rms_norm_scale") is None


# ============================================================
# 6) CRC behavior at the very end (monkeypatch whole tail)
# ============================================================

def test_decode_returns_mib_none_when_crc_fails(monkeypatch):
    rx = make_min_rx(pbch_spread_subframes=4)

    # drive to near-end quickly by patching internal stages
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[1, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    # PBCH extractor per subframe path is used; easiest is to bypass that by patching QPSK demap to give E bits
    # We'll patch the loop by patching PBCHExtractor.extract globally isn't trivial here, so:
    # patch qpsk_demapper.demap and upstream pbch_syms creation by patching np.concatenate through method isn't ideal.
    # Instead: patch extract_active_subcarriers to raise? no, we want reach CRC stage.
    #
    # We'll patch the PBCHExtractor class used in decode by patching receiver.resource_grid_extractor.PBCHExtractor.extract
    import receiver.resource_grid_extractor as rge
    monkeypatch.setattr(rge.PBCHExtractor, "extract", lambda self, grid, reserved_re_mask=None: np.zeros((240,), dtype=np.complex64))

    monkeypatch.setattr(rx.qpsk_demapper, "demap", lambda syms: np.zeros((1920,), dtype=np.uint8))
    monkeypatch.setattr(rx.de_rm, "derate_match", lambda bits, return_soft=False: np.zeros((120,), dtype=np.uint8))
    monkeypatch.setattr(rx, "_pbch_deinterleave_120", lambda b: np.zeros((120,), dtype=np.uint8))
    monkeypatch.setattr(rx.viterbi, "decode", lambda coded, tail_biting=True: np.zeros((40,), dtype=np.uint8))

    # CRC fails
    monkeypatch.setattr(rx.crc, "check", lambda decoded40: (decoded40[:24].astype(np.uint8), False))

    x = rand_cpx(6000, seed=13)
    res = rx.decode(x)
    assert res.crc_ok is False
    assert res.mib_bits is None


def test_decode_returns_mib_24_when_crc_ok(monkeypatch):
    rx = make_min_rx(pbch_spread_subframes=4)

    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    import receiver.resource_grid_extractor as rge
    monkeypatch.setattr(rge.PBCHExtractor, "extract", lambda self, grid, reserved_re_mask=None: np.zeros((240,), dtype=np.complex64))

    monkeypatch.setattr(rx.qpsk_demapper, "demap", lambda syms: np.zeros((1920,), dtype=np.uint8))
    monkeypatch.setattr(rx.de_rm, "derate_match", lambda bits, return_soft=False: np.zeros((120,), dtype=np.uint8))
    monkeypatch.setattr(rx, "_pbch_deinterleave_120", lambda b: np.zeros((120,), dtype=np.uint8))

    decoded40 = np.random.default_rng(0).integers(0, 2, size=40, dtype=np.uint8)
    monkeypatch.setattr(rx.viterbi, "decode", lambda coded, tail_biting=True: decoded40)

    monkeypatch.setattr(rx.crc, "check", lambda d40: (d40[:24].astype(np.uint8), True))

    x = rand_cpx(6000, seed=14)
    res = rx.decode(x)

    assert res.crc_ok is True
    assert res.mib_bits is not None
    assert np.asarray(res.mib_bits).shape == (24,)
    u = set(np.unique(res.mib_bits).tolist())
    assert u.issubset({0, 1})


# ============================================================
# 7) PSS correlation output shape handling
# ============================================================

def test_decode_accepts_1d_correlation_output(monkeypatch):
    rx = make_min_rx()
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr1d = np.zeros((L,), dtype=np.complex64)
    corr1d[tau_exp] = 10 + 0j  # should reshape to (1,L)

    # Need n_id_2_candidates length >= 1; assume it's [0,1,2] in your PSSSynchronizer
    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr1d)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=15)
    res = rx.decode(x)
    assert isinstance(res, RxResult)
    assert "tau_search_mode" in res.debug


def test_decode_fallback_global_argmax_when_window_invalid(monkeypatch):
    rx = make_min_rx()
    # Force tiny correlation length so that hi<=lo for window (signal too short)
    corr = np.zeros((3, 10), dtype=np.complex64)
    corr[1, 7] = 5 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    # fail later
    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: (_ for _ in ()).throw(RuntimeError("demod boom")))

    x = rand_cpx(6000, seed=16)
    res = rx.decode(x)
    assert res.debug.get("tau_search_mode") == "global_argmax_fallback"


# ============================================================
# 8) Configuration edge cases
# ============================================================

def test_init_with_pbch_spread_subframes_1_sets_extract_len_240():
    rx = make_min_rx(pbch_spread_subframes=1)
    assert rx.pbch_cfg.pbch_symbols_to_extract == 240


def test_init_with_pbch_spread_subframes_4_sets_extract_len_960():
    rx = make_min_rx(pbch_spread_subframes=4)
    assert rx.pbch_cfg.pbch_symbols_to_extract == 960


def test_enable_cfo_correction_false_sets_cfo_hat_none(monkeypatch):
    rx = make_min_rx(enable_cfo_correction=False)
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: (_ for _ in ()).throw(RuntimeError("demod boom")))

    x = rand_cpx(6000, seed=17)
    res = rx.decode(x)

    assert res.cfo_hat is None
    assert res.debug.get("cfo_hat_hz") is None


def test_enable_descrambling_flag_recorded_in_debug(monkeypatch):
    rx = make_min_rx(enable_descrambling=False, pbch_spread_subframes=4)

    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))

    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=18)
    res = rx.decode(x)

    assert res.debug.get("descrambling") is False


# ============================================================
# 9) Result container sanity
# ============================================================

def test_decode_result_fields_types_on_fail(monkeypatch):
    rx = make_min_rx()
    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: np.zeros((3, 4000), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: (_ for _ in ()).throw(RuntimeError("demod boom")))

    x = rand_cpx(6000, seed=19)
    res = rx.decode(x)

    assert isinstance(res, RxResult)
    assert isinstance(res.n_id_2_hat, int)
    assert isinstance(res.tau_hat, int)
    assert isinstance(res.crc_ok, bool)
    assert isinstance(res.debug, dict)


def test_decode_debug_has_grid_shapes_when_demod_ok(monkeypatch):
    rx = make_min_rx()
    tau_exp = rx._offset_samples_to_pss_cp_start()
    L = max(tau_exp + 200, 2500)
    corr = np.zeros((3, L), dtype=np.complex64)
    corr[0, tau_exp] = 9 + 0j

    monkeypatch.setattr(rx.pss_sync, "correlate", lambda sig: corr)
    monkeypatch.setattr(rx.pss_sync, "estimate_cfo", lambda sig, tau, nid: 0.0)
    monkeypatch.setattr(rx.pss_sync, "apply_cfo_correction", lambda sig, cfo: sig)

    monkeypatch.setattr(rx.ofdm_demod, "demodulate", lambda sig: np.zeros((128, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.ofdm_demod, "extract_active_subcarriers", lambda grid: np.zeros((72, 56), dtype=np.complex64))
    monkeypatch.setattr(rx.pbch_extractor, "extract", lambda grid, reserved_re_mask=None: (_ for _ in ()).throw(RuntimeError("pbch fail")))

    x = rand_cpx(6000, seed=20)
    res = rx.decode(x)

    assert "grid_full_shape" in res.debug
    assert "grid_active_shape" in res.debug
    assert tuple(res.debug["grid_full_shape"]) == (128, 56)
    assert tuple(res.debug["grid_active_shape"]) == (72, 56)
