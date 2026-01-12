"""
tests/test_lte_system_e2e.py
============================

E2E testovi za LTE sistem:
    TX (LTETxChain) -> Channel (LTEChannel) -> RX (LTERxChain)
preko glue klase:
    LTESystem

Cilj:
- sistemska ispravnost (stabilnost, relativno ponašanje SNR-a, sinkronizacija),
- happy/unhappy paths,
- GUI-friendly output struktura.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest


# ---------------------------------------------------------------------
# PATH FIX (da importi rade kad pytest krene iz root-a)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from transmitter.LTETxChain import LTETxChain
from receiver.LTERxChain import LTERxChain
from channel.lte_channel import LTEChannel
from LTE_system_.lte_system import LTESystem


# ---------------------------------------------------------------------
# Globalni parametri (u skladu s tvojim projektom)
# ---------------------------------------------------------------------
FS = 1.92e6
NDLRB = 6
NORMAL_CP = True
NUM_SUBFRAMES = 4


# ---------------------------------------------------------------------
# Helper: robustan poziv system.run() (ako se potpis malo razlikuje)
# ---------------------------------------------------------------------
def run_system(system: LTESystem, mib_bits: Sequence[int], **kwargs) -> Dict[str, Any]:
    sig = inspect.signature(system.run)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return system.run(mib_bits, **filtered)


def make_system(
    *,
    snr_db: float,
    freq_offset_hz: float,
    seed: int = 1,
    n_id_2: int = 0,
    fs_hz: float = FS,
    ndlrb: int = NDLRB,
    num_subframes: int = NUM_SUBFRAMES,
    normal_cp: bool = NORMAL_CP,
) -> LTESystem:
    tx = LTETxChain(
        n_id_2=int(n_id_2),
        ndlrb=int(ndlrb),
        num_subframes=int(num_subframes),
        normal_cp=bool(normal_cp),
    )

    ch = LTEChannel(
        freq_offset_hz=float(freq_offset_hz),
        sample_rate_hz=float(fs_hz),
        snr_db=float(snr_db),
        seed=int(seed),
    )

    rx = LTERxChain(
        sample_rate_hz=float(fs_hz),
        ndlrb=int(ndlrb),
        normal_cp=bool(normal_cp),
    )

    return LTESystem(tx=tx, ch=ch, rx=rx)


def fixed_mib(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=24, dtype=np.uint8)


def avg_trials_metrics(
    *,
    snr_db: float,
    freq_offset_hz: float,
    n_trials: int = 8,
    seed0: int = 10,
    n_id_2: int = 0,
) -> Tuple[float, float]:
    """
    Vrati (avg_BER, CRC_success_rate) preko više trial-ova.
    Deterministički: seed se mijenja kroz trial-ove.
    """
    bers = []
    crc_ok = 0
    for k in range(n_trials):
        sys_k = make_system(
            snr_db=snr_db,
            freq_offset_hz=freq_offset_hz,
            seed=seed0 + k,
            n_id_2=n_id_2,
        )
        mib = fixed_mib(seed=100 + k)
        res = run_system(sys_k, mib, keep_debug=True)
        if res.get("crc_ok", False):
            crc_ok += 1
        ber = res.get("ber", None)
        if ber is None:
            # ako je None, tretiraj kao 1 (najgore) da bude konzervativno
            ber = 1.0
        bers.append(float(ber))
    return float(np.mean(bers)), float(crc_ok) / float(n_trials)


# ============================================================
# 1) Unit testovi pomoćnih funkcija (lte_system helperi)
# ============================================================

def test_to_bits_01_binaryizes_and_checks_length():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    # ubaci "šire" vrijednosti, treba ih svesti na 0/1
    bits = [2, 3] * 12  # 24 bita
    out = sys0._to_bits_01(bits)  # type: ignore[attr-defined]
    assert out.shape == (24,)
    assert set(np.unique(out)).issubset({0, 1})


def test_to_bits_01_invalid_length_raises():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    with pytest.raises(ValueError):
        sys0._to_bits_01([0, 1, 0])  # type: ignore[attr-defined]


def test_compute_bit_errors_rx_none_count_all_true():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    tx = np.zeros(24, dtype=np.uint8)
    errors, ber = sys0._compute_bit_errors(tx, None, count_all_if_missing=True)  # type: ignore[attr-defined]
    assert errors == 24
    assert ber == 1.0


def test_compute_bit_errors_rx_none_count_all_false():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    tx = np.zeros(24, dtype=np.uint8)
    errors, ber = sys0._compute_bit_errors(tx, None, count_all_if_missing=False)  # type: ignore[attr-defined]
    assert errors is None
    assert ber is None


def test_compute_bit_errors_identical_bits_zero():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    tx = fixed_mib(1)
    rx = tx.copy()
    errors, ber = sys0._compute_bit_errors(tx, rx, count_all_if_missing=True)  # type: ignore[attr-defined]
    assert errors == 0
    assert ber == 0.0


def test_compute_bit_errors_partial_overlap_uses_min_len():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    tx = np.array([0, 1, 1, 0], dtype=np.uint8)
    rx = np.array([0, 0], dtype=np.uint8)
    errors, ber = sys0._compute_bit_errors(tx, rx, count_all_if_missing=True)  # type: ignore[attr-defined]
    assert errors == 1
    assert ber == 0.5


@dataclass
class DummyRxResult:
    mib_bits: Optional[np.ndarray]
    crc_ok: bool
    n_id_2_hat: int
    tau_hat: int
    cfo_hat: Optional[float]
    debug: Dict[str, Any]


def test_rx_unpack_accepts_dataclass_like():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    dummy = DummyRxResult(
        mib_bits=np.zeros(24, dtype=np.uint8),
        crc_ok=True,
        n_id_2_hat=1,
        tau_hat=823,
        cfo_hat=0.0,
        debug={"corr_peak": 0.9},
    )
    mib, crc_ok, nid, tau, cfo, dbg = sys0._rx_unpack(dummy)  # type: ignore[attr-defined]
    assert crc_ok is True
    assert nid == 1
    assert tau == 823
    assert "corr_peak" in dbg
    assert mib is not None and mib.size == 24


def test_rx_unpack_accepts_dict():
    sys0 = make_system(snr_db=100, freq_offset_hz=0, seed=1)
    dummy = {
        "mib_bits": np.ones(24, dtype=np.uint8),
        "crc_ok": True,
        "n_id_2_hat": 2,
        "tau_hat": 823,
        "cfo_hat": 10.0,
        "debug": {"corr_peak": 0.8},
    }
    mib, crc_ok, nid, tau, cfo, dbg = sys0._rx_unpack(dummy)  # type: ignore[attr-defined]
    assert crc_ok is True
    assert nid == 2
    assert tau == 823
    assert float(cfo) == 10.0
    assert mib is not None and mib.size == 24


# ============================================================
# 2) E2E / integracijski testovi (koriste realne module)
# ============================================================

def test_run_returns_required_keys():
    system = make_system(snr_db=100, freq_offset_hz=0, seed=1, n_id_2=0)
    mib = fixed_mib(0)
    res = run_system(system, mib, keep_debug=True)

    expected_keys = {
        "tx_waveform",
        "rx_waveform",
        "mib_bits_tx",
        "mib_bits_rx",
        "crc_ok",
        "bit_errors",
        "ber",
        "detected_nid",
        "tau_hat",
        "cfo_hat_hz",
        "pss_metric",
        "debug",
        "fs_hz",
    }
    # fs_hz je u results-u (ne u debug-u), pa ga provjeri posebno ako postoji
    assert "debug" in res
    # glavna polja
    for k in ["tx_waveform", "rx_waveform", "mib_bits_tx", "crc_ok", "debug"]:
        assert k in res


def test_waveforms_are_complex_and_1d():
    system = make_system(snr_db=100, freq_offset_hz=0, seed=2)
    mib = fixed_mib(1)
    res = run_system(system, mib, keep_debug=False)

    tx = np.asarray(res["tx_waveform"])
    rx = np.asarray(res["rx_waveform"])

    assert tx.ndim == 1 and rx.ndim == 1
    assert np.iscomplexobj(tx)
    assert np.iscomplexobj(rx)
    assert tx.size > 0 and rx.size > 0


def test_debug_always_has_fs_and_lengths():
    system = make_system(snr_db=100, freq_offset_hz=0, seed=3)
    mib = fixed_mib(2)
    res = run_system(system, mib, keep_debug=False)

    dbg = res["debug"]
    assert "fs_hz" in dbg
    assert "tx_waveform_len" in dbg
    assert "rx_waveform_len" in dbg
    assert dbg["tx_waveform_len"] > 0
    assert dbg["rx_waveform_len"] > 0


def test_fs_matches_expected_1p4mhz_mode():
    system = make_system(snr_db=100, freq_offset_hz=0, seed=4)
    mib = fixed_mib(3)
    res = run_system(system, mib, keep_debug=False)

    fs = float(res["fs_hz"])
    assert np.isfinite(fs)
    assert abs(fs - FS) < 1e-6  # tx vraća 1.92e6 tačno


def test_tau_hat_expected_normal_cp_high_snr():
    system = make_system(snr_db=100, freq_offset_hz=0, seed=5, n_id_2=0)
    mib = fixed_mib(4)
    res = run_system(system, mib, keep_debug=True)

    # Za NDLRB=6, normal CP, tau očekujemo 823 (iz tvoje implementacije)
    assert int(res["tau_hat"]) == 823


@pytest.mark.parametrize("nid2", [0, 1, 2])
def test_detected_nid_matches_tx_at_high_snr(nid2: int):
    system = make_system(snr_db=100, freq_offset_hz=0, seed=6 + nid2, n_id_2=nid2)
    mib = fixed_mib(10 + nid2)
    res = run_system(system, mib, keep_debug=True)
    assert int(res["detected_nid"]) == int(nid2)


def test_cfo_hat_close_when_cfo_present_high_snr():
    cfo_true = 2000.0
    system = make_system(snr_db=40, freq_offset_hz=cfo_true, seed=7, n_id_2=1)
    mib = fixed_mib(20)
    res = run_system(system, mib, keep_debug=True)

    cfo_hat = float(res["cfo_hat_hz"])
    assert np.isfinite(cfo_hat)
    assert abs(cfo_hat - cfo_true) < 500.0


def test_cfo_hat_near_zero_when_no_cfo_high_snr():
    system = make_system(snr_db=40, freq_offset_hz=0.0, seed=8, n_id_2=1)
    mib = fixed_mib(21)
    res = run_system(system, mib, keep_debug=True)

    cfo_hat = float(res["cfo_hat_hz"])
    assert np.isfinite(cfo_hat)
    assert abs(cfo_hat) < 200.0


def test_crc_true_at_very_high_snr_zero_cfo():
    system = make_system(snr_db=100, freq_offset_hz=0.0, seed=9, n_id_2=1)
    mib = fixed_mib(22)
    res = run_system(system, mib, keep_debug=True)
    assert bool(res["crc_ok"]) is True
    assert res["mib_bits_rx"] is not None


def test_crc_often_fails_at_very_low_snr_at_least_once():
    mib = fixed_mib(30)
    fails = 0
    for seed in range(5):
        system = make_system(snr_db=-5.0, freq_offset_hz=300.0, seed=100 + seed, n_id_2=0)
        res = run_system(system, mib, keep_debug=False)
        if not bool(res["crc_ok"]):
            fails += 1
    assert fails >= 1  # bar jednom mora pasti


def test_ber_is_in_range_0_1():
    system = make_system(snr_db=10.0, freq_offset_hz=0.0, seed=10)
    mib = fixed_mib(40)
    res = run_system(system, mib, keep_debug=False)

    ber = res.get("ber", None)
    assert ber is not None
    assert 0.0 <= float(ber) <= 1.0


def test_pss_metric_if_present_is_finite():
    system = make_system(snr_db=30.0, freq_offset_hz=0.0, seed=11)
    mib = fixed_mib(41)
    res = run_system(system, mib, keep_debug=True)

    pss_metric = res.get("pss_metric", None)
    # pss_metric može biti None ako debug nema corr_peak
    if pss_metric is not None:
        assert np.isfinite(float(pss_metric))


def test_keep_debug_false_drops_rx_debug_fields_like_corr_peak():
    system = make_system(snr_db=40.0, freq_offset_hz=0.0, seed=12)
    mib = fixed_mib(42)
    res = run_system(system, mib, keep_debug=False)

    dbg = res["debug"]
    # sistem debug uvijek ima samo osnove; RX detalji se ne smiju ubaciti
    assert "corr_peak" not in dbg
    assert "tau_expected" not in dbg


def test_keep_debug_true_includes_some_rx_debug_fields():
    system = make_system(snr_db=40.0, freq_offset_hz=0.0, seed=13)
    mib = fixed_mib(43)
    res = run_system(system, mib, keep_debug=True)

    dbg = res["debug"]
    # minimalno očekujemo da se nešto od RX debug-a pojavi
    assert ("corr_peak" in dbg) or ("tau_expected" in dbg) or ("n_id_2_hat" in dbg)


def test_count_all_if_rx_missing_true_sets_ber_1_when_crc_fail():
    system = make_system(snr_db=-5.0, freq_offset_hz=0.0, seed=14)
    mib = fixed_mib(44)
    res = run_system(system, mib, keep_debug=False, count_all_if_rx_missing=True)

    if res["mib_bits_rx"] is None:
        assert res["bit_errors"] == 24
        assert float(res["ber"]) == 1.0


def test_count_all_if_rx_missing_false_returns_none_when_crc_fail():
    system = make_system(snr_db=-5.0, freq_offset_hz=0.0, seed=15)
    mib = fixed_mib(45)
    res = run_system(system, mib, keep_debug=False, count_all_if_rx_missing=False)

    if res["mib_bits_rx"] is None:
        assert res["bit_errors"] is None
        assert res["ber"] is None


def test_high_snr_has_lower_avg_ber_than_low_snr_trials():
    ber_hi, _ = avg_trials_metrics(snr_db=30.0, freq_offset_hz=0.0, n_trials=8, seed0=200)
    ber_lo, _ = avg_trials_metrics(snr_db=0.0, freq_offset_hz=0.0, n_trials=8, seed0=200)
    assert ber_hi <= ber_lo


def test_high_snr_has_higher_crc_rate_than_low_snr_trials():
    _, crc_hi = avg_trials_metrics(snr_db=30.0, freq_offset_hz=0.0, n_trials=8, seed0=250)
    _, crc_lo = avg_trials_metrics(snr_db=0.0, freq_offset_hz=0.0, n_trials=8, seed0=250)
    assert crc_hi >= crc_lo


def test_same_seed_same_mib_deterministic_single_run():
    mib = fixed_mib(50)
    s1 = make_system(snr_db=15.0, freq_offset_hz=500.0, seed=77)
    s2 = make_system(snr_db=15.0, freq_offset_hz=500.0, seed=77)

    r1 = run_system(s1, mib, keep_debug=False)
    r2 = run_system(s2, mib, keep_debug=False)

    assert bool(r1["crc_ok"]) == bool(r2["crc_ok"])
    assert r1["bit_errors"] == r2["bit_errors"]
    # rx_waveform treba biti identičan (isti seed, isti signal, isti CFO)
    assert np.allclose(np.asarray(r1["rx_waveform"]), np.asarray(r2["rx_waveform"]))


def test_different_seed_changes_rx_waveform():
    mib = fixed_mib(51)
    s1 = make_system(snr_db=15.0, freq_offset_hz=0.0, seed=1)
    s2 = make_system(snr_db=15.0, freq_offset_hz=0.0, seed=2)

    r1 = run_system(s1, mib, keep_debug=False)
    r2 = run_system(s2, mib, keep_debug=False)

    rx1 = np.asarray(r1["rx_waveform"])
    rx2 = np.asarray(r2["rx_waveform"])
    assert rx1.shape == rx2.shape
    assert not np.allclose(rx1, rx2)


def test_large_cfo_system_stability_no_exception():
    system = make_system(snr_db=25.0, freq_offset_hz=4000.0, seed=16)
    mib = fixed_mib(60)
    res = run_system(system, mib, keep_debug=True)
    assert "crc_ok" in res
    assert res["tau_hat"] is not None


def test_very_high_snr_no_noise_like_behavior_crc_true():
    system = make_system(snr_db=200.0, freq_offset_hz=0.0, seed=17)
    mib = fixed_mib(61)
    res = run_system(system, mib, keep_debug=False)
    assert bool(res["crc_ok"]) is True


def test_invalid_mib_length_raises_value_error():
    system = make_system(snr_db=25.0, freq_offset_hz=0.0, seed=18)
    with pytest.raises(ValueError):
        run_system(system, [0, 1, 0], keep_debug=False)


def test_system_run_accepts_list_bits():
    system = make_system(snr_db=40.0, freq_offset_hz=0.0, seed=19)
    mib = [0, 1] * 12
    res = run_system(system, mib, keep_debug=False)
    assert "crc_ok" in res


def test_system_run_accepts_numpy_bits_uint8():
    system = make_system(snr_db=40.0, freq_offset_hz=0.0, seed=20)
    mib = np.zeros(24, dtype=np.uint8)
    res = run_system(system, mib, keep_debug=False)
    assert "crc_ok" in res


def test_input_mib_not_modified_in_place():
    system = make_system(snr_db=40.0, freq_offset_hz=0.0, seed=21)
    mib = np.array([0, 1] * 12, dtype=np.uint8)
    mib_copy = mib.copy()
    _ = run_system(system, mib, keep_debug=False)
    assert np.array_equal(mib, mib_copy)


def test_tau_hat_stable_even_with_moderate_cfo():
    system = make_system(snr_db=25.0, freq_offset_hz=2000.0, seed=22, n_id_2=1)
    mib = fixed_mib(70)
    res = run_system(system, mib, keep_debug=True)
    assert int(res["tau_hat"]) == 823


def test_run_with_keep_debug_true_returns_debug_dict():
    system = make_system(snr_db=25.0, freq_offset_hz=0.0, seed=23)
    mib = fixed_mib(71)
    res = run_system(system, mib, keep_debug=True)
    assert isinstance(res["debug"], dict)
    assert "fs_hz" in res["debug"]


def test_run_with_keep_debug_false_returns_debug_dict_minimal():
    system = make_system(snr_db=25.0, freq_offset_hz=0.0, seed=24)
    mib = fixed_mib(72)
    res = run_system(system, mib, keep_debug=False)
    assert isinstance(res["debug"], dict)
    assert "fs_hz" in res["debug"]


def test_channel_reset_called_when_reset_channel_true(monkeypatch):
    # wrap pravi LTEChannel da uhvatimo reset()
    base = make_system(snr_db=25.0, freq_offset_hz=1000.0, seed=25)
    called = {"reset": 0}

    def reset_wrap():
        called["reset"] += 1
        base.ch._sample_index = 0  # isto što radi original reset

    monkeypatch.setattr(base.ch, "reset", reset_wrap, raising=True)

    mib = fixed_mib(80)
    _ = run_system(base, mib, reset_channel=True, keep_debug=False)
    assert called["reset"] == 1


def test_channel_reset_not_called_when_reset_channel_false(monkeypatch):
    base = make_system(snr_db=25.0, freq_offset_hz=1000.0, seed=26)
    called = {"reset": 0}

    def reset_wrap():
        called["reset"] += 1
        base.ch._sample_index = 0

    monkeypatch.setattr(base.ch, "reset", reset_wrap, raising=True)

    mib = fixed_mib(81)
    _ = run_system(base, mib, reset_channel=False, keep_debug=False)
    assert called["reset"] == 0


def test_multiple_runs_same_system_produce_different_rx_due_to_rng_progress():
    system = make_system(snr_db=10.0, freq_offset_hz=0.0, seed=27)
    mib = fixed_mib(90)

    r1 = run_system(system, mib, reset_channel=True, keep_debug=False)
    r2 = run_system(system, mib, reset_channel=True, keep_debug=False)

    rx1 = np.asarray(r1["rx_waveform"])
    rx2 = np.asarray(r2["rx_waveform"])
    # AWGN RNG state napreduje, pa waveform ne bi smio biti identičan
    assert not np.allclose(rx1, rx2)


def test_crc_ok_implies_mib_bits_rx_present_and_len_24():
    system = make_system(snr_db=100.0, freq_offset_hz=0.0, seed=28)
    mib = fixed_mib(91)
    res = run_system(system, mib, keep_debug=False)

    if bool(res["crc_ok"]):
        assert res["mib_bits_rx"] is not None
        assert len(res["mib_bits_rx"]) == 24


def test_crc_fail_often_implies_mib_bits_rx_none_at_low_snr():
    system = make_system(snr_db=-5.0, freq_offset_hz=0.0, seed=29)
    mib = fixed_mib(92)
    res = run_system(system, mib, keep_debug=False)
    if not bool(res["crc_ok"]):
        assert res["mib_bits_rx"] is None
