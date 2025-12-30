# tests/test_pss_synchronizer.py
from __future__ import annotations

import numpy as np
import pytest


from receiver.pss_sync import PSSSynchronizer
import receiver.pss_sync as pss_mod



def _make_sync_with_templates(
    *,
    fs: float = 1_000_000.0,
    candidates: tuple[int, ...] = (0, 1, 2),
    templates: dict[int, np.ndarray] | None = None,
) -> PSSSynchronizer:
    """
    Kreira PSSSynchronizer bez pozivanja __init__ (da ne gradi LTE template-e),
    i ručno setuje atribute + template-e.
    """
    sync = object.__new__(PSSSynchronizer)
    sync.sample_rate_hz = float(fs)
    sync.n_id_2_candidates = tuple(candidates)
    sync.ndlrb = 6
    sync.normal_cp = True
    if templates is None:
        # default: 3 template-a iste dužine
        rng = np.random.default_rng(0)
        templates = {nid: (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex128)
                     for nid in candidates}
    sync._templates = templates
    return sync


# -----------------------------
# HAPPY PATHS
# -----------------------------

def test_correlate_returns_expected_shape():
    sync = _make_sync_with_templates()
    L = next(iter(sync._templates.values())).size
    rx = (np.random.default_rng(1).standard_normal(L + 10) + 1j * np.random.default_rng(2).standard_normal(L + 10)).astype(np.complex128)

    corr = sync.correlate(rx)
    assert corr.shape == (3, (L + 10) - L + 1)


def test_correlate_peak_close_to_1_for_perfect_match():
    # rx == template -> normalizovana korelacija na tau=0 treba biti ~1
    t = (np.random.default_rng(3).standard_normal(64) + 1j * np.random.default_rng(4).standard_normal(64)).astype(np.complex128)
    sync = _make_sync_with_templates(templates={0: t, 1: t, 2: t})
    corr = sync.correlate(t)
    # bilo koji kandidat ima magnitude ~1 na tau=0
    assert np.allclose(np.abs(corr[:, 0]), 1.0, atol=1e-10)


def test_estimate_timing_finds_correct_tau_and_nid_for_embedded_template():
    rng = np.random.default_rng(5)
    L = 80
    t0 = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex128)
    t1 = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex128)
    t2 = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex128)

    templates = {0: t0, 1: t1, 2: t2}
    sync = _make_sync_with_templates(templates=templates)

    offset = 37
    rx = (0.01 * (rng.standard_normal(L + offset + 20) + 1j * rng.standard_normal(L + offset + 20))).astype(np.complex128)
    rx[offset:offset + L] += t1  # ubaci nid=1

    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)

    assert nid_hat == 1
    assert tau_hat == offset


def test_estimate_timing_returns_expected_for_known_corr_matrix():
    sync = _make_sync_with_templates()
    corr = np.zeros((3, 20), dtype=np.complex128)
    corr[2, 7] = 10 + 1j
    corr[1, 3] = 9 + 0j

    tau_hat, nid_hat = sync.estimate_timing(corr)
    assert tau_hat == 7
    assert nid_hat == 2


def test_estimate_cfo_zero_for_no_rotation():
    rng = np.random.default_rng(6)
    t = (rng.standard_normal(128) + 1j * rng.standard_normal(128)).astype(np.complex128)
    sync = _make_sync_with_templates(fs=1_000_000.0, templates={0: t, 1: t, 2: t})

    rx = t.copy()
    cfo_hat = sync.estimate_cfo(rx, tau_hat=0, n_id_2=1)
    assert abs(cfo_hat) < 1e-9


def test_estimate_cfo_recovers_known_cfo():
    fs = 1_000_000.0
    cfo_true = 5000.0  # Hz (unutar ±fs/2)
    rng = np.random.default_rng(7)
    L = 256
    t = (rng.standard_normal(L) + 1j * rng.standard_normal(L)).astype(np.complex128)

    sync = _make_sync_with_templates(fs=fs, templates={0: t, 1: t, 2: t})

    n = np.arange(L, dtype=np.float64)
    rx = t * np.exp(1j * 2.0 * np.pi * cfo_true * n / fs)
    cfo_hat = sync.estimate_cfo(rx, tau_hat=0, n_id_2=1)

    # treba biti jako blizu (numerički)
    assert abs(cfo_hat - cfo_true) < 1e-6


def test_apply_cfo_correction_uses_negative_cfo_and_returns_apply_output(monkeypatch):
    sync = _make_sync_with_templates(fs=1_920_000.0)
    rx = (np.ones(10) + 1j * np.zeros(10)).astype(np.complex128)
    cfo_hat = 1234.0

    captured = {}

    class FakeFO:
        def __init__(self, freq_offset_hz, sample_rate_hz):
            captured["freq_offset_hz"] = freq_offset_hz
            captured["sample_rate_hz"] = sample_rate_hz

        def apply(self, x):
            return x * (2 + 0j)

    monkeypatch.setattr(pss_mod, "FrequencyOffset", FakeFO)

    out = sync.apply_cfo_correction(rx, cfo_hat)

    assert captured["freq_offset_hz"] == -cfo_hat
    assert captured["sample_rate_hz"] == sync.sample_rate_hz
    assert np.allclose(out, rx * (2 + 0j))


def test_correlate_accepts_real_input_and_outputs_complex():
    sync = _make_sync_with_templates()
    L = next(iter(sync._templates.values())).size
    rx = np.ones(L + 5, dtype=np.float64)

    corr = sync.correlate(rx)
    assert np.iscomplexobj(corr)
    assert corr.shape[1] == 6


def test_correlate_is_finite_for_all_zero_rx():
    sync = _make_sync_with_templates()
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L + 10, dtype=np.complex128)

    corr = sync.correlate(rx)
    assert np.all(np.isfinite(corr))
    assert np.allclose(corr, 0.0)


def test_correlate_custom_candidate_count():
    rng = np.random.default_rng(8)
    t0 = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex128)
    t2 = (rng.standard_normal(32) + 1j * rng.standard_normal(32)).astype(np.complex128)
    sync = _make_sync_with_templates(
        candidates=(0, 2),
        templates={0: t0, 2: t2},
    )
    rx = np.concatenate([np.zeros(10, dtype=np.complex128), t2, np.zeros(5, dtype=np.complex128)])
    corr = sync.correlate(rx)
    assert corr.shape[0] == 2  # samo 2 kandidata


# -----------------------------
# UNHAPPY PATHS
# -----------------------------

def test_correlate_raises_when_rx_too_short():
    sync = _make_sync_with_templates()
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L - 1, dtype=np.complex128)

    with pytest.raises(ValueError, match="prekratak"):
        sync.correlate(rx)


def test_estimate_cfo_raises_when_segment_too_short():
    rng = np.random.default_rng(9)
    t = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(np.complex128)
    sync = _make_sync_with_templates(templates={0: t, 1: t, 2: t})

    rx = np.zeros(100, dtype=np.complex128)
    # tau_hat blizu kraja -> nema L uzoraka
    with pytest.raises(ValueError, match="prekratak"):
        sync.estimate_cfo(rx, tau_hat=80, n_id_2=1)


def test_estimate_cfo_raises_on_unknown_nid():
    sync = _make_sync_with_templates()
    rx = np.zeros(500, dtype=np.complex128)

    with pytest.raises(KeyError):
        sync.estimate_cfo(rx, tau_hat=0, n_id_2=99)


def test_build_time_templates_raises_on_fs_mismatch(monkeypatch):
    # Ovaj test stvarno gađa granu u _build_time_templates koja diže ValueError (fs mismatch)
    # bez pravog LTE chain-a: monkeypatch LTETxChain i OFDMModulator.
    class FakeTx:
        def __init__(self, n_id_2, ndlrb, num_subframes, normal_cp):
            self.grid = np.zeros((1, 1), dtype=np.complex128)

        def generate_waveform(self, mib_bits=None):
            # dovoljno dugačak signal da slicing radi
            return np.zeros(5000, dtype=np.complex64), 123.0  # namjerno kriv fs

    class FakeOfdm:
        def __init__(self, grid):
            self.N = 128
            self.n_symbols_per_slot = 7
            self.cp_lengths = [9] * 7
            self.num_ofdm_symbols = 14

    monkeypatch.setattr(pss_mod, "LTETxChain", FakeTx)
    monkeypatch.setattr(pss_mod, "OFDMModulator", FakeOfdm)

    with pytest.raises(ValueError, match="Template fs"):
        PSSSynchronizer(sample_rate_hz=1_000_000.0)


def test_symbol_start_indices_computation():
    # Direktno testira pomoćnu metodu _symbol_start_indices
    class FakeOfdm:
        N = 4
        n_symbols_per_slot = 2
        cp_lengths = [1, 2]
        num_ofdm_symbols = 4

    starts = PSSSynchronizer._symbol_start_indices(FakeOfdm())
    # sym0: cp=1 -> len=5, sym1: cp=2 -> len=6, sym2: cp=1 -> len=5, sym3: cp=2 -> len=6
    assert starts == [0, 5, 11, 16]
