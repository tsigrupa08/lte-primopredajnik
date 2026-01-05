# tests/test_pss_sync.py
from __future__ import annotations

import numpy as np
import pytest

# --- Robust imports (prilagodi ako ti je drugačija struktura) ---

from receiver.pss_sync import PSSSynchronizer


from transmitter.LTETxChain import LTETxChain



from transmitter.ofdm import OFDMModulator


def apply_cfo(x: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    """x[n] * exp(j*2*pi*cfo*n/fs)"""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(x.size, dtype=np.float64)
    return x * np.exp(1j * 2.0 * np.pi * cfo_hz * n / fs)


@pytest.fixture(scope="module")
def tx_setup():
    """
    Pravi TX waveform (1 subframe) i vraća:
    - tx_waveform
    - fs
    - true_nid2
    - pss_start (početak CP-a PSS simbola unutar tx_waveform)
    """
    ndlrb = 6
    normal_cp = True
    true_nid2 = 1

    tx = LTETxChain(n_id_2=true_nid2, ndlrb=ndlrb, num_subframes=1, normal_cp=normal_cp)
    tx_waveform, fs = tx.generate_waveform(mib_bits=None)
    tx_waveform = np.asarray(tx_waveform, dtype=np.complex128)
    fs = float(fs)

    # izračunaj gdje PSS simbol počinje u vremenu (CP start)
    ofdm = OFDMModulator(tx.grid)
    starts = PSSSynchronizer._symbol_start_indices(ofdm)
    pss_sym = 6 if normal_cp else 5
    pss_start = int(starts[pss_sym])

    return {
        "ndlrb": ndlrb,
        "normal_cp": normal_cp,
        "true_nid2": true_nid2,
        "tx_waveform": tx_waveform,
        "fs": fs,
        "pss_start": pss_start,
    }


@pytest.fixture(scope="module")
def sync(tx_setup):
    """Pravi PSSSynchronizer (bez stubovanja)."""
    return PSSSynchronizer(
        sample_rate_hz=tx_setup["fs"],
        n_id_2_candidates=(0, 1, 2),
        ndlrb=tx_setup["ndlrb"],
        normal_cp=tx_setup["normal_cp"],
    )


# =========================================================
# HAPPY PATHS
# =========================================================

def test_correlate_output_shape_and_type(sync):
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L + 100, dtype=np.complex128)

    corr = sync.correlate(rx)
    assert corr.shape == (3, (L + 100) - L + 1)
    assert np.iscomplexobj(corr)
    assert np.all(np.isfinite(corr))


def test_estimate_timing_finds_correct_tau_and_nid_no_noise(tx_setup, sync):
    timing_offset = 2000
    rx = np.concatenate([np.zeros(timing_offset, dtype=np.complex128), tx_setup["tx_waveform"]])

    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)

    tau_expected = timing_offset + tx_setup["pss_start"]
    assert nid_hat == tx_setup["true_nid2"]
    assert tau_hat == tau_expected


def test_correlate_normalization_invariant_to_scaling(sync):
    # Normalizovana korelacija => ako rx = a*template, peak magnitude ~1
    t = sync._templates[0]
    rx = 3.0 * t
    corr = sync.correlate(rx)
    peak = np.abs(corr[0, 0])
    assert peak == pytest.approx(1.0, abs=1e-10)


def test_estimate_cfo_near_zero_when_no_cfo(tx_setup, sync):
    timing_offset = 1000
    rx = np.concatenate([np.zeros(timing_offset, dtype=np.complex128), tx_setup["tx_waveform"]])

    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)

    cfo_hat = sync.estimate_cfo(rx, tau_hat=tau_hat, n_id_2=nid_hat)
    assert abs(cfo_hat) < 200.0  # dovoljno strogo, a realno (zavisno od metode)


@pytest.mark.parametrize("cfo_true", [5000.0, -5000.0])
def test_estimate_cfo_recovers_known_cfo(tx_setup, sync, cfo_true):
    timing_offset = 1200
    rx = np.concatenate([np.zeros(timing_offset, dtype=np.complex128), tx_setup["tx_waveform"]])
    rx = apply_cfo(rx, fs=tx_setup["fs"], cfo_hz=cfo_true)

    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)

    cfo_hat = sync.estimate_cfo(rx, tau_hat=tau_hat, n_id_2=nid_hat)
    # CFO metoda zavisi od implementacije; ova tolerancija ti je stabilna u praksi
    assert cfo_hat == pytest.approx(cfo_true, abs=250.0)


def test_apply_cfo_correction_reduces_estimated_cfo(tx_setup, sync):
    cfo_true = 5000.0
    timing_offset = 800
    rx = np.concatenate([np.zeros(timing_offset, dtype=np.complex128), tx_setup["tx_waveform"]])
    rx = apply_cfo(rx, fs=tx_setup["fs"], cfo_hz=cfo_true)

    corr = sync.correlate(rx)
    tau_hat, nid_hat = sync.estimate_timing(corr)

    cfo_hat = sync.estimate_cfo(rx, tau_hat=tau_hat, n_id_2=nid_hat)
    rx_corr = sync.apply_cfo_correction(rx, cfo_hat)

    cfo_after = sync.estimate_cfo(rx_corr, tau_hat=tau_hat, n_id_2=nid_hat)

    assert abs(cfo_after) < abs(cfo_hat)  # mora biti bolje
    assert abs(cfo_after) < 300.0         # nakon korekcije treba biti blizu nule


def test_apply_cfo_correction_preserves_shape_and_complex(tx_setup, sync):
    rx = np.concatenate([np.zeros(100, dtype=np.complex128), tx_setup["tx_waveform"]])
    out = sync.apply_cfo_correction(rx, cfo_hat=123.0)

    assert out.shape == rx.shape
    assert np.iscomplexobj(out)


def test_estimate_timing_prefers_stronger_peak(sync):
    corr = np.zeros((3, 50), dtype=np.complex128)
    corr[0, 10] = 2 + 1j
    corr[1, 20] = 5 + 0j  # jači peak
    corr[2, 30] = 4 + 0j

    tau_hat, nid_hat = sync.estimate_timing(corr)
    assert tau_hat == 20
    assert nid_hat == 1


# =========================================================
# UNHAPPY / SAD PATHS
# =========================================================

def test_correlate_raises_when_rx_too_short(sync):
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L - 1, dtype=np.complex128)

    with pytest.raises(ValueError):
        sync.correlate(rx)


def test_estimate_cfo_raises_when_segment_too_short(sync):
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L + 10, dtype=np.complex128)

    # tau_hat postavi tako da segment izađe van rx
    with pytest.raises(ValueError):
        sync.estimate_cfo(rx, tau_hat=50, n_id_2=0)




def test_correlate_accepts_real_input(sync):
    L = next(iter(sync._templates.values())).size
    rx = np.ones(L + 5, dtype=np.float64)

    corr = sync.correlate(rx)
    assert np.iscomplexobj(corr)
    assert corr.shape[1] == 6


def test_correlate_finite_on_all_zeros(sync):
    L = next(iter(sync._templates.values())).size
    rx = np.zeros(L + 100, dtype=np.complex128)

    corr = sync.correlate(rx)
    assert np.all(np.isfinite(corr))
    assert np.allclose(corr, 0.0)
