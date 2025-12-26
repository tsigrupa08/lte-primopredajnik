import numpy as np
import pytest

from transmitter.pss import PSSGenerator
from channel.frequency_offset import FrequencyOffset
from receiver.pss_sync import PSSSynchronizer


# ============================================================
# Helper: generisanje RX signala sa PSS-om
# ============================================================

def generate_rx_pss(
    n_id_2: int,
    timing_offset: int,
    sample_rate: float,
    rx_len: int,
    cfo_hz: float = 0.0,
) -> np.ndarray:
    """
    Generiše RX signal koji sadrži JEDAN PSS
    sa poznatim timing ofsetom i opcionalnim CFO-om.
    """
    pss = PSSGenerator.generate(n_id_2)

    rx = np.zeros(rx_len, dtype=np.complex128)
    rx[timing_offset: timing_offset + len(pss)] = pss

    if cfo_hz != 0.0:
        rx = FrequencyOffset(
            freq_offset_hz=cfo_hz,
            sample_rate_hz=sample_rate
        ).apply(rx)

    return rx


# ============================================================
# TEST 1: Timing + CFO estimacija (osnovni RX test)
# ============================================================

@pytest.mark.parametrize("n_id_2", [0, 1, 2])
def test_pss_timing_and_cfo_estimation(n_id_2):
    """
    Generiše RX signal sa poznatim:
    - N_ID_2
    - timing ofsetom
    - CFO-om

    Provjerava da RX pronađe:
    - ispravan N_ID_2
    - ispravan tau_hat
    - CFO u toleranciji
    """
    sample_rate = 1.92e6
    rx_len = 4096

    timing_offset = 200
    cfo_true = 500.0  # Hz

    rx = generate_rx_pss(
        n_id_2=n_id_2,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=cfo_true,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)

    corr = sync.correlate(rx)
    tau_hat, detected_nid = sync.estimate_timing(corr)
    cfo_hat = sync.estimate_cfo(rx, tau_hat, detected_nid)

    assert detected_nid == n_id_2
    assert abs(tau_hat - timing_offset) <= 1
    assert abs(cfo_hat - cfo_true) < 100.0


# ============================================================
# TEST 2: Selektivnost – samo jedan PSS ima najveći peak
# ============================================================

def test_pss_selectivity_only_one_peak():
    """
    Testira da od sva 3 PSS kandidata
    SAMO ispravan ima najveći korelacijski peak.
    """
    sample_rate = 1.92e6
    rx_len = 4096

    true_nid = 1
    timing_offset = 300

    rx = generate_rx_pss(
        n_id_2=true_nid,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=0.0,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    peak_magnitudes = np.max(np.abs(corr), axis=1)
    detected_index = int(np.argmax(peak_magnitudes))
    detected_nid = sync.n_id_2_candidates[detected_index]

    assert detected_nid == true_nid


# ============================================================
# TEST 3: Globalni maksimum daje ispravan PSS i timing
# ============================================================

def test_all_pss_candidates_compete():
    """
    Eksplicitno provjerava da:
    - sva 3 PSS-a učestvuju u korelaciji
    - ali samo jedan ima GLOBALNI maksimum
    """
    sample_rate = 1.92e6
    rx_len = 4096

    true_nid = 2
    timing_offset = 150

    rx = generate_rx_pss(
        n_id_2=true_nid,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=0.0,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    max_idx = np.unravel_index(np.abs(corr).argmax(), corr.shape)
    detected_nid = sync.n_id_2_candidates[max_idx[0]]
    tau_hat = max_idx[1]

    assert detected_nid == true_nid
    assert abs(tau_hat - timing_offset) <= 1


# ============================================================
# TEST 4: Timing stabilnost za više ofseta
# ============================================================

@pytest.mark.parametrize("timing_offset", [50, 128, 512, 900, 1500])
def test_pss_timing_multiple_offsets(timing_offset):
    """
    Provjerava tačnost timing detekcije
    za različite pozicije PSS-a u RX signalu.
    """
    sample_rate = 1.92e6
    rx_len = 4096
    n_id_2 = 0

    rx = generate_rx_pss(
        n_id_2=n_id_2,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=0.0,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    tau_hat, detected_nid = sync.estimate_timing(corr)

    assert detected_nid == n_id_2
    assert abs(tau_hat - timing_offset) <= 1


# ============================================================
# TEST 5: CFO = 0 → procijenjeni CFO ≈ 0
# ============================================================

@pytest.mark.parametrize("n_id_2", [0, 1, 2])
def test_cfo_zero_case(n_id_2):
    """
    Ako nema CFO-a, procijenjeni CFO
    mora biti praktično nula.
    """
    sample_rate = 1.92e6
    rx_len = 4096
    timing_offset = 256

    rx = generate_rx_pss(
        n_id_2=n_id_2,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=0.0,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    tau_hat, detected_nid = sync.estimate_timing(corr)
    cfo_hat = sync.estimate_cfo(rx, tau_hat, detected_nid)

    assert detected_nid == n_id_2
    assert abs(cfo_hat) < 1.0


# ============================================================
# TEST 6: Znak CFO-a (pozitivan / negativan)
# ============================================================

@pytest.mark.parametrize("cfo_true", [-800.0, -300.0, 300.0, 800.0])
def test_cfo_sign_detection(cfo_true):
    """
    Provjerava da je znak procijenjenog CFO-a
    isti kao znak stvarnog CFO-a.
    """
    sample_rate = 1.92e6
    rx_len = 4096
    timing_offset = 200
    n_id_2 = 1

    rx = generate_rx_pss(
        n_id_2=n_id_2,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=cfo_true,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    tau_hat, detected_nid = sync.estimate_timing(corr)
    cfo_hat = sync.estimate_cfo(rx, tau_hat, detected_nid)

    assert detected_nid == n_id_2
    assert np.sign(cfo_hat) == np.sign(cfo_true)


# ============================================================
# TEST 7: CFO korekcija ne smije promijeniti timing
# ============================================================

def test_cfo_correction_preserves_timing():
    """
    Provjerava da CFO korekcija
    ne utiče na procjenu tau_hat.
    """
    sample_rate = 1.92e6
    rx_len = 4096
    timing_offset = 350
    n_id_2 = 2
    cfo_true = 600.0

    rx = generate_rx_pss(
        n_id_2=n_id_2,
        timing_offset=timing_offset,
        sample_rate=sample_rate,
        rx_len=rx_len,
        cfo_hz=cfo_true,
    )

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)

    corr1 = sync.correlate(rx)
    tau1, nid1 = sync.estimate_timing(corr1)
    cfo_hat = sync.estimate_cfo(rx, tau1, nid1)

    rx_corr = sync.apply_cfo_correction(rx, cfo_hat)

    corr2 = sync.correlate(rx_corr)
    tau2, nid2 = sync.estimate_timing(corr2)

    assert nid1 == n_id_2
    assert nid2 == n_id_2
    assert abs(tau1 - tau2) <= 1


# ============================================================
# TEST 8: Nema PSS-a → nema dominantnog pika
# ============================================================

def test_no_pss_no_dominant_peak():
    """
    Ako RX signal ne sadrži PSS,
    korelacije ne smiju imati dominantan peak.
    """
    sample_rate = 1.92e6
    rx_len = 4096

    rng = np.random.default_rng(0)
    rx = rng.normal(size=rx_len) + 1j * rng.normal(size=rx_len)

    sync = PSSSynchronizer(sample_rate_hz=sample_rate)
    corr = sync.correlate(rx)

    peaks = np.max(np.abs(corr), axis=1)

    ratio = np.max(peaks) / np.mean(peaks)
    assert ratio < 3.0
