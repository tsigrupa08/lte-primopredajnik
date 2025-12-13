"""
LTE Tx → Kanal pipeline sa vizualizacijom PBCH konstelacije.

Ovaj skript:
1) Kreira LTE predajni lanac (LTETxChain),
2) Enkodira 24-bitni MIB u PBCH QPSK,
3) Generiše OFDM talasni oblik (Tx),
4) Prolazi kroz kompozitni LTE kanal (frekvencijski ofset + AWGN),
5) Iscrtava dio Tx/Rx talasnog oblika,
6) Iscrtava PBCH QPSK konstelaciju (predajna strana).

Koristi NumPy stil i modularnu organizaciju.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import LTE predajnog lanca i kompozitnog kanala
from transmitter.LTETxChain import LTETxChain
from channel.lte_channel import LTEChannel
from transmitter.pbch import PBCHEncoder


def make_mib_bits(num_bits: int = 24) -> np.ndarray:
    """
    Kreira jednostavan 24-bitni MIB vektor.

    Parametri
    ---------
    num_bits : int
        Broj bitova za MIB (default 24).

    Povratna vrijednost
    -------------------
    np.ndarray
        1D niz oblika (num_bits,) sa vrijednostima {0,1}.
    """
    # Fiksno sjeme radi reproduktivnosti; možeš zamijeniti svojom MIB semantikom
    rng = np.random.default_rng(42)
    bits = rng.integers(low=0, high=2, size=num_bits, dtype=np.int64)
    return bits


def build_transmitter(nid2: int = 0, ndlrb: int = 6, num_subframes: int = 1, normal_cp: bool = True) -> LTETxChain:
    """
    Kreira LTE predajni lanac.

    Parametri
    ---------
    nid2 : int
        Fizički identitet ćelije (NID2), vrijednost 0–2.
    ndlrb : int
        Broj downlink resurs blokova (minimalno 6 za 1.4 MHz).
    num_subframes : int
        Broj subframe-ova.
    normal_cp : bool
        Ako je True koristi se normalni cyclic prefix, inače extended CP.

    Povratna vrijednost
    -------------------
    LTETxChain
        Konfigurisani LTE predajni lanac.
    """
    return LTETxChain(nid2=nid2, ndlrb=ndlrb, num_subframes=num_subframes, normal_cp=normal_cp)


def generate_tx_waveform(tx: LTETxChain, mib_bits: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Generiše LTE OFDM talasni oblik sa PSS + PBCH.

    Parametri
    ---------
    tx : LTETxChain
        LTE predajni lanac.
    mib_bits : np.ndarray
        24-bitni MIB vektor za PBCH enkodiranje.

    Povratna vrijednost
    -------------------
    tuple
        (waveform, fs) gdje je waveform kompleksni signal u vremenskoj domeni,
        a fs sample rate u Hz.
    """
    waveform, fs = tx.generate_waveform(mib_bits=mib_bits)
    return waveform, fs


def build_channel(freq_offset_hz: float, fs: float, snr_db: float, seed: int | None = 123, initial_phase_rad: float = 0.0) -> LTEChannel:
    """
    Kreira kompozitni LTE kanal (frekvencijski ofset + AWGN).

    Parametri
    ---------
    freq_offset_hz : float
        Frekvencijski ofset u Hz (Δf).
    fs : float
        Sample rate u Hz.
    snr_db : float
        Ciljani SNR u dB.
    seed : int ili None
        Sjeme za AWGN generator.
    initial_phase_rad : float
        Početna faza kompleksnog eksponencijala.

    Povratna vrijednost
    -------------------
    LTEChannel
        Konfigurisani kompozitni kanal.
    """
    return LTEChannel(
        freq_offset_hz=freq_offset_hz,
        sample_rate_hz=fs,
        snr_db=snr_db,
        seed=seed,
        initial_phase_rad=initial_phase_rad,
    )


def apply_channel(ch: LTEChannel, x: np.ndarray) -> np.ndarray:
    """
    Primjenjuje LTE kanal na predajni talasni oblik.

    Parametri
    ---------
    ch : LTEChannel
        Kompozitni kanal.
    x : np.ndarray
        Kompleksni predajni signal.

    Povratna vrijednost
    -------------------
    np.ndarray
        Kompleksni primljeni signal nakon impairments-a.
    """
    # Reset internog stanja (npr. brojača uzoraka u FrequencyOffset)
    ch.reset()
    y = ch.apply(x)
    return y


def encode_pbch_symbols(mib_bits: np.ndarray, target_bits: int = 384, verbose: bool = False) -> np.ndarray:
    """
    Enkodira PBCH iz MIB bitova i vraća QPSK simbole (predajna konstelacija).

    Parametri
    ---------
    mib_bits : np.ndarray
        24-bitni MIB vektor.
    target_bits : int
        Dužina enkodiranog PBCH (384 bita).
    verbose : bool
        Flag za detaljan ispis.

    Povratna vrijednost
    -------------------
    np.ndarray
        Kompleksni QPSK simboli za PBCH.
    """
    enc = PBCHEncoder(target_bits=target_bits, verbose=verbose)
    symbols = enc.encode(mib_bits)
    return symbols


def plot_waveforms(tx: np.ndarray, rx: np.ndarray, fs: float, num_samples: int = 2000) -> None:
    """
    Iscrtava segment Tx i Rx talasnog oblika (realni i imaginarni dio).

    Parametri
    ---------
    tx : np.ndarray
        Predajni signal.
    rx : np.ndarray
        Primljeni signal.
    fs : float
        Sample rate u Hz.
    num_samples : int
        Broj uzoraka za prikaz.
    """
    n = min(num_samples, tx.size, rx.size)
    t = np.arange(n) / fs

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, tx[:n].real, label="Tx Real")
    plt.plot(t, tx[:n].imag, label="Tx Imag", alpha=0.8)
    plt.title("Predajni talasni oblik (segment)")
    plt.xlabel("Vrijeme [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(t, rx[:n].real, label="Rx Real")
    plt.plot(t, rx[:n].imag, label="Rx Imag", alpha=0.8)
    plt.title("Primljeni talasni oblik (segment)")
    plt.xlabel("Vrijeme [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_constellation(symbols: np.ndarray, title: str = "PBCH QPSK konstelacija (Tx)") -> None:
    """
    Scatter-plot kompleksne konstelacije.

    Parametri
    ---------
    symbols : np.ndarray
        Kompleksni simboli za prikaz.
    title : str
        Naslov grafa.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(symbols.real, symbols.imag, s=10, alpha=0.7)
    plt.axhline(0.0, color='gray', linewidth=0.8)
    plt.axvline(0.0, color='gray', linewidth=0.8)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.axis("equal")
    plt.show()


def main() -> None:
    """
    Izvršava cijeli pipeline:
    - Kreira Tx lanac
    - Generiše MIB
    - Generiše OFDM talasni oblik
    - Kreira i primjenjuje kanal
    - Iscrtava Tx/Rx talasne oblike
    - Iscrtava PBCH konstelaciju (Tx strana)
    """
    # 1) Predajni lanac
    tx = build_transmitter(nid2=0, ndlrb=6, num_subframes=1, normal_cp=True)

    # 2) MIB (24 bita)
    mib_bits = make_mib_bits(num_bits=24)

    # 3) Generisanje Tx talasnog oblika
    tx_waveform, fs = generate_tx_waveform(tx, mib_bits=mib_bits)

    # 4) Parametri kanala (primjer)
    snr_db = 10.0
    delta_f_hz = 300.0
    ch = build_channel(
        freq_offset_hz=delta_f_hz,
        fs=fs,
        snr_db=snr_db,
        seed=123,
        initial_phase_rad=0.0
    )

    # 5) Primjena kanala na predajni signal
    rx_waveform = apply_channel(ch, tx_waveform)

    # 6) Iscrtavanje segmenta Tx/Rx talasnog oblika
    plot_waveforms(tx_waveform, rx_waveform, fs, num_samples=2000)

    # 7) PBCH konstelacija (predajna strana)
    pbch_symbols = encode_pbch_symbols(mib_bits=mib_bits, target_bits=384, verbose=False)
    plot_constellation(pbch_symbols, title="PBCH QPSK konstelacija (Tx)")


if __name__ == "__main__":
    main()
