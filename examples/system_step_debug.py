# examples/system_step_debug.py
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Import path fix (da radi: python examples/system_step_debug.py)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# auto-create __init__.py (da relative importi rade u channel/)
for pkg in ("transmitter", "receiver", "channel", "examples"):
    p = ROOT / pkg
    if p.exists() and p.is_dir():
        init = p / "__init__.py"
        if not init.exists():
            try:
                init.write_text("# auto-created\n", encoding="utf-8")
            except Exception:
                pass

# ------------------------------------------------------------
# Imports iz tvog projekta
# ------------------------------------------------------------
from transmitter.LTETxChain import LTETxChain
from transmitter.pbch import PBCHEncoder

from receiver.LTERxChain import LTERxChain
from receiver.OFDM_demodulator import OFDMDemodulator
from receiver.resource_grid_extractor import PBCHConfig, PBCHExtractor, pbch_symbol_indices_for_subframes

from channel.lte_channel import LTEChannel


# ------------------------------------------------------------
# Helpers: gain fit + EVM
# ------------------------------------------------------------
def best_fit_gain(x_ref: np.ndarray, x_hat: np.ndarray) -> complex:
    x_ref = np.asarray(x_ref).ravel()
    x_hat = np.asarray(x_hat).ravel()
    num = np.vdot(x_ref, x_hat)
    den = np.vdot(x_ref, x_ref)
    if den == 0:
        return 1.0 + 0j
    return num / den

def evm_db(x_ref: np.ndarray, x_hat: np.ndarray) -> float:
    x_ref = np.asarray(x_ref).ravel()
    x_hat = np.asarray(x_hat).ravel()
    if x_ref.size != x_hat.size:
        return float("nan")
    a = best_fit_gain(x_ref, x_hat)
    err = x_hat - a * x_ref
    p_ref = np.mean(np.abs(a * x_ref) ** 2) + 1e-12
    p_err = np.mean(np.abs(err) ** 2) + 1e-12
    return 10.0 * np.log10(p_err / p_ref)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def zero_fraction(x: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    return float(np.mean(np.isclose(x.real, 0.0) & np.isclose(x.imag, 0.0)))


# ------------------------------------------------------------
# PBCH extraction modes
# ------------------------------------------------------------
def extract_pbch_naive_960(grid_active: np.ndarray, ndlrb: int, normal_cp: bool, pbch_sf: int = 4) -> np.ndarray:
    # Ovo je "stari" način: uzme 960 preko svih PBCH simbola bez chunk po subfrejmu
    idx = pbch_symbol_indices_for_subframes(num_subframes=pbch_sf, normal_cp=normal_cp, start_subframe=0)
    cfg = PBCHConfig(ndlrb=ndlrb, normal_cp=normal_cp, pbch_symbol_indices=idx, pbch_symbols_to_extract=240 * pbch_sf)
    return PBCHExtractor(cfg).extract(grid_active)

def extract_pbch_per_sf_240(grid_active: np.ndarray, ndlrb: int, normal_cp: bool, pbch_sf: int = 4) -> np.ndarray:
    # Ispravno za tvoj TX: 240 po subfrejmu, pa konkatenacija => 960
    parts = []
    for sf in range(pbch_sf):
        idx_sf = pbch_symbol_indices_for_subframes(num_subframes=1, normal_cp=normal_cp, start_subframe=sf)
        cfg_sf = PBCHConfig(ndlrb=ndlrb, normal_cp=normal_cp, pbch_symbol_indices=idx_sf, pbch_symbols_to_extract=240)
        parts.append(PBCHExtractor(cfg_sf).extract(grid_active))
    return np.concatenate(parts)


# ------------------------------------------------------------
# Decode PBCH iz već izvučenih QPSK simbola (bez ponovne OFDM demod)
# koristi tvoje RX blokove: demap -> descramble -> deRM -> deinterleave -> viterbi -> CRC
# ------------------------------------------------------------
def decode_from_pbch_syms(rx: LTERxChain, pbch_syms: np.ndarray, pci_used: int) -> Tuple[Optional[np.ndarray], bool]:
    old_pci = int(rx.pci)
    rx.pci = int(pci_used)

    bits_E = rx.qpsk_demapper.demap(pbch_syms).astype(np.uint8)

    if rx.enable_descrambling:
        bits_E = rx._descramble_pbch_bits(bits_E)

    bits_120_int = rx.de_rm.derate_match(bits_E, return_soft=False).astype(np.uint8)
    bits_120_coded = rx._pbch_deinterleave_120(bits_120_int)
    decoded_40 = rx.viterbi.decode(bits_120_coded, tail_biting=True).astype(np.uint8)
    if decoded_40.size < 40:
        rx.pci = old_pci
        return None, False

    decoded_40 = decoded_40[:40]
    payload_24, ok = rx.crc.check(decoded_40)

    rx.pci = old_pci
    return payload_24.astype(np.uint8) if ok else None, bool(ok)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    # ======= Parametri testa =======
    ndlrb = 6
    normal_cp = True
    pbch_sf = 4

    n_id_2 = 0              # SISO, ok
    pci_tx = n_id_2         # tvoj dizajn: pci = n_id_2
    snr_db = 30.0
    cfo_hz = 0.0
    seed = 123

    # Ako želiš “idealno” bez šuma:
    # snr_db = 200.0

    out_dir = ROOT / "results" / "step_debug"
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")

    rng = np.random.default_rng(seed)
    mib_tx = rng.integers(0, 2, size=24, dtype=np.uint8)

    # ======= TX: waveform + grid =======
    tx = LTETxChain(n_id_2=n_id_2, ndlrb=ndlrb, num_subframes=pbch_sf, normal_cp=normal_cp)
    tx_wave, fs = tx.generate_waveform(mib_bits=mib_tx)
    tx_grid = np.asarray(tx.grid)  # (72, Ns_total)

    # PBCH symbols direktno iz enkodera (referenca)
    enc = PBCHEncoder(verbose=False, pci=pci_tx, enable_scrambling=True)
    pbch_sym_ref = np.asarray(enc.encode(mib_tx)).ravel()  # (960,)

    # ======= Sanity 1: da li TX grid stvarno sadrži te simbole u tom redoslijedu? =======
    pbch_from_txuc = extract_pbch_per_sf_240(tx_grid, ndlrb=ndlrb, normal_cp=normal_cp, pbch_sf=pbch_sf)
    pbch_from_grid = extract_pbch_per_sf_240(tx_grid, ndlrb=ndlrb, normal_cp=normal_cp, pbch_sf=pbch_sf)

    evm_tx_map = evm_db(pbch_sym_ref, pbch_from_grid)
    zf_tx = zero_fraction(pbch_from_grid)

    print("\n================ STEP DEBUG REPORT ================")
    print(f"fs={fs} Hz | ndlrb={ndlrb} | normal_cp={normal_cp} | pbch_sf={pbch_sf}")
    print(f"TX: n_id_2={n_id_2} | pci_tx={pci_tx}")
    print("---------------------------------------------------")
    print(f"[TX sanity] EVM(encoder PBCH vs TX-grid extracted) = {evm_tx_map:.2f} dB")
    print(f"[TX sanity] TX extracted PBCH zero fraction         = {zf_tx:.3f}")
    if evm_tx_map > -20.0:
        print("!!! Ako je ovo loše (npr. > -20 dB na idealnom), TX mapiranje/ekstrakcija redoslijed nisu isti.")
    print("---------------------------------------------------")

    # ======= Kanal =======
    ch = LTEChannel(
    freq_offset_hz=float(cfo_hz),
    sample_rate_hz=float(fs),
    snr_db=float(snr_db),
    seed=int(seed),
    )

    rx_wave = ch.apply(tx_wave)

    # ======= RX: prvo uradi pun RX decode (da dobijemo tau_hat/start_sf0) =======
    rx = LTERxChain(
        sample_rate_hz=float(fs),
        ndlrb=ndlrb,
        normal_cp=normal_cp,
        pbch_spread_subframes=pbch_sf,
        pci=0,
        enable_cfo_correction=True,
        enable_descrambling=True,
        normalize_before_pss=True,
    )

    res = rx.decode(rx_wave)
    print(f"[RX chain] crc_ok={res.crc_ok} | tau_hat={res.tau_hat} | n_id_2_hat={res.n_id_2_hat} | cfo_hat={res.cfo_hat}")

    # ======= OFDM demod + active grid =======
    start_sf0 = int(res.debug.get("start_sf0", 0))
    rx_aligned = rx_wave[start_sf0:]

    demod = OFDMDemodulator(ndlrb=ndlrb, normal_cp=normal_cp)
    grid_full = demod.demodulate(rx_aligned)
    grid_active = demod.extract_active_subcarriers(grid_full)

    print(f"[OFDM] grid_full={grid_full.shape} | grid_active={grid_active.shape}")
    if tx_grid.shape[1] != grid_active.shape[1]:
        print("!!! PAŽNJA: broj OFDM simbola u RX i TX gridu nije isti (dimenzije se ne poklapaju).")

    # ======= PBCH extraction: naivno vs ispravno =======
    pbch_rx_naive = extract_pbch_naive_960(grid_active, ndlrb=ndlrb, normal_cp=normal_cp, pbch_sf=pbch_sf)
    pbch_rx_fixed = extract_pbch_per_sf_240(grid_active, ndlrb=ndlrb, normal_cp=normal_cp, pbch_sf=pbch_sf)

    evm_naive = evm_db(pbch_sym_ref, pbch_rx_naive)
    evm_fixed = evm_db(pbch_sym_ref, pbch_rx_fixed)

    print(f"[PBCH RX] naive960: EVM={evm_naive:.2f} dB | zero_frac={zero_fraction(pbch_rx_naive):.3f}")
    print(f"[PBCH RX] 4x240  : EVM={evm_fixed:.2f} dB | zero_frac={zero_fraction(pbch_rx_fixed):.3f}")

    # ======= Decode iz oba slučaja (da vidiš ko ubija CRC) =======
    pci_used = int(res.n_id_2_hat)  # tvoj dizajn: pci = n_id_2
    mib_naive, ok_naive = decode_from_pbch_syms(rx, pbch_rx_naive, pci_used=pci_used)
    mib_fixed, ok_fixed = decode_from_pbch_syms(rx, pbch_rx_fixed, pci_used=pci_used)

    print(f"[Decode-from-PBCH] naive960: crc_ok={ok_naive}")
    print(f"[Decode-from-PBCH] 4x240  : crc_ok={ok_fixed}")

    if ok_fixed and mib_fixed is not None:
        ber_mib = float(np.mean(mib_fixed != mib_tx))
        print(f"[MIB] BER(24b)={ber_mib:.4e}")
    else:
        print("[MIB] nije dekodiran (CRC fail).")

    print("===================================================\n")

    # ======= Plots =======
    # 1) Konstelacije: naive vs fixed
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(pbch_rx_naive.real, pbch_rx_naive.imag, s=6)
    ax1.set_title("PBCH RX constellation (naive960)")
    ax1.grid(True)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pbch_rx_fixed.real, pbch_rx_fixed.imag, s=6)
    ax2.set_title("PBCH RX constellation (4x240)")
    ax2.grid(True)

    fig.tight_layout()
    fig_path = out_dir / f"{ts}_pbch_constellations.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # 2) “Error vector” magnitude (gain-comp) za fixed
    a = best_fit_gain(pbch_sym_ref, pbch_rx_fixed)
    err = pbch_rx_fixed - a * pbch_sym_ref
    fig = plt.figure(figsize=(10, 3))
    plt.plot(np.abs(err))
    plt.title("PBCH error vector magnitude per symbol (4x240, gain-comp)")
    plt.xlabel("symbol index")
    plt.ylabel("|e[n]|")
    plt.grid(True)
    fig.tight_layout()
    fig_path2 = out_dir / f"{ts}_pbch_error_fixed.png"
    fig.savefig(fig_path2, dpi=180)
    plt.close(fig)

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
