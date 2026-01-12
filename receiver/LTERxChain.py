# receiver/LTERxChain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Robust imports (radi i kad pokrećeš kao package i kao "script" iz foldera)
# ---------------------------------------------------------------------
try:
    from receiver.utils import RxUtils
    from receiver.pss_sync import PSSSynchronizer
    from receiver.OFDM_demodulator import OFDMDemodulator
    from receiver.resource_grid_extractor import (
        PBCHConfig,
        PBCHExtractor,
        pbch_symbol_indices_for_subframes,
    )
    from receiver.QPSK_demapiranje import QPSKDemapper
    from receiver.de_rate_matching import DeRateMatcherPBCH
    from receiver.viterbi_decoder import ViterbiDecoder
    from receiver.crc_checker import CRCChecker
except Exception:  # fallback: isti direktorij
    from utils import RxUtils
    from pss_sync import PSSSynchronizer
    from OFDM_demodulator import OFDMDemodulator
    from resource_grid_extractor import PBCHConfig, PBCHExtractor, pbch_symbol_indices_for_subframes
    from QPSK_demapiranje import QPSKDemapper
    from de_rate_matching import DeRateMatcherPBCH
    from viterbi_decoder import ViterbiDecoder
    from crc_checker import CRCChecker


# =======================================================
# Result container
# =======================================================
@dataclass
class RxResult:
    mib_bits: Optional[np.ndarray]          # (24,) ako CRC OK, inače None
    crc_ok: bool
    n_id_2_hat: int
    tau_hat: int                            # timing (CP-start PSS simbola) u uzorcima
    cfo_hat: Optional[float]                # Hz (ako enable_cfo_correction)
    debug: Dict[str, Any]


# =======================================================
# LTE RX Chain (PSS + PBCH)
# =======================================================
class LTERxChain:
    """
    LTE RX chain usklađen s tvojim TX (LTETxChain):
      - PSS detekcija (korelacija) -> tau_hat i N_ID_2_hat
      - (opc.) CFO procjena + korekcija
      - OFDM demodulacija (FFT + CP uklanjanje)
      - Ekstrakcija PBCH RE (240 po subfrejmu) kroz 4 subfrejma => 960 QPSK simbola
      - QPSK hard demap => 1920 bitova
      - descramble (Gold, c_init = PCI)
      - de-rate matching => 120 interleaved bitova
      - PBCH deinterleave (inverz TX sub-block interleavera) => 120 coded bitova
      - Viterbi tail-biting decode => 40 bitova (24 payload + 16 CRC)
      - CRC check => 24 MIB bita
    """

    # TX permutacija kolona (ista kao u transmitter/pbch.py)
    _PERM_PATTERN = np.array(
        [1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
         0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30],
        dtype=int
    )

    def __init__(
        self,
        *,
        sample_rate_hz: float | None = None,
        ndlrb: int = 6,
        normal_cp: bool = True,
        pci: int = 0,
        pbch_spread_subframes: int = 4,
        enable_cfo_correction: bool = True,
        enable_descrambling: bool = True,
        normalize_before_pss: bool = True,
        # PSS pretraga: pošto TX mapira PSS u SF0, tražimo peak u prozoru oko očekivane pozicije
        pss_window_samples: int | None = None,
        # ako je CFO estimator “divlji”, clampaj da ne uništi signal
        cfo_limit_hz: float = 3000.0,
    ) -> None:
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)
        self.pci = int(pci)

        self.pbch_spread_subframes = int(pbch_spread_subframes)
        self.enable_cfo_correction = bool(enable_cfo_correction)
        self.enable_descrambling = bool(enable_descrambling)
        self.normalize_before_pss = bool(normalize_before_pss)

        self.pss_window_samples = pss_window_samples
        self.cfo_limit_hz = float(cfo_limit_hz)

        # pomoćni alati
        self.u = RxUtils()

        # OFDM demod (ovdje dobijemo očekivani Fs)
        self.ofdm_demod = OFDMDemodulator(ndlrb=self.ndlrb, normal_cp=self.normal_cp)
        fs_expected = float(self.ofdm_demod.sample_rate)

        # Ako nije zadat sample_rate_hz, uzmi očekivani iz OFDM parametara
        if sample_rate_hz is None:
            self.sample_rate_hz = fs_expected
        else:
            self.sample_rate_hz = float(sample_rate_hz)
            # tolerantnije poređenje (da ne puca na float sitnice)
            if not np.isclose(self.sample_rate_hz, fs_expected, rtol=0.0, atol=1e-3):
                raise ValueError(
                    f"sample_rate_hz={self.sample_rate_hz} ne odgovara očekivanom Fs={fs_expected} "
                    f"(NDLRB={self.ndlrb}, Δf=15 kHz, NFFT={self.ofdm_demod.fft_size})."
                )

        # PSS sync
        self.pss_sync = PSSSynchronizer(
            sample_rate_hz=self.sample_rate_hz,
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp,
        )

        # PBCH extractor konfiguracija (po TX: 240 simbola po subfrejmu, 4 subfrejma => 960)
        pbch_symbol_indices = pbch_symbol_indices_for_subframes(
            num_subframes=self.pbch_spread_subframes,
            normal_cp=self.normal_cp,
            start_subframe=0,
        )
        self.pbch_cfg = PBCHConfig(
            ndlrb=self.ndlrb,
            normal_cp=self.normal_cp,
            pbch_symbol_indices=pbch_symbol_indices,
            pbch_symbols_to_extract=240 * self.pbch_spread_subframes,
        )
        self.pbch_extractor = PBCHExtractor(self.pbch_cfg)

        # QPSK demap
        self.qpsk_demapper = QPSKDemapper(mode="hard")

        # de-rate matcher (E -> 120 interleaved)
        self.de_rm = DeRateMatcherPBCH(n_coded=120)

        # Viterbi (rate 1/3, K=7, polinomi kao TX)
        self.viterbi = ViterbiDecoder(
            constraint_len=7,
            generators=[0o133, 0o171, 0o165],
            rate=1 / 3,
        )

        # CRC
        self.crc = CRCChecker()

    # -------------------------------------------------------
    # Gold scrambler (isti algoritam kao u transmitter/pbch.py)
    # -------------------------------------------------------
    @staticmethod
    def _gold_sequence_pbch(c_init: int, length: int) -> np.ndarray:
        c_init = int(c_init)
        if not (0 <= c_init <= 503):
            raise ValueError("PCI (c_init) mora biti u [0,503].")

        x1 = np.zeros(31, dtype=np.uint8)
        x1[0] = 1
        x2 = np.array([(c_init >> i) & 1 for i in range(31)], dtype=np.uint8)

        def step(reg: np.ndarray, taps: Tuple[int, ...]) -> np.ndarray:
            new_bit = np.uint8(0)
            for t in taps:
                new_bit ^= reg[t]
            reg = np.roll(reg, -1)
            reg[-1] = new_bit
            return reg

        # warm-up
        for _ in range(1600):
            x1 = step(x1, (0, 3))
            x2 = step(x2, (0, 1, 2, 3))

        out = np.zeros(int(length), dtype=np.uint8)
        for n in range(int(length)):
            out[n] = x1[0] ^ x2[0]
            x1 = step(x1, (0, 3))
            x2 = step(x2, (0, 1, 2, 3))

        return out

    def _descramble_pbch_bits(self, bits: np.ndarray) -> np.ndarray:
        b = np.asarray(bits, dtype=np.uint8).ravel()
        c = self._gold_sequence_pbch(self.pci, b.size)
        return (b ^ c).astype(np.uint8)

    # -------------------------------------------------------
    # Inverz sub-block interleavera (120 -> 120)
    # -------------------------------------------------------
    def _pbch_deinterleave_120(self, interleaved_120: np.ndarray) -> np.ndarray:
        b = np.asarray(interleaved_120, dtype=np.uint8).ravel()
        if b.size != 120:
            raise ValueError("Očekujem tačno 120 interleaved bitova.")

        C = 32
        R = 2
        Nin = 40
        Nd = 24

        # out_pos -> in_pos map (kao u TX, ali sa markerima)
        linear = np.concatenate((np.full(Nd, -1, dtype=np.int16), np.arange(Nin, dtype=np.int16)))
        mat = linear.reshape((C, R), order="F").T          # (2,32)
        mat = mat[:, self._PERM_PATTERN]                   # permute columns
        out = mat.flatten(order="F")
        out = out[out != -1].astype(int)                   # length 40
        idx_map = out                                      # out_pos -> in_pos

        # split streams
        s0 = b[0:40]
        s1 = b[40:80]
        s2 = b[80:120]

        d0 = np.zeros(40, dtype=np.uint8)
        d1 = np.zeros(40, dtype=np.uint8)
        d2 = np.zeros(40, dtype=np.uint8)

        # invert mapping: original[in_pos] = received[out_pos]
        d0[idx_map] = s0
        d1[idx_map] = s1
        d2[idx_map] = s2

        coded = np.zeros(120, dtype=np.uint8)
        coded[0::3] = d0
        coded[1::3] = d1
        coded[2::3] = d2
        return coded

    # -------------------------------------------------------
    # Koliko uzoraka od starta subfrejma do CP-start PSS simbola (u SF0)
    # -------------------------------------------------------
    def _offset_samples_to_pss_cp_start(self) -> int:
        N = self.ofdm_demod.fft_size
        cps = self.ofdm_demod.cp_lengths  # per-slot (7 simbola za normal CP)

        if self.normal_cp:
            # PSS u l=6 (zadnji simbol slota0). Treba proći simbole 0..5.
            cp0 = int(cps[0])
            cp1 = int(cps[1])
            return (N + cp0) + 5 * (N + cp1)
        else:
            # PSS u l=5 (zadnji simbol slota0). Treba proći simbole 0..4.
            cp = int(cps[0])
            return 5 * (N + cp)

    def _samples_per_subframe(self) -> int:
        N = self.ofdm_demod.fft_size
        cps = self.ofdm_demod.cp_lengths  # CP pattern za 1 slot

        slot = int(sum((N + int(cp)) for cp in cps))   # 1 slot
        return 2 * slot                                # 1 subframe = 2 slota

    # =======================================================
    # Glavna funkcija: RX decode
    # =======================================================
    def decode(self, rx_waveform: np.ndarray) -> RxResult:
        dbg: Dict[str, Any] = {}

        # 1) Validacija / shape
        rx = self.u.validate_rx_samples(rx_waveform, min_num_samples=256)
        rx = self.u.ensure_1d_time_axis(rx)

        # (opc.) normalizacija RMS prije korelacije (stabilniji peak)
        if self.normalize_before_pss:
            rx, scale = self.u.normalize_rms(rx, target_rms=1.0)
            dbg["rms_norm_scale"] = float(scale)
        else:
            dbg["rms_norm_scale"] = None

        # 2) PSS korelacija -> tau_hat, n_id_2_hat
        corr = self.pss_sync.correlate(rx)
        abs_corr = np.abs(corr)

        offset_to_pss = self._offset_samples_to_pss_cp_start()
        dbg["tau_expected"] = int(offset_to_pss)

        # Robust: podrži i 1D i 2D output korelacije
        if abs_corr.ndim == 1:
            abs_corr = abs_corr.reshape(1, -1)

        # Pošto TX mapira PSS u SF0, tražimo peak u prozoru oko očekivane pozicije
        N = int(self.ofdm_demod.fft_size)
        win = self.pss_window_samples
        if win is None:
            win = max(2 * N, 128)  # npr. za N=128 -> 256

        lo = max(0, int(offset_to_pss) - int(win))
        hi = min(abs_corr.shape[1], int(offset_to_pss) + int(win) + 1)

        dbg["tau_win"] = (int(lo), int(hi))

        if hi > lo:
            cand = np.arange(lo, hi)
            sub = abs_corr[:, cand]  # (K, len(cand))
            k_idx, c_idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            tau_hat = int(cand[int(c_idx)])
            n_id_2_hat = int(self.pss_sync.n_id_2_candidates[int(k_idx)])
            dbg["tau_search_mode"] = "expected_pss_window_sf0"
            corr_peak = float(abs_corr[int(k_idx), int(tau_hat)])
        else:
            # fallback (prekratak signal)
            k_idx, tau_hat = np.unravel_index(int(np.argmax(abs_corr)), abs_corr.shape)
            n_id_2_hat = int(self.pss_sync.n_id_2_candidates[int(k_idx)])
            dbg["tau_search_mode"] = "global_argmax_fallback"
            corr_peak = float(abs_corr[int(k_idx), int(tau_hat)])

        dbg["corr_peak"] = corr_peak
        dbg["tau_hat_raw"] = int(tau_hat)
        dbg["n_id_2_hat"] = int(n_id_2_hat)
        # nakon PSS detekcije
        self.pci = int(n_id_2_hat)   # jer je u TX scrambling rađen sa pci = n_id_2
        dbg["pci_used_for_pbch"] = self.pci
        # 3) CFO procjena + korekcija (opc.)
        cfo_hat = None
        rx_corr = rx

        if self.enable_cfo_correction:
            cfo_hat = float(self.pss_sync.estimate_cfo(rx, int(tau_hat), int(n_id_2_hat)))

            # clamp da se ne pokvari signal ako estimator “odleti”
            if abs(cfo_hat) > self.cfo_limit_hz:
                dbg["cfo_clamped"] = True
                cfo_hat = 0.0
            else:
                dbg["cfo_clamped"] = False

            rx_corr = self.pss_sync.apply_cfo_correction(rx, cfo_hat)

        dbg["cfo_hat_hz"] = cfo_hat

        # 4) Align na start subfrejma
        #    start_sf0_raw je koliko uzoraka prije PSS CP-starta treba “nazad” do SF0 starta.
        start_sf0_raw = int(tau_hat) - int(offset_to_pss)

        # snap na najbližu subframe granicu (u tvojoj simulaciji ovo treba postaviti na 0)
        spsf = self._samples_per_subframe()
        if spsf > 0:
            start_sf0 = int(np.rint(start_sf0_raw / spsf) * spsf)
        else:
            start_sf0 = start_sf0_raw

        if start_sf0 < 0:
            start_sf0 = 0

        dbg["start_sf0_raw"] = int(start_sf0_raw)
        dbg["start_sf0"] = int(start_sf0)
        dbg["samples_per_subframe"] = int(spsf)

        rx_aligned = rx_corr[start_sf0:]

        # 5) OFDM demod (FFT grid: (NFFT, Ns))
        try:
            grid_full = self.ofdm_demod.demodulate(rx_aligned)
        except Exception as e:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": f"OFDM demod failed: {e}"},
            )

        dbg["grid_full_shape"] = tuple(grid_full.shape)

        # 6) Aktivni subcarriers (72, Ns)
        try:
            grid_active = self.ofdm_demod.extract_active_subcarriers(grid_full)  # (72, Ns)
        except Exception as e:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": f"Active subcarrier extraction failed: {e}"},
            )

        dbg["grid_active_shape"] = tuple(grid_active.shape)

        # 7) Extract PBCH symbols (960 za 4 subfrejma)
        # 7) PBCH ekstrakcija: 240 simbola PO subfrejmu (TX radi chunk po subfrejmu)
        try:
            pbch_syms_parts = []
            for sf in range(self.pbch_spread_subframes):
                idx_sf = pbch_symbol_indices_for_subframes(
                    num_subframes=1,
                    normal_cp=self.normal_cp,
                    start_subframe=sf,
                )
                cfg_sf = PBCHConfig(
                    ndlrb=self.ndlrb,
                    normal_cp=self.normal_cp,
                    pbch_symbol_indices=idx_sf,
                    pbch_symbols_to_extract=240,   # KLJUČNO: 240 po subfrejmu
                )
                pbch_syms_sf = PBCHExtractor(cfg_sf).extract(grid_active)  # (240,)
                pbch_syms_parts.append(pbch_syms_sf)

            pbch_syms = np.concatenate(pbch_syms_parts)  # (240*pbch_spread_subframes,) => (960,)
        except Exception as e:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": f"PBCH extract failed: {e}"},
            )


        dbg["pbch_syms_len"] = int(pbch_syms.size)

        # 8) QPSK demap -> 1920 bits
        bits_E = self.qpsk_demapper.demap(pbch_syms).astype(np.uint8)
        dbg["bits_E_len"] = int(bits_E.size)

        # 9) Descramble – prije de-rate-matching
        if self.enable_descrambling:
            bits_E = self._descramble_pbch_bits(bits_E)
        dbg["descrambling"] = bool(self.enable_descrambling)

        # 10) De-rate matching: E -> 120 interleaved
        try:
            bits_120_int = self.de_rm.derate_match(bits_E, return_soft=False).astype(np.uint8)
        except Exception as e:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": f"De-rate-matching failed: {e}"},
            )

        dbg["bits_120_interleaved_len"] = int(bits_120_int.size)

        # 11) Deinterleave: 120 interleaved -> 120 coded
        try:
            bits_120_coded = self._pbch_deinterleave_120(bits_120_int)
        except Exception as e:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": f"PBCH deinterleave failed: {e}"},
            )

        dbg["bits_120_coded_len"] = int(bits_120_coded.size)

        # 12) Viterbi tail-biting decode: 120 -> 40
        decoded_40 = self.viterbi.decode(bits_120_coded, tail_biting=True).astype(np.uint8)
        dbg["decoded_40_len"] = int(decoded_40.size)

        if decoded_40.size < 40:
            return RxResult(
                mib_bits=None,
                crc_ok=False,
                n_id_2_hat=int(n_id_2_hat),
                tau_hat=int(tau_hat),
                cfo_hat=cfo_hat,
                debug={**dbg, "error": "Viterbi nije vratio 40 bitova (prekratko)."},
            )

        decoded_40 = decoded_40[:40]

        # 13) CRC check: 40 -> 24 + CRC OK
        payload_24, ok = self.crc.check(decoded_40)
        ok = bool(ok)
        dbg["crc_ok"] = ok

        


        mib_hat_24 = decoded_40[:24].astype(np.uint8).copy()
        dbg["mib_hat_24"] = mib_hat_24

        return RxResult(
            mib_bits=payload_24.astype(np.uint8) if ok else None,
            crc_ok=ok,
            n_id_2_hat=int(n_id_2_hat),
            tau_hat=int(tau_hat),
            cfo_hat=cfo_hat,
            debug=dbg,
        )

# =======================================================
# Quick local smoke test
# =======================================================
if __name__ == "__main__":
    print("LTERxChain loaded. Use: rx = LTERxChain(...); res = rx.decode(rx_waveform)")
