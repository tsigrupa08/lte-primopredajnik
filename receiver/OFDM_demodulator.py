"""
OFDM_demodulator.py

OFDM demodulator za LTE 1.4 MHz (NDLRB=6, NFFT=128) i generalno za LTE downlink
(NDLRB ∈ {6, 15, 25, 50, 75, 100}).

Cilj: biti 1:1 kompatibilan sa tvojim TX:
- OFDMModulator: IFFT + CP + fftshift/ifftshift konvencija
- resource_grid: oblik (subcarriers, symbols)
- PBCH/PSS mapiranje radi preko kolona (simbol index)

Ovaj demodulator vraća:
1) full FFT grid: shape (NFFT, num_symbols)
2) active LTE subcarriers (bez DC): shape (12*NDLRB, num_symbols)
"""

from __future__ import annotations

import numpy as np


class OFDMDemodulator:
    def __init__(self, ndlrb: int = 6, normal_cp: bool = True):
        self.ndlrb = int(ndlrb)
        self.normal_cp = bool(normal_cp)

        self.fft_size = self._determine_fft_size(self.ndlrb)
        self.n_active = 12 * self.ndlrb  # broj aktivnih podnosioca (bez DC)

        # LTE: 7 simbola/slot za normal CP, 6 za extended CP
        self.n_symbols_per_slot = 7 if self.normal_cp else 6

        self.cp_lengths = self._determine_cp_lengths()
        self.sample_rate = 15_000 * self.fft_size  # Fs = NFFT * Δf (Δf=15 kHz)

    # ------------------------------------------------------------------
    # Parametri za LTE (osnovne vrijednosti)
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_fft_size(ndlrb: int) -> int:
        """
        Odabir FFT veličine tipičan za LTE:
        1.4 MHz (6 RB)  -> 128
        3   MHz (15 RB) -> 256
        5   MHz (25 RB) -> 512
        10  MHz (50 RB) -> 1024
        15  MHz (75 RB) -> 1536
        20  MHz (100RB) -> 2048
        """
        mapping = {
            6: 128,
            15: 256,
            25: 512,
            50: 1024,
            75: 1536,
            100: 2048,
        }
        if ndlrb not in mapping:
            raise ValueError(f"Nepodržan NDLRB={ndlrb}. Očekujem jedan od {list(mapping.keys())}.")
        return mapping[ndlrb]

    def _determine_cp_lengths(self) -> list[int]:
        """
        CP dužine skalirane sa referentnog LTE FFT=2048.

        Normal CP (FFT=2048):
          - prvi simbol svakog slota: 160
          - ostali simboli slota:     144
          - 7 simbola po slotu

        Extended CP (FFT=2048):
          - svaki simbol slota: 512
          - 6 simbola po slotu
        """
        if self.normal_cp:
            cp_first_ref = 160
            cp_others_ref = 144
            cp_first = int(round(self.fft_size * cp_first_ref / 2048))
            cp_others = int(round(self.fft_size * cp_others_ref / 2048))
            # 7 simbola/slot
            return [cp_first] + [cp_others] * 6
        else:
            cp_ext_ref = 512
            cp_ext = int(round(self.fft_size * cp_ext_ref / 2048))
            # 6 simbola/slot
            return [cp_ext] * 6

    # ------------------------------------------------------------------
    # Demodulacija
    # ------------------------------------------------------------------
    def demodulate(self, rx_waveform: np.ndarray) -> np.ndarray:
        """
        Pretvara vremenski OFDM signal u frekvencijski grid (FFT binovi).

        Ulaz:
            rx_waveform: kompleksni 1D niz (sa CP)

        Izlaz:
            grid_full: np.ndarray shape (NFFT, num_symbols)
                - kolone = OFDM simboli
                - redovi = FFT binovi (fftshift-ovani, DC je u centru)
        """
        rx_waveform = np.asarray(rx_waveform)

        if rx_waveform.ndim != 1:
            raise ValueError("rx_waveform mora biti 1D niz (kompleksni uzorci).")
        if not np.iscomplexobj(rx_waveform):
            raise ValueError("rx_waveform mora biti kompleksan (I/Q).")

        symbols_freq = []
        sample_idx = 0
        symbol_idx = 0

        # čitaj simbol po simbol dok ima dovoljno uzoraka
        while True:
            sym_in_slot = symbol_idx % self.n_symbols_per_slot
            cp_len = self.cp_lengths[sym_in_slot]
            total_len = cp_len + self.fft_size

            if sample_idx + total_len > rx_waveform.size:
                break

            # ukloni CP
            ofdm_symbol_time = rx_waveform[sample_idx + cp_len : sample_idx + total_len]

            # FFT + skala (kompatibilno s TX gdje radi IFFT bez dodatne skale)
            ofdm_symbol_freq = np.fft.fft(ofdm_symbol_time) / self.fft_size

            # centriraj DC u sredinu (kompatibilno s TX fftshift mapiranjem)
            ofdm_symbol_freq = np.fft.fftshift(ofdm_symbol_freq)

            symbols_freq.append(ofdm_symbol_freq)

            sample_idx += total_len
            symbol_idx += 1

        if len(symbols_freq) == 0:
            raise ValueError("Ulazni signal je prekratak za OFDM demodulaciju (nema ni jednog simbola).")

        # Stack u shape (NFFT, num_symbols)
        grid_full = np.stack(symbols_freq, axis=1).astype(np.complex128)
        return grid_full

    def extract_active_subcarriers(self, grid_full: np.ndarray) -> np.ndarray:
        """
        Iz full FFT grida (NFFT, Ns) izdvaja aktivne LTE podnosioca (bez DC),
        i vraća resursni grid u obliku (12*NDLRB, Ns) kompatibilan sa tvojim
        TX resource_grid i resource_grid_extractor.

        Pretpostavka:
            grid_full je fftshift-ovan: DC bin je na indexu center = NFFT/2.

        Vraća:
            grid_active: shape (12*NDLRB, Ns)
                - prvo negativne frekvencije (ispod DC),
                  zatim pozitivne (iznad DC), DC se preskače.
        """
        grid_full = np.asarray(grid_full)
        if grid_full.ndim != 2:
            raise ValueError("grid_full mora biti 2D: shape (NFFT, num_symbols).")
        if grid_full.shape[0] != self.fft_size:
            raise ValueError(f"grid_full ima NFFT={grid_full.shape[0]}, očekujem {self.fft_size}.")

        Ns = grid_full.shape[1]
        center = self.fft_size // 2
        half = self.n_active // 2  # npr. 72/2 = 36

        # Negativni podnosioci: [DC-half .. DC-1]
        neg = grid_full[center - half : center, :]  # (half, Ns)

        # Pozitivni podnosioci: [DC+1 .. DC+half] (preskoči DC)
        pos = grid_full[center + 1 : center + 1 + half, :]  # (half, Ns)

        if neg.shape[0] != half or pos.shape[0] != half:
            raise ValueError("Ne mogu izdvojiti aktivne podnosioca: provjeri NFFT i NDLRB.")

        grid_active = np.vstack([neg, pos]).astype(np.complex128)  # (12*NDLRB, Ns)
        return grid_active


# ---------------------------------------------------------------------
# Brzi self-test (opcionalno)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Ovo je samo “shape sanity” bez pravog TX signala
    demod = OFDMDemodulator(ndlrb=6, normal_cp=True)
    print("NFFT:", demod.fft_size, "Fs:", demod.sample_rate)
    print("CP lengths per slot:", demod.cp_lengths)


# ==============================================================
# PRIMJERI KORIŠTENJA (KAO KOMENTAR)
# ==============================================================

"""
PRIMJER 1: Osnovna demodulacija RX signala

demod = OFDMDemodulator(ndlrb=6)

rx_waveform = np.load("rx_capture.npy")   # ili SDR buffer
grid = demod.demodulate(rx_waveform)

print(grid.shape)


PRIMJER 2: Izdvajanje aktivnih subcarriera

active_grid = demod.extract_active_subcarriers(grid)
print(active_grid.shape)   # (N_sym, 72)


PRIMJER 3: Demodulacija jednog OFDM simbola (test)

freq = np.zeros(demod.fft_size, dtype=complex)
freq[demod.fft_size // 2 + 3] = 1 + 0j

time = np.fft.ifft(np.fft.ifftshift(freq))
cp = demod.cp_lengths[0]

tx = np.concatenate([time[-cp:], time])
grid = demod.demodulate(tx)


PRIMJER 4: Debug FFT / CP poravnanja

peak = np.argmax(np.abs(grid[0]))
print("Peak index:", peak)


PRIMJER 5: Tipični receiver chain

rx_waveform
    → OFDMDemodulator.demodulate()
    → extract_active_subcarriers()
    → channel estimation
    → equalization
    → QAM demapper
"""
