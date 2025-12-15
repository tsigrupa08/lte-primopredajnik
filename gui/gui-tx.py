# gui/lte_gui.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from transmitter.LTETxChain import LTETxChain
from channel.lte_channel import LTEChannel


class LTEGuiApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LTE TX + Channel GUI")
        self.geometry("1200x650")

        self.tx_wave: np.ndarray | None = None
        self.rx_wave: np.ndarray | None = None
        self.fs: float | None = None
        self.grid: np.ndarray | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)

        # ---------------- Controls ----------------
        tx_box = ttk.LabelFrame(left, text="TX", padding=10)
        tx_box.pack(fill="x", pady=(0, 10))

        self.var_nid2 = tk.IntVar(value=0)
        self.var_normal_cp = tk.BooleanVar(value=True)
        self.var_include_pbch = tk.BooleanVar(value=True)
        self.var_frame_mod4 = tk.IntVar(value=0)
        self.var_mib_bits = tk.StringVar(value="")  # 24 bita ili prazno

        ttk.Label(tx_box, text="NID2 (0-2):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(tx_box, from_=0, to=2, textvariable=self.var_nid2, width=6).grid(row=0, column=1, sticky="w")

        ttk.Checkbutton(tx_box, text="Normal CP", variable=self.var_normal_cp).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tx_box, text="Uključi PBCH", variable=self.var_include_pbch).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        ttk.Label(tx_box, text="Frame mod 4 (0-3):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(tx_box, from_=0, to=3, textvariable=self.var_frame_mod4, width=6).grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(tx_box, text="MIB bits (24 bita, opcionalno):").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Entry(tx_box, textvariable=self.var_mib_bits, width=28).grid(row=5, column=0, columnspan=2, sticky="we")

        ch_box = ttk.LabelFrame(left, text="Kanal", padding=10)
        ch_box.pack(fill="x", pady=(0, 10))

        self.var_apply_channel = tk.BooleanVar(value=True)
        self.var_freq_offset = tk.DoubleVar(value=100.0)
        self.var_snr_db = tk.DoubleVar(value=15.0)
        self.var_seed = tk.StringVar(value="42")
        self.var_init_phase = tk.DoubleVar(value=0.0)

        ttk.Checkbutton(ch_box, text="Primijeni kanal", variable=self.var_apply_channel).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(ch_box, text="Frekv. ofset (Hz):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ch_box, textvariable=self.var_freq_offset, width=10).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(ch_box, text="SNR (dB):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ch_box, textvariable=self.var_snr_db, width=10).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(ch_box, text="Seed (prazno=None):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ch_box, textvariable=self.var_seed, width=10).grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(ch_box, text="Početna faza (rad):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ch_box, textvariable=self.var_init_phase, width=10).grid(row=4, column=1, sticky="w", pady=(6, 0))

        ttk.Button(left, text="Generiši TX (+ kanal)", command=self.on_generate).pack(fill="x", pady=(0, 8))

        # ---------------- Plots ----------------
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)

        self.fig_grid, self.ax_grid = plt.subplots(figsize=(7.8, 4.8))
        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, master=self.nb)
        tab1 = ttk.Frame(self.nb)
        self.canvas_grid.get_tk_widget().pack(in_=tab1, fill="both", expand=True)
        self.nb.add(tab1, text="Resource grid")

        self.fig_wave, self.ax_wave = plt.subplots(figsize=(7.8, 4.8))
        self.canvas_wave = FigureCanvasTkAgg(self.fig_wave, master=self.nb)
        tab2 = ttk.Frame(self.nb)
        self.canvas_wave.get_tk_widget().pack(in_=tab2, fill="both", expand=True)
        self.nb.add(tab2, text="Waveform")

        self.fig_spec, self.ax_spec = plt.subplots(figsize=(7.8, 4.8))
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.nb)
        tab3 = ttk.Frame(self.nb)
        self.canvas_spec.get_tk_widget().pack(in_=tab3, fill="both", expand=True)
        self.nb.add(tab3, text="Spectrum")

    def _parse_mib(self) -> np.ndarray | None:
        s = self.var_mib_bits.get().strip()
        if s == "":
            return None
        if len(s) != 24 or any(ch not in "01" for ch in s):
            raise ValueError("MIB mora biti string od tačno 24 bita (0/1).")
        return np.array([int(ch) for ch in s], dtype=np.uint8)

    def on_generate(self) -> None:
        try:
            nid2 = int(self.var_nid2.get())
            normal_cp = bool(self.var_normal_cp.get())
            include_pbch = bool(self.var_include_pbch.get())
            frame_mod4 = int(self.var_frame_mod4.get()) % 4

            tx = LTETxChain(nid2=nid2, ndlrb=6, num_subframes=1, normal_cp=normal_cp)

            mib_bits = self._parse_mib()
            if include_pbch and mib_bits is None:
                mib_bits = np.random.randint(0, 2, 24, dtype=np.uint8)

            self.tx_wave, self.fs = tx.generate_waveform(mib_bits=mib_bits if include_pbch else None,
                                                        frame_mod4=frame_mod4)
            self.grid = tx.grid

            self.rx_wave = None
            if self.var_apply_channel.get():
                seed_str = self.var_seed.get().strip()
                seed = None if seed_str == "" else int(seed_str)

                ch = LTEChannel(freq_offset_hz=float(self.var_freq_offset.get()),
                                sample_rate_hz=float(self.fs),
                                snr_db=float(self.var_snr_db.get()),
                                seed=seed,
                                initial_phase_rad=float(self.var_init_phase.get()))
                self.rx_wave = ch.apply(self.tx_wave)

            self._update_plots()

        except Exception as e:
            messagebox.showerror("Greška", str(e))

    def _update_plots(self) -> None:
        assert self.grid is not None and self.tx_wave is not None and self.fs is not None

        # Grid
        self.ax_grid.clear()
        gabs = np.abs(self.grid)
        im = self.ax_grid.imshow(gabs, origin="lower", aspect="auto", interpolation="nearest")
        self.ax_grid.set_title("|RE| na resource grid-u")
        self.ax_grid.set_xlabel("OFDM simbol l")
        self.ax_grid.set_ylabel("Subcarrier k")
        self.fig_grid.colorbar(im, ax=self.ax_grid)

        dc = gabs.shape[0] // 2
        self.ax_grid.axhline(dc, linestyle=":", linewidth=1.0)
        self.canvas_grid.draw()

        # Waveform
        self.ax_wave.clear()
        n = min(self.tx_wave.size, 6000)
        t = np.arange(n) / self.fs
        self.ax_wave.plot(t, np.abs(self.tx_wave[:n]), label="TX |x[n]|")
        if self.rx_wave is not None:
            self.ax_wave.plot(t, np.abs(self.rx_wave[:n]), label="RX |y[n]|")
        self.ax_wave.grid(True)
        self.ax_wave.legend()
        self.ax_wave.set_title("Talasni oblik")
        self.canvas_wave.draw()

        # Spectrum
        self.ax_spec.clear()
        nfft = 4096 if self.tx_wave.size >= 4096 else int(2 ** np.ceil(np.log2(max(self.tx_wave.size, 32))))
        f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / self.fs))

        def plot_fft(sig: np.ndarray, label: str) -> None:
            X = np.fft.fftshift(np.fft.fft(sig[:nfft], nfft))
            mag_db = 20 * np.log10(np.maximum(np.abs(X), 1e-12))
            self.ax_spec.plot(f / 1e3, mag_db, label=label)

        plot_fft(self.tx_wave, "TX")
        if self.rx_wave is not None:
            plot_fft(self.rx_wave, "RX")

        self.ax_spec.grid(True)
        self.ax_spec.legend()
        self.ax_spec.set_title("Spektar (FFT)")
        self.ax_spec.set_xlabel("f [kHz]")
        self.ax_spec.set_ylabel("|X(f)| [dB]")
        self.canvas_spec.draw()


def main() -> None:
    app = LTEGuiApp()
    app.mainloop()


if __name__ == "__main__":
    main()
