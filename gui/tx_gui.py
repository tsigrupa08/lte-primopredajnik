"""
TX GUI (LTE) — samo predajni lanac (TX)

Ovaj GUI radi:
- generiše PSS i mapira u resource grid
- (opcionalno) enkodira MIB (24 bita) -> PBCH simboli i mapira u grid
- radi OFDM modulaciju (IFFT + CP) i iscrtava rezultate

Pokretanje (iz root direktorija repoa):s
    python -m examples.tx_gui
ili:
    python examples/tx_gui.py

Ako dobiješ: ModuleNotFoundError: No module named 'transmitter'
- provjeri da si u root folderu repoa
- provjeri da postoji transmitter/__init__.py (prazan fajl je OK)
"""

from __future__ import annotations

import os
import sys
import traceback
import numpy as np

# ---------------------------------------------------------------------
# Osiguraj da importi rade i kada pokrećeš fajl direktno
# (dodaj root repo folder u sys.path)
# ---------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Tkinter + Matplotlib (embed u GUI)
# ---------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------
# Import tvojih TX modula (iz transmitter/)
# ---------------------------------------------------------------------
IMPORT_ERROR = None
try:
    from transmitter.LTETxChain import LTETxChain
    from transmitter.ofdm import OFDMModulator
except Exception as e:
    IMPORT_ERROR = e


class TxGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LTE TX GUI (PSS / PBCH / OFDM)")

        # -------------------- stanje --------------------
        self.last_grid: np.ndarray | None = None
        self.last_waveform: np.ndarray | None = None
        self.last_fs: float | int | None = None
        self.last_modulator: OFDMModulator | None = None

        # -------------------- UI varijable --------------------
        self.var_nid2 = tk.IntVar(value=0)
        self.var_ndlrb = tk.IntVar(value=6)
        self.var_normal_cp = tk.BooleanVar(value=True)
        self.var_num_subframes = tk.IntVar(value=1)

        self.var_enable_pbch = tk.BooleanVar(value=False)
        self.var_mib_bits = tk.StringVar(value="")  # 24 bita u tekstu

        self.var_spectrum_symbol = tk.IntVar(value=6)  # simbol za prikaz spektra

        # -------------------- layout --------------------
        self._build_layout()

        # početna poruka
        self.log("Spremno. Klikni 'Generiši TX'.")

        # ako import nije uspio, odmah prikaži korisnu poruku
        if IMPORT_ERROR is not None:
            self.log("[ERROR] Ne mogu importovati transmitter module.")
            self.log(str(IMPORT_ERROR))
            self.log("Provjeri da pokrećeš GUI iz root foldera repoa, i da imaš transmitter/__init__.py.")
            messagebox.showerror(
                "Import greška",
                "Ne mogu importovati transmitter module.\n"
                "Pokreni iz root foldera repoa i provjeri transmitter/__init__.py.\n\n"
                f"Detalji:\n{IMPORT_ERROR}",
            )

    # =================================================================
    # UI
    # =================================================================
    def _build_layout(self) -> None:
        self.root.geometry("1200x720")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        # lijevo: kontrole + log
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")

        # desno: tabovi sa grafovima
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        self._build_controls(left)
        self._build_plots(right)
        self._build_log(left)

    def _build_controls(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Parametri TX", padding=10)
        box.pack(fill="x", padx=5, pady=5)

        # NID2
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="N_ID_2 (0–2):", width=18).pack(side="left")
        ttk.Spinbox(row, from_=0, to=2, textvariable=self.var_nid2, width=6).pack(side="left")

        # NDLRB
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="NDLRB:", width=18).pack(side="left")
        ndlrb_combo = ttk.Combobox(row, textvariable=self.var_ndlrb, width=10, state="readonly")
        ndlrb_combo["values"] = (6, 15, 25, 50, 75, 100)
        ndlrb_combo.pack(side="left")
        ndlrb_combo.current(0)

        # CP
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Cyclic Prefix:", width=18).pack(side="left")
        ttk.Radiobutton(row, text="Normal", value=True, variable=self.var_normal_cp,
                        command=self._sync_spectrum_symbol_default).pack(side="left")
        ttk.Radiobutton(row, text="Extended", value=False, variable=self.var_normal_cp,
                        command=self._sync_spectrum_symbol_default).pack(side="left")

        # num_subframes
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Broj subfrejmova:", width=18).pack(side="left")
        ttk.Spinbox(row, from_=1, to=10, textvariable=self.var_num_subframes, width=6).pack(side="left")

        # PBCH enable
        row = ttk.Frame(box)
        row.pack(fill="x", pady=6)
        ttk.Checkbutton(row, text="Uključi PBCH (MIB 24 bita)", variable=self.var_enable_pbch).pack(side="left")

        # MIB bits entry
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="MIB bits (24):", width=18).pack(side="left")
        ttk.Entry(row, textvariable=self.var_mib_bits, width=28).pack(side="left", padx=3)
        ttk.Button(row, text="Random 24", command=self._fill_random_mib).pack(side="left")

        # Spectrum symbol selection (za tab "Spectrum")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Spectrum simbol l:", width=18).pack(side="left")
        ttk.Spinbox(row, from_=0, to=200, textvariable=self.var_spectrum_symbol, width=6).pack(side="left")
        ttk.Button(row, text="Osvježi grafove", command=self._refresh_plots_only).pack(side="left", padx=6)

        # akcije
        row = ttk.Frame(box)
        row.pack(fill="x", pady=10)
        ttk.Button(row, text="Generiši TX", command=self.on_generate_tx).pack(side="left", padx=3)
        ttk.Button(row, text="Spasi .npz", command=self.on_save_npz).pack(side="left", padx=3)
        ttk.Button(row, text="Očisti", command=self.on_clear).pack(side="left", padx=3)

        # info label
        self.lbl_info = ttk.Label(box, text="fs: -   |   len(waveform): -")
        self.lbl_info.pack(fill="x", pady=5)

        self._sync_spectrum_symbol_default()

    def _build_log(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Log", padding=8)
        box.pack(fill="both", expand=True, padx=5, pady=5)

        self.txt_log = ScrolledText(box, height=18, width=52)
        self.txt_log.pack(fill="both", expand=True)

    def _build_plots(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: waveform
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="Waveform")

        self.fig_wav = Figure(figsize=(6, 4), dpi=100)
        self.ax_wav = self.fig_wav.add_subplot(111)
        self.ax_wav.set_title("OFDM waveform (Re/Im)")
        self.ax_wav.set_xlabel("n")
        self.ax_wav.set_ylabel("amplituda")

        self.canvas_wav = FigureCanvasTkAgg(self.fig_wav, master=tab1)
        self.canvas_wav.get_tk_widget().pack(fill="both", expand=True)

        # Tab 2: grid
        tab2 = ttk.Frame(self.notebook)
        self.notebook.add(tab2, text="Resource Grid")

        self.fig_grid = Figure(figsize=(6, 4), dpi=100)
        self.ax_grid = self.fig_grid.add_subplot(111)
        self.ax_grid.set_title("|Grid| (magnituda)")
        self.ax_grid.set_xlabel("OFDM simbol (l)")
        self.ax_grid.set_ylabel("subcarrier indeks (k)")

        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, master=tab2)
        self.canvas_grid.get_tk_widget().pack(fill="both", expand=True)

        # Tab 3: spectrum (IFFT input bins za izabrani simbol)
        tab3 = ttk.Frame(self.notebook)
        self.notebook.add(tab3, text="Spectrum (IFFT bins)")

        self.fig_spec = Figure(figsize=(6, 4), dpi=100)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.ax_spec.set_title("IFFT ulaz (fftshift) — vidi DC rupu")
        self.ax_spec.set_xlabel("bin indeks (fftshift)")
        self.ax_spec.set_ylabel("|X[k]|")

        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=tab3)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)

    # =================================================================
    # Helpers
    # =================================================================
    def log(self, msg: str) -> None:
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")

    def _fill_random_mib(self) -> None:
        bits = np.random.randint(0, 2, size=24)
        self.var_mib_bits.set("".join(str(int(b)) for b in bits))
        self.log("Popunjeno: random 24 MIB bita.")

    def _sync_spectrum_symbol_default(self) -> None:
        # PSS simbol: normal -> 6, extended -> 5 (za prvi subframe)
        self.var_spectrum_symbol.set(6 if self.var_normal_cp.get() else 5)

    def _parse_mib_bits(self) -> np.ndarray:
        s = self.var_mib_bits.get().strip().replace(" ", "")
        if s == "":
            # ako nije unijeto, napravimo random
            bits = np.random.randint(0, 2, size=24)
            self.var_mib_bits.set("".join(str(int(b)) for b in bits))
            return bits.astype(np.uint8)

        if any(c not in "01" for c in s):
            raise ValueError("MIB bits smiju biti samo '0' i '1' (bez slova).")

        if len(s) != 24:
            raise ValueError(f"MIB mora imati tačno 24 bita. Trenutno: {len(s)}")

        return np.array([int(c) for c in s], dtype=np.uint8)

    def _build_ifft_input_for_symbol(self, grid: np.ndarray, mod: OFDMModulator, sym_idx: int) -> np.ndarray:
        """
        Rekonstruiše IFFT ulaz (N binova) za dati OFDM simbol, na isti način
        kao u OFDMModulator.modulate().
        """
        if sym_idx < 0 or sym_idx >= grid.shape[1]:
            raise ValueError(f"symbol index l={sym_idx} je van opsega (0..{grid.shape[1]-1}).")

        N = mod.N
        num_sc = mod.num_subcarriers

        ifft_input = np.zeros(N, dtype=complex)
        dc_index = N // 2
        ifft_input[dc_index] = 0.0

        pos_freq_indices = np.arange(dc_index + 1, dc_index + 1 + num_sc // 2)
        neg_freq_indices = np.arange(dc_index - num_sc // 2, dc_index)

        pos_subcarriers = np.arange(num_sc // 2, num_sc)
        neg_subcarriers = np.arange(0, num_sc // 2)

        ifft_input[pos_freq_indices] = grid[pos_subcarriers, sym_idx]
        ifft_input[neg_freq_indices] = grid[neg_subcarriers, sym_idx]

        return ifft_input

    # =================================================================
    # Actions
    # =================================================================
    def on_clear(self) -> None:
        self.last_grid = None
        self.last_waveform = None
        self.last_fs = None
        self.last_modulator = None

        self.ax_wav.clear()
        self.ax_grid.clear()
        self.ax_spec.clear()

        self.ax_wav.set_title("OFDM waveform (Re/Im)")
        self.ax_grid.set_title("|Grid| (magnituda)")
        self.ax_spec.set_title("IFFT ulaz (fftshift) — vidi DC rupu")

        self.canvas_wav.draw()
        self.canvas_grid.draw()
        self.canvas_spec.draw()

        self.lbl_info.config(text="fs: -   |   len(waveform): -")
        self.log("Očišćeno.")

    def _refresh_plots_only(self) -> None:
        if self.last_grid is None or self.last_modulator is None:
            self.log("Nema prethodnih rezultata za refresh. Prvo klikni 'Generiši TX'.")
            return
        try:
            self._update_plots()
            self.log("Grafovi osvježeni.")
        except Exception as e:
            self.log(f"[ERROR] Ne mogu osvježiti grafove: {e}")

    def on_generate_tx(self) -> None:
        if IMPORT_ERROR is not None:
            self.log("[ERROR] Import modula nije uspio — ne mogu generisati TX.")
            return

        self.log("---- Generišem TX ----")
        try:
            n_id_2 = int(self.var_nid2.get())
            ndlrb = int(self.var_ndlrb.get())
            normal_cp = bool(self.var_normal_cp.get())
            num_subframes = int(self.var_num_subframes.get())
            enable_pbch = bool(self.var_enable_pbch.get())

            mib_bits = None
            if enable_pbch:
                mib_bits = self._parse_mib_bits()

            # Kreiraj TX chain
            tx = LTETxChain(
                n_id_2=n_id_2,
                ndlrb=ndlrb,
                num_subframes=num_subframes,
                normal_cp=normal_cp,
            )

            # generate_waveform vraća (waveform, fs)
            waveform, fs = tx.generate_waveform(mib_bits=mib_bits)

            # zapamti rezultate
            self.last_grid = tx.grid
            self.last_waveform = waveform
            self.last_fs = fs

            # modulator samo za N / mapping (za spectrum tab)
            self.last_modulator = OFDMModulator(self.last_grid)

            self.lbl_info.config(text=f"fs: {fs} Hz   |   len(waveform): {len(waveform)}")
            self.log(f"OK: grid shape={self.last_grid.shape}, waveform len={len(waveform)}, fs={fs}")

            self._update_plots()

        except Exception as e:
            self.log("[ERROR] " + str(e))
            self.log(traceback.format_exc())

    def _update_plots(self) -> None:
        # ---------- Waveform ----------
        self.ax_wav.clear()
        self.ax_wav.set_title("OFDM waveform (Re/Im)")
        self.ax_wav.set_xlabel("n")
        self.ax_wav.set_ylabel("amplituda")

        if self.last_waveform is not None:
            x = self.last_waveform
            n_plot = min(len(x), 5000)
            self.ax_wav.plot(np.real(x[:n_plot]), label="Re")
            self.ax_wav.plot(np.imag(x[:n_plot]), label="Im")
            self.ax_wav.legend(loc="upper right")
            self.ax_wav.grid(True)

        self.canvas_wav.draw()

        # ---------- Grid ----------
        self.ax_grid.clear()
        self.ax_grid.set_title("|Grid| (magnituda)")
        self.ax_grid.set_xlabel("OFDM simbol (l)")
        self.ax_grid.set_ylabel("subcarrier indeks (k)")

        if self.last_grid is not None:
            gabs = np.abs(self.last_grid)
            im = self.ax_grid.imshow(gabs, aspect="auto", origin="lower", interpolation="nearest")
            self.fig_grid.colorbar(im, ax=self.ax_grid, fraction=0.046, pad=0.04)
        self.canvas_grid.draw()

        # ---------- Spectrum (IFFT input bins) ----------
        self.ax_spec.clear()
        self.ax_spec.set_title("IFFT ulaz (fftshift) — vidi DC rupu")
        self.ax_spec.set_xlabel("bin indeks (fftshift)")
        self.ax_spec.set_ylabel("|X[k]|")

        if self.last_grid is not None and self.last_modulator is not None:
            l = int(self.var_spectrum_symbol.get())
            ifft_in = self._build_ifft_input_for_symbol(self.last_grid, self.last_modulator, l)
            spec = np.fft.fftshift(ifft_in)
            self.ax_spec.plot(np.abs(spec))
            self.ax_spec.grid(True)

            # označi DC (sredina nakon fftshift)
            dc = len(spec) // 2
            self.ax_spec.axvline(dc, linestyle="--")
            self.ax_spec.text(dc + 2, np.max(np.abs(spec)) * 0.9, "DC", fontsize=9)

        self.canvas_spec.draw()

    def on_save_npz(self) -> None:
        if self.last_waveform is None or self.last_grid is None or self.last_fs is None:
            self.log("Nema šta da se snimi. Prvo generiši TX.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")],
            title="Spasi TX rezultate",
        )
        if not path:
            return

        np.savez(
            path,
            waveform=self.last_waveform,
            fs=self.last_fs,
            grid=self.last_grid,
        )
        self.log(f"Snimljeno: {path}")


def main() -> None:
    root = tk.Tk()

    # malo ljepši ttk izgled
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = TxGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
