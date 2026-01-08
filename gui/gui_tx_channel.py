# gui_tx_channel.py
# Kako pokrenuti:
#pip install streamlit matplotlib numpy
##streamlit run gui/gui_tx_channel.py
from __future__ import annotations

import json
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------
# Path setup: pronađi PROJECT_ROOT koji sadrži folder "transmitter"
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
while not (PROJECT_ROOT / "transmitter").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from transmitter.LTETxChain import LTETxChain
from transmitter.ofdm import OFDMModulator
from channel.lte_channel import LTEChannel


# ---------------------------------------------------------------------
# Helperi
# ---------------------------------------------------------------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def compute_symbol_starts_from_ofdm(ofdm: OFDMModulator) -> List[int]:
    """Start indeksi (početak CP) za svaki OFDM simbol u waveform-u."""
    starts = []
    idx = 0
    for sym_idx in range(ofdm.num_ofdm_symbols):
        starts.append(idx)
        cp_len = int(ofdm.cp_lengths[sym_idx % ofdm.n_symbols_per_slot])
        idx += ofdm.N + cp_len
    return starts


def build_ifft_input_bins(grid_sym: np.ndarray, N: int) -> np.ndarray:
    """
    Rekonstruiše IFFT ulaz (bins) iz jednog OFDM simbola resource grida,
    identično logici u vašem OFDMModulatoru:
    - DC bin = 0
    - negativne frekvencije: subcarriers [0..half-1] -> bins [dc-half .. dc-1]
    - pozitivne frekvencije: subcarriers [half..end] -> bins [dc+1 .. dc+half]
    """
    num_subcarriers = grid_sym.size
    half = num_subcarriers // 2
    dc = N // 2

    ifft_in = np.zeros(N, dtype=np.complex128)
    ifft_in[dc] = 0.0

    pos_freq_bins = np.arange(dc + 1, dc + 1 + half)
    neg_freq_bins = np.arange(dc - half, dc)

    pos_sub = np.arange(half, num_subcarriers)
    neg_sub = np.arange(0, half)

    ifft_in[pos_freq_bins] = grid_sym[pos_sub]
    ifft_in[neg_freq_bins] = grid_sym[neg_sub]
    return ifft_in


def safe_parse_mib_bits(bitstr: str) -> Optional[np.ndarray]:
    """
    Prihvata string dužine 24, npr: "0101...".
    Vraća np.ndarray int {0,1} ili None ako nije validno.
    """
    s = bitstr.strip().replace(" ", "")
    if len(s) != 24:
        return None
    if any(ch not in "01" for ch in s):
        return None
    return np.array([int(ch) for ch in s], dtype=int)


def estimate_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Procjena SNR = P_signal / P_noise, gdje je noise = noisy-clean.
    Radi smisleno kad je clean i noisy poravnato (npr. CFO isti ili CFO=0).
    """
    clean = np.asarray(clean, dtype=np.complex128)
    noisy = np.asarray(noisy, dtype=np.complex128)
    n = noisy - clean
    p_sig = float(np.mean(np.abs(clean) ** 2))
    p_n = float(np.mean(np.abs(n) ** 2))
    if p_n <= 0:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_n)


@dataclass
class TxConfig:
    ndlrb: int
    normal_cp: bool
    num_subframes: int
    n_id_2: int
    pbch_enabled: bool
    mib_mode: str  # "random" | "manual"
    mib_seed: int
    mib_manual_bits: str


@dataclass
class ChannelConfig:
    freq_offset_hz: float
    snr_db: float
    seed: int
    initial_phase_rad: float


@dataclass
class RunConfig:
    tx: TxConfig
    ch: ChannelConfig


@st.cache_data(show_spinner=False)
def run_tx_channel(cfg: RunConfig) -> dict:
    """
    Generiše TX waveform + grid, zatim provuče kroz LTEChannel.
    Vraća dict sa rezultatima za GUI.
    """
    # --- MIB bits ---
    mib_bits = None
    if cfg.tx.pbch_enabled:
        if cfg.tx.mib_mode == "random":
            rng = np.random.default_rng(cfg.tx.mib_seed)
            mib_bits = rng.integers(0, 2, size=24, dtype=int).tolist()
        else:
            parsed = safe_parse_mib_bits(cfg.tx.mib_manual_bits)
            if parsed is None:
                raise ValueError("Manual MIB nije validan: treba 24 bita (0/1).")
            mib_bits = parsed.tolist()

    # --- TX ---
    tx = LTETxChain(
        n_id_2=cfg.tx.n_id_2,
        ndlrb=cfg.tx.ndlrb,
        num_subframes=cfg.tx.num_subframes,
        normal_cp=cfg.tx.normal_cp,
    )
    tx_waveform, fs = tx.generate_waveform(mib_bits=mib_bits)
    tx_waveform = np.asarray(tx_waveform, dtype=np.complex128)
    fs = float(fs)

    grid = np.asarray(tx.grid, dtype=np.complex128)

    # --- Channel ---
    ch = LTEChannel(
        freq_offset_hz=float(cfg.ch.freq_offset_hz),
        sample_rate_hz=fs,
        snr_db=float(cfg.ch.snr_db),
        seed=int(cfg.ch.seed),
        initial_phase_rad=float(cfg.ch.initial_phase_rad),
    )
    rx_waveform = ch.apply(tx_waveform)
    rx_waveform = np.asarray(rx_waveform, dtype=np.complex128)

    # --- OFDM metadata ---
    ofdm = OFDMModulator(grid)
    starts = compute_symbol_starts_from_ofdm(ofdm)

    # PSS i PBCH simbol indeksi po LTETxChain logici
    pss_sym_idx = tx._pss_symbol_index()
    pbch_syms = []
    if cfg.tx.pbch_enabled:
        # mapiranje 4x240 kroz 4 subfrejma: PBCH simboli u svakom subfrejmu
        for sf in range(min(cfg.tx.num_subframes, 4)):
            pbch_syms.extend(tx._pbch_symbol_indices_for_subframe(sf))

    return {
        "fs": fs,
        "tx_waveform": tx_waveform,
        "rx_waveform": rx_waveform,
        "grid": grid,
        "Nfft": ofdm.N,
        "cp_lengths": list(ofdm.cp_lengths),
        "symbols_per_subframe": (14 if cfg.tx.normal_cp else 12),
        "num_symbols_total": ofdm.num_ofdm_symbols,
        "symbol_starts": starts,
        "pss_symbol_index": int(pss_sym_idx),
        "pbch_symbol_indices": pbch_syms,
    }


# ---------------------------------------------------------------------
# Streamlit GUI
# ---------------------------------------------------------------------
st.set_page_config(page_title="LTE TX + Channel GUI", layout="wide")

st.title("LTE TX + Channel GUI (TX → Channel → RX)")
st.caption("Cilj: vizualno i metrikama provjeriti TX i kanal (CFO + AWGN) na više nivoa: grid → OFDM bins → vrijeme → spektar + sanity checks.")

# -----------------------------
# Sidebar: konfiguracija
# -----------------------------
st.sidebar.header("TX konfiguracija")

ndlrb = st.sidebar.selectbox("NDLRB", [6, 15, 25, 50, 75, 100], index=0)
normal_cp = st.sidebar.checkbox("Normal CP", value=True)
num_subframes = st.sidebar.slider("Broj subfrejmova", min_value=1, max_value=10, value=1, step=1)
n_id_2 = st.sidebar.selectbox("N_ID_2", [0, 1, 2], index=1)

st.sidebar.markdown("---")
pbch_enabled = st.sidebar.checkbox("Uključi PBCH (MIB)", value=False)
mib_mode = "random"
mib_seed = 123
mib_manual_bits = "0" * 24

if pbch_enabled:
    mib_mode = st.sidebar.radio("MIB mode", ["random", "manual"], index=0, horizontal=True)
    if mib_mode == "random":
        mib_seed = st.sidebar.number_input("MIB seed", min_value=0, max_value=2_000_000_000, value=123, step=1)
    else:
        mib_manual_bits = st.sidebar.text_input("Manual MIB (24 bita)", value="0" * 24)

st.sidebar.header("Kanal konfiguracija")
freq_offset_hz = st.sidebar.number_input("CFO / freq_offset_hz [Hz]", value=0.0, step=50.0)
snr_db = st.sidebar.number_input("SNR [dB]", value=20.0, step=1.0)
awgn_seed = st.sidebar.number_input("AWGN seed", min_value=0, max_value=2_000_000_000, value=123, step=1)
initial_phase = st.sidebar.number_input("Initial phase [rad]", value=0.0, step=0.1)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run TX → Channel", type="primary")

cfg = RunConfig(
    tx=TxConfig(
        ndlrb=int(ndlrb),
        normal_cp=bool(normal_cp),
        num_subframes=int(num_subframes),
        n_id_2=int(n_id_2),
        pbch_enabled=bool(pbch_enabled),
        mib_mode=str(mib_mode),
        mib_seed=int(mib_seed),
        mib_manual_bits=str(mib_manual_bits),
    ),
    ch=ChannelConfig(
        freq_offset_hz=float(freq_offset_hz),
        snr_db=float(snr_db),
        seed=int(awgn_seed),
        initial_phase_rad=float(initial_phase),
    ),
)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
    st.session_state["last_cfg"] = None

if run_btn:
    try:
        res = run_tx_channel(cfg)
        st.session_state["last_result"] = res
        st.session_state["last_cfg"] = cfg
        st.success("OK: TX i kanal su izvršeni.")
    except Exception as e:
        st.session_state["last_result"] = None
        st.session_state["last_cfg"] = None
        st.error(f"Greška pri izvršavanju: {e}")

res = st.session_state["last_result"]

tabs = st.tabs(["Overview", "Grid inspector", "OFDM bins", "Time waveform", "Spectrum", "Sanity checks", "Sweeps"])

# -----------------------------
# Tab: Overview
# -----------------------------
with tabs[0]:
    st.subheader("Overview")

    if res is None:
        st.info("Klikni **Run TX → Channel** da dobiješ rezultate.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("fs [Hz]", f"{res['fs']:.0f}")
        col2.metric("Nfft", f"{res['Nfft']}")
        col3.metric("Grid shape", f"{res['grid'].shape[0]} × {res['grid'].shape[1]}")
        col4.metric("Waveform length", f"{res['tx_waveform'].size}")

        st.write("**Očekivanje**")
        st.markdown(
            """
- **CFO** (freq_offset_hz) pravi rotaciju faze kroz vrijeme; amplituda se ne “mora” promijeniti.
- **AWGN** dodaje šum: na **nižem SNR** waveform i spektar postaju “šumovitiji”.
- Resource grid ti je “istina”: PSS/PBCH su mapirani tu, pa OFDM radi IFFT+CP.
            """.strip()
        )

        # Export config
        cfg_json = json.dumps(asdict(st.session_state["last_cfg"]), indent=2)
        st.download_button(
            "Download config.json",
            data=cfg_json.encode("utf-8"),
            file_name="tx_channel_config.json",
            mime="application/json",
        )


# -----------------------------
# Tab: Grid inspector
# -----------------------------
with tabs[1]:
    st.subheader("Resource grid inspector")

    if res is None:
        st.info("Prvo pokreni pipeline.")
    else:
        grid = res["grid"]
        n_sym = grid.shape[1]

        sym_sel = st.slider("Izaberi OFDM simbol (kolona grida)", 0, n_sym - 1, int(res["pss_symbol_index"]))
        show_db = st.checkbox("Prikaži magnitude u dB (log)", value=True)

        mag = np.abs(grid)
        if show_db:
            eps = 1e-12
            mag = 20 * np.log10(mag + eps)

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(mag, aspect="auto", origin="lower")
        ax.set_title("Grid magnitude (subcarrier × symbol)")
        ax.set_xlabel("OFDM simbol (l)")
        ax.set_ylabel("Subcarrier indeks (k)")
        ax.axvline(sym_sel, linestyle="--")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        # Oznake PSS/PBCH simbola
        ax.axvline(res["pss_symbol_index"], linestyle=":", linewidth=2)
        for l in res["pbch_symbol_indices"]:
            ax.axvline(l, linestyle=":", linewidth=1)

        st.pyplot(fig)

        st.markdown(
            f"""
**Očekivano na gridu:**
- **PSS** je u simbolu **l = {res['pss_symbol_index']}** (označeno debljom tačkastom linijom).
- Ako je PBCH uključen, PBCH simboli su označeni tankim tačkastim linijama.
- U odabranom simbolu (isprekidana linija) možeš vizuelno vidjeti gdje je energija.
            """.strip()
        )

        # Snapshot odabranog simbola
        sym_vec = grid[:, sym_sel]
        fig2 = plt.figure(figsize=(12, 3))
        ax2 = fig2.add_subplot(111)
        ax2.plot(np.abs(sym_vec))
        ax2.set_title(f"|grid[:, {sym_sel}]| po subcarrierima")
        ax2.set_xlabel("k")
        ax2.set_ylabel("|.|")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.download_button(
            "Download grid snapshot (PNG)",
            data=fig_to_png_bytes(fig2),
            file_name=f"grid_symbol_{sym_sel}.png",
            mime="image/png",
        )


# -----------------------------
# Tab: OFDM bins (IFFT input)
# -----------------------------
with tabs[2]:
    st.subheader("OFDM bins (IFFT input)")

    if res is None:
        st.info("Prvo pokreni pipeline.")
    else:
        grid = res["grid"]
        N = int(res["Nfft"])
        n_sym = grid.shape[1]

        sym_sel = st.slider("Simbol za OFDM bins", 0, n_sym - 1, int(res["pss_symbol_index"]), key="bins_sym_sel")

        ifft_in = build_ifft_input_bins(grid[:, sym_sel], N)
        dc = N // 2

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        ax.plot(np.abs(ifft_in))
        ax.axvline(dc, linestyle="--", label="DC bin")
        ax.set_title(f"|IFFT input bins| za simbol l={sym_sel} (DC mora biti 0)")
        ax.set_xlabel("bin index")
        ax.set_ylabel("|.|")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            """
**Šta ovdje trebaš vidjeti:**
- IFFT ulaz ima energiju “oko DC”, ali je **DC bin uvijek 0** (ne koristi se).
- Ovo je odličan dokaz da je mapping grida u IFFT bins konzistentan.
            """.strip()
        )


# -----------------------------
# Tab: Time waveform
# -----------------------------
# -----------------------------
# Tab: Time waveform (TX i RX odvojeno)
# -----------------------------
with tabs[3]:
    st.subheader("Time waveform — TX i RX odvojeno (subplots)")

    if res is None:
        st.info("Prvo pokreni pipeline.")
    else:
        tx_w = res["tx_waveform"]
        rx_w = res["rx_waveform"]

        # Zoom kontrola
        max_start = max(0, tx_w.size - 1)
        start_idx = st.slider("Start sample", 0, max_start, 0, key="time_start")
        win_len = st.slider("Window length", 200, min(50000, tx_w.size), 5000, step=100, key="time_win")
        end_idx = min(tx_w.size, start_idx + win_len)

        seg_tx = tx_w[start_idx:end_idx]
        seg_rx = rx_w[start_idx:end_idx]
        x = np.arange(start_idx, end_idx)

        fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

        # --- TX ---
        axs[0, 0].plot(x, np.real(seg_tx))
        axs[0, 0].set_title("TX real")
        axs[0, 0].set_ylabel("amplitude")
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(x, np.abs(seg_tx))
        axs[0, 1].set_title("|TX|")
        axs[0, 1].grid(True, alpha=0.3)

        # --- RX ---
        axs[1, 0].plot(x, np.real(seg_rx))
        axs[1, 0].set_title("RX real")
        axs[1, 0].set_xlabel("sample index")
        axs[1, 0].set_ylabel("amplitude")
        axs[1, 0].grid(True, alpha=0.3)

        axs[1, 1].plot(x, np.abs(seg_rx))
        axs[1, 1].set_title("|RX|")
        axs[1, 1].set_xlabel("sample index")
        axs[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Time-domain window [{start_idx}:{end_idx}] | "
            f"CFO={cfg.ch.freq_offset_hz:.1f} Hz, SNR={cfg.ch.snr_db:.1f} dB",
            y=0.98,
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
**Šta trebaš vidjeti:**
- **TX real** je “čist”, a **RX real** postaje sve šumovitiji kad smanjiš SNR.
- **|TX|** i **|RX|**: AWGN pravi “nazubljen” envelope u RX-u.
- CFO najviše utiče na fazu → real dio RX-a može izgledati dosta drugačije i kad |.| ostane sličan.
            """.strip()
        )

        st.download_button(
            "Download time subplots (PNG)",
            data=fig_to_png_bytes(fig),
            file_name="time_tx_rx_separate_subplots.png",
            mime="image/png",
        )

# -----------------------------
# Tab: Spectrum
# -----------------------------
# -----------------------------
# Tab: Spectrum (subplots)
# -----------------------------
with tabs[4]:
    st.subheader("Spectrum — subplots")

    if res is None:
        st.info("Prvo pokreni pipeline.")
    else:
        fs = float(res["fs"])
        tx_w = res["tx_waveform"]
        rx_w = res["rx_waveform"]

        # Segment za FFT
        start_idx = st.slider("FFT segment start", 0, max(0, tx_w.size - 1), 0, key="spec_start")
        seg_len = st.selectbox("FFT segment length", [1024, 2048, 4096, 8192, 16384], index=2, key="spec_len")
        start_idx = min(start_idx, max(0, tx_w.size - seg_len))

        seg_tx = tx_w[start_idx:start_idx + seg_len]
        seg_rx = rx_w[start_idx:start_idx + seg_len]

        S_tx = np.fft.fftshift(np.fft.fft(seg_tx))
        S_rx = np.fft.fftshift(np.fft.fft(seg_rx))
        f = np.fft.fftshift(np.fft.fftfreq(seg_len, d=1.0 / fs))

        mag_tx_db = 20 * np.log10(np.abs(S_tx) + 1e-12)
        mag_rx_db = 20 * np.log10(np.abs(S_rx) + 1e-12)

        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axs[0].plot(f, mag_tx_db)
        axs[0].set_title("TX |FFT| (dB)")
        axs[0].set_ylabel("dB")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(f, mag_rx_db)
        axs[1].set_title("RX |FFT| (dB)")
        axs[1].set_xlabel("frequency [Hz]")
        axs[1].set_ylabel("dB")
        axs[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Spectrum view (segment [{start_idx}:{start_idx + seg_len}]) | "
            f"CFO={cfg.ch.freq_offset_hz:.1f} Hz, SNR={cfg.ch.snr_db:.1f} dB",
            y=0.98,
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
**Očekivano:**
- Sa manjim SNR, RX “noise floor” raste (donji subplot postaje šumovitiji).
- CFO je vremenska rotacija faze; u spektru može djelovati kao “pomak/neravnoteža” zavisno od segmenta,
  ali ključna stvar je da RX izgleda “prljavije” kad je SNR manji.
            """.strip()
        )

        st.download_button(
            "Download spectrum subplots (PNG)",
            data=fig_to_png_bytes(fig),
            file_name="spectrum_subplots.png",
            mime="image/png",
        )


# -----------------------------
# Tab: Sanity checks
# -----------------------------
with tabs[5]:
    st.subheader("Sanity checks (must-pass)")

    if res is None:
        st.info("Prvo pokreni pipeline.")
    else:
        fs = float(res["fs"])
        grid = res["grid"]
        N = int(res["Nfft"])
        tx_w = res["tx_waveform"]
        rx_w = res["rx_waveform"]

        checks = []

        # 1) grid dim
        checks.append(("Grid rows == 12*NDLRB", grid.shape[0] == 12 * int(cfg.tx.ndlrb), f"rows={grid.shape[0]}"))

        # 2) symbols total
        exp_cols = (14 if cfg.tx.normal_cp else 12) * int(cfg.tx.num_subframes)
        checks.append(("Grid cols == symbols_per_subframe*num_subframes", grid.shape[1] == exp_cols, f"cols={grid.shape[1]} expected={exp_cols}"))

        # 3) fs
        checks.append(("fs == 15000*Nfft", abs(fs - 15000 * N) < 1e-6, f"fs={fs:.0f}, 15000*N={15000*N}"))

        # 4) waveform length match with OFDMModulator output_length
        ofdm = OFDMModulator(grid)
        checks.append(("len(waveform) == ofdm.output_length", tx_w.size == int(ofdm.output_length), f"len={tx_w.size}, outlen={ofdm.output_length}"))

        # 5) RX shape equals TX shape
        checks.append(("RX length equals TX length", rx_w.size == tx_w.size, f"tx={tx_w.size}, rx={rx_w.size}"))

        # 6) PBCH rule (ako uključen)
        if cfg.tx.pbch_enabled:
            checks.append(("PBCH requires num_subframes>=4 (simplification)", int(cfg.tx.num_subframes) >= 4, f"num_subframes={cfg.tx.num_subframes}"))

        # 7) SNR estimate sanity (samo kad CFO=0)
        if abs(cfg.ch.freq_offset_hz) < 1e-12:
            snr_est = estimate_snr_db(tx_w, rx_w)
            checks.append(("Measured SNR approx input SNR (CFO=0)", abs(snr_est - cfg.ch.snr_db) < 3.0, f"snr_est={snr_est:.2f} dB"))

        # Render
        for name, ok, detail in checks:
            if ok:
                st.success(f"PASS: {name}  —  {detail}")
            else:
                st.error(f"FAIL: {name}  —  {detail}")

        st.markdown(
            """
Ako ti nešto FAIL-a ovdje, to je znak da:
- grid shape nije konzistentan s konfiguracijom,
- OFDM mapping ili CP pattern nije usklađen,
- ili PBCH “pravila projekta” (4 subfrejma) nisu ispoštovana.
            """.strip()
        )


# -----------------------------
# Tab: Sweeps / Stress test
# -----------------------------
with tabs[6]:
    st.subheader("Sweeps / Stress test")

    st.markdown(
        """
Ovdje možeš brzo “istrčati” set testova i dobiti tabelu:
- CFO i SNR kombinacije
- procijenjeni SNR (poređenjem RX sa “clean CFO-only” signalom)

**Napomena:** za realno mjerenje SNR-a u prisustvu CFO-a, mi prvo napravimo “clean” signal
koji je samo CFO primijenjen, pa tek onda noise = RX - clean.
        """.strip()
    )

    if res is None:
        st.info("Prvo pokreni bar jedan run (da imamo konfiguraciju i fs).")
    else:
        sweep_snr = st.multiselect("SNR vrijednosti [dB]", options=[0, 2, 5, 10, 15, 20, 30, 40], default=[5, 10, 15, 20])
        sweep_cfo = st.multiselect("CFO vrijednosti [Hz]", options=[0, 500, 1500, 5000, -5000], default=[0, 1500, 5000])

        if st.button("Run sweep"):
            fs = float(res["fs"])
            tx_w = res["tx_waveform"]

            rows = []
            for cfo in sweep_cfo:
                # clean CFO-only signal
                ch_clean = LTEChannel(freq_offset_hz=float(cfo), sample_rate_hz=fs, snr_db=200.0, seed=0, initial_phase_rad=float(cfg.ch.initial_phase_rad))
                clean = ch_clean.apply(tx_w)

                for snr in sweep_snr:
                    ch = LTEChannel(freq_offset_hz=float(cfo), sample_rate_hz=fs, snr_db=float(snr), seed=int(cfg.ch.seed), initial_phase_rad=float(cfg.ch.initial_phase_rad))
                    rx = ch.apply(tx_w)

                    # SNR estimate relative to clean CFO-only
                    snr_est = estimate_snr_db(clean, rx)

                    rows.append((cfo, snr, snr_est))

            # Show table
            st.write("Rezultati (CFO, SNR_in, SNR_est):")
            st.dataframe(
                {
                    "CFO [Hz]": [r[0] for r in rows],
                    "SNR_in [dB]": [r[1] for r in rows],
                    "SNR_est [dB]": [float(r[2]) for r in rows],
                }
            )
