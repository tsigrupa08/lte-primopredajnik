# gui_tx_channel.py
"""
LTE Pro Analyzer - Glavna GUI Aplikacija
========================================

Ova skripta pokreće Streamlit aplikaciju koja simulira kompletan LTE komunikacijski lanac:
Transmitter (TX) -> Channel (Kanal) -> Receiver (RX).

Funkcionalnosti:
    - Generisanje LTE resursnog grida (PSS, SSS, CRS, PBCH, Podaci).
    - Simulacija kanala (AWGN šum, frekvencijski ofset - CFO).
    - Prijemnik sa sinhronizacijom (PSS korelacija) i korekcijom frekvencije.
    - Napredna vizualizacija: Resource Map, Konstelacijski dijagram (EVM), OFDM spektar.

Pokretanje:
    $ streamlit run gui_tx_channel.py
"""

from __future__ import annotations

import json
import io
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import streamlit as st


# 1. SETUP PUTANJA
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
while not (PROJECT_ROOT / "transmitter").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importi modula uz hvatanje grešaka
try:
    from transmitter.LTETxChain import LTETxChain
    from transmitter.ofdm import OFDMModulator
    from channel.lte_channel import LTEChannel
    from receiver.LTERxChain import LTERxChain
    from receiver.pss_sync import PSSSynchronizer
    from receiver.OFDM_demodulator import OFDMDemodulator
    from receiver.resource_grid_extractor import (
        PBCHConfig, PBCHExtractor, pbch_symbol_indices_for_subframes
    )
    from channel.frequency_offset import FrequencyOffset
except ImportError as e:
    st.error(f"Greška pri importu modula! Provjeri putanje. Detalji: {e}")
    st.stop()

# 2. HELPER FUNKCIJE (VISUALIZATION & UTILS)

def generate_resource_map(grid_shape: Tuple[int, int], 
                          ndlrb: int, 
                          n_id_2: int, 
                          num_subframes: int, 
                          normal_cp: bool) -> np.ndarray:
    """
    Generiše matricu (mapu) koja označava tip logičkog kanala za svaki element
    vremensko-frekvencijskog grida. Koristi se za vizualizaciju strukture okvira.

    Parameters
    ----------
    grid_shape : Tuple[int, int]
        Dimenzije grida u formatu (broj_podnosioca, broj_simbola).
    ndlrb : int
        Broj resursnih blokova (bandwidth), npr. 6, 15, 25, 50.
    n_id_2 : int
        Identitet grupe ćelije (0, 1 ili 2). Određuje pomak referentnih signala (CRS).
    num_subframes : int
        Broj subfrejmova koji se simuliraju.
    normal_cp : bool
        Ako je True, koristi Normal Cyclic Prefix (7 simbola po slotu).
        Ako je False, koristi Extended CP (6 simbola po slotu).

    Returns
    -------
    np.ndarray
        2D matrica (int) istih dimenzija kao grid. Vrijednosti označavaju:
        - 0: Podaci ili Prazno (Data/Empty)
        - 1: CRS (Cell-Specific Reference Signals - Piloti)
        - 2: Sinhronizacijski signali (PSS/SSS)
        - 3: Broadcast kanal (PBCH)
    """
    n_sc, n_sym = grid_shape
    resource_map = np.zeros(grid_shape, dtype=int)
    
    n_sym_per_slot = 7 if normal_cp else 6
    n_sym_per_sf = n_sym_per_slot * 2
    center_sc = n_sc // 2
    
    # --- 1. CRS (Cell Reference Signals - PILOTI) [ID: 1] ---
    # Shift po frekvenciji zavisi od N_ID_2 (mod 6)
    v_shift = n_id_2 % 6
    
    for sf in range(num_subframes):
        sf_start = sf * n_sym_per_sf
        for slot in range(2):
            slot_start = sf_start + slot * n_sym_per_slot
            # CRS se nalaze na simbolima 0 i 4 unutar slota (za Normal CP, Port 0)
            crs_syms = [0, 4] if normal_cp else [0, 3]
            
            for local_sym in crs_syms:
                abs_sym = slot_start + local_sym
                if abs_sym >= n_sym: continue
                # Frekvencijski pomak zavisi od simbola unutar slota
                k_shift = v_shift if local_sym == 0 else (v_shift + 3) % 6
                # Svaki 6. podnosilac je pilot
                resource_map[k_shift::6, abs_sym] = 1 

    # --- 2. PSS / SSS (Sync Signals) [ID: 2] ---
    # Nalaze se u subfrejmovima 0 i 5, zauzimaju centralna 62 podnosioca.
    pss_width = 62
    sc_start = center_sc - (pss_width // 2)
    sc_end = center_sc + (pss_width // 2)
    
    for sf in range(num_subframes):
        if sf in [0, 5]: 
            sf_start = sf * n_sym_per_sf
            # PSS: Zadnji simbol prvog slota u subfrejmu
            pss_sym = sf_start + (n_sym_per_slot - 1)
            # SSS: Simbol prije PSS-a
            sss_sym = pss_sym - 1
            
            if pss_sym < n_sym:
                resource_map[sc_start:sc_end, pss_sym] = 2 
            if sss_sym < n_sym:
                resource_map[sc_start:sc_end, sss_sym] = 2

    # --- 3. PBCH (Broadcast Channel) [ID: 3] ---
    # Nalazi se u Subfrejmu 0, Slot 1, prva 4 simbola. Centralna 72 podnosioca.
    pbch_width = 72
    pbch_sc_start = center_sc - (pbch_width // 2)
    pbch_sc_end = center_sc + (pbch_width // 2)
    
    if num_subframes > 0:
        # PBCH je uvijek u subfrejmu 0, počinje od drugog slota (indeks n_sym_per_slot)
        slot1_start = n_sym_per_slot 
        pbch_syms = [slot1_start + i for i in range(4)]
        
        for sym in pbch_syms:
            if sym < n_sym:
                current_col = resource_map[pbch_sc_start:pbch_sc_end, sym]
                # Prebriši sve što nije CRS (1) sa PBCH (3)
                # Ovo čuva CRS "rupe" unutar PBCH bloka
                mask = (current_col != 1)
                current_col[mask] = 3
                resource_map[pbch_sc_start:pbch_sc_end, sym] = current_col

    return resource_map

def safe_parse_mib_bits(bitstr: str) -> Optional[np.ndarray]:
    """
    Parsira string jedinica i nula u NumPy niz bitova.

    Parameters
    ----------
    bitstr : str
        Ulazni string (npr. "10101...").

    Returns
    -------
    Optional[np.ndarray]
        Vraća niz int-ova (0 ili 1) dužine 24 ako je parsiranje uspješno.
        Vraća None ako je string neispravan ili pogrešne dužine.
    """
    s = bitstr.strip().replace(" ", "")
    if len(s) != 24: return None
    if any(ch not in "01" for ch in s): return None
    return np.array([int(ch) for ch in s], dtype=int)

def build_ifft_input_bins(grid_sym: np.ndarray, N: int) -> np.ndarray:
    """
    Mapira aktivne podnosioce iz baseband grida na ulaze IFFT-a.
    Ovo uključuje umetanje DC nule i pomjeranje frekvencija (fftshift logika).

    Parameters
    ----------
    grid_sym : np.ndarray
        Jedan OFDM simbol iz grida (vektor kompleksnih brojeva).
    N : int
        Veličina IFFT-a (npr. 128, 256, 512...).

    Returns
    -------
    np.ndarray
        Vektor dužine N spreman za IFFT.
    """
    num_subcarriers = grid_sym.size
    half = num_subcarriers // 2
    dc = N // 2
    
    ifft_in = np.zeros(N, dtype=np.complex128)
    ifft_in[dc] = 0.0 # DC komponenta je nula u LTE basebandu
    
    # Mapiranje pozitivnih frekvencija (desna polovina grida)
    pos_freq_bins = np.arange(dc + 1, dc + 1 + half)
    pos_sub = np.arange(half, num_subcarriers)
    
    # Mapiranje negativnih frekvencija (lijeva polovina grida)
    neg_freq_bins = np.arange(dc - half, dc)
    neg_sub = np.arange(0, half)
    
    ifft_in[pos_freq_bins] = grid_sym[pos_sub]
    ifft_in[neg_freq_bins] = grid_sym[neg_sub]
    
    return ifft_in

# ---------------------------------------------------------------------
# 3. KONFIGURACIJA (DATA CLASSES)
# ---------------------------------------------------------------------
@dataclass
class TxConfig:
    """Konfiguracija predajnika."""
    ndlrb: int              # Broj Resource Blockova
    normal_cp: bool         # Tip cikličnog prefiksa
    num_subframes: int      # Trajanje simulacije
    n_id_2: int             # ID unutar grupe (0-2)
    pbch_enabled: bool      # Da li generisati PBCH
    mib_mode: str           # "random" ili "manual"
    mib_seed: int           # Seed za random MIB
    mib_manual_bits: str    # String bitova za manual mode

@dataclass
class ChannelConfig:
    """Konfiguracija kanala."""
    freq_offset_hz: float   # Frekvencijski ofset (CFO)
    snr_db: float           # Odnos signal-šum
    seed: int               # Seed za šum
    initial_phase_rad: float

@dataclass
class RxConfig:
    """Konfiguracija prijemnika."""
    enable_cfo_correction: bool # Da li vršiti korekciju frekvencije

@dataclass
class RunConfig:
    """Objedinjena konfiguracija za cijelu simulaciju."""
    tx: TxConfig
    ch: ChannelConfig
    rx: RxConfig

# 4. GLAVNA LOGIKA SIMULACIJE
@st.cache_data(show_spinner=False)
def run_simulation(cfg: RunConfig) -> Dict[str, Any]:
    """
    Izvršava kompletan LTE pipeline: TX -> Channel -> RX.
    Rezultati se keširaju radi performansi u Streamlitu.

    Parameters
    ----------
    cfg : RunConfig
        Objekat koji sadrži sve parametre za TX, Channel i RX.

    Returns
    -------
    Dict[str, Any]
        Rječnik sa svim rezultatima simulacije:
        - 'tx_waveform': Generisani signal (vremenska domena).
        - 'rx_waveform': Signal na prijemu (sa šumom i CFO).
        - 'grid': Originalni TX resursni grid.
        - 'rx_result': Rezultati dekodiranja (RXResult objekat).
        - 'pss_corr': Vektor korelacije za sinhronizaciju.
        - 'rx_pbch_symbols': Ekstraktovani PBCH simboli (za EVM).
    """
    results = {}
    
    # --- 1. TRANSMITTER (TX) ---
    mib_bits = None
    if cfg.tx.pbch_enabled:
        if cfg.tx.mib_mode == "random":
            rng = np.random.default_rng(cfg.tx.mib_seed)
            mib_bits = rng.integers(0, 2, size=24, dtype=int).tolist()
        else:
            parsed = safe_parse_mib_bits(cfg.tx.mib_manual_bits)
            if parsed is None: raise ValueError("Invalid Manual MIB")
            mib_bits = parsed.tolist()

    tx_chain = LTETxChain(
        n_id_2=cfg.tx.n_id_2, 
        ndlrb=cfg.tx.ndlrb,
        num_subframes=cfg.tx.num_subframes, 
        normal_cp=cfg.tx.normal_cp,
    )
    
    # Generisanje talasnog oblika
    tx_waveform, fs = tx_chain.generate_waveform(mib_bits=mib_bits)
    fs = float(fs)
    
    # Spremanje TX rezultata
    results['fs'] = fs
    results['tx_waveform'] = tx_waveform
    results['grid'] = tx_chain.grid
    results['mib_tx'] = mib_bits
    
    # Pomoćni modulator samo za dohvat Nfft veličine
    temp_ofdm = OFDMModulator(tx_chain.grid)
    results['Nfft'] = temp_ofdm.N
    results['pss_index'] = tx_chain._pss_symbol_index()

    # --- 2. CHANNEL (KANAL) ---
    channel = LTEChannel(
        freq_offset_hz=float(cfg.ch.freq_offset_hz), 
        sample_rate_hz=fs,
        snr_db=float(cfg.ch.snr_db), 
        seed=int(cfg.ch.seed),
        initial_phase_rad=float(cfg.ch.initial_phase_rad),
    )
    rx_waveform = channel.apply(tx_waveform)
    results['rx_waveform'] = rx_waveform

    # --- 3. RECEIVER (RX) PROCESSING ---
    rx_chain = LTERxChain(
        sample_rate_hz=fs, 
        ndlrb=cfg.tx.ndlrb, 
        normal_cp=cfg.tx.normal_cp,
        pci=0, # Inicijalni PCI, biće prebrisan detekcijom
        enable_cfo_correction=cfg.rx.enable_cfo_correction,
        pbch_spread_subframes=cfg.tx.num_subframes
    )
    
    t0 = time.time()
    rx_res = rx_chain.decode(rx_waveform)
    results['rx_result'] = rx_res
    results['process_time'] = time.time() - t0

    # --- 4. ANALITIKA (PSS, EVM, Constellation) ---
    
    # 4.1 PSS Korelacija (za vizualizaciju peak-ova)
    pss_sync = PSSSynchronizer(fs, ndlrb=cfg.tx.ndlrb, normal_cp=cfg.tx.normal_cp)
    rx_norm = rx_waveform / (np.sqrt(np.mean(np.abs(rx_waveform)**2)) + 1e-12)
    pss_corr = pss_sync.correlate(rx_norm)
    results['pss_corr'] = pss_corr

    # 4.2 Ekstrakcija PBCH simbola za EVM analizu
    # Ovo je dodatni korak koji simulira "savršenu" ekstrakciju nakon sinhronizacije
    # da bismo mogli nacrtati konstelacijski dijagram (QPSK tačke).
    results['rx_pbch_symbols'] = None
    
    if rx_res.tau_hat is not None:
        # Primijeni CFO korekciju ako je detektovana
        cfo = rx_res.cfo_hat if (cfg.rx.enable_cfo_correction and rx_res.cfo_hat) else 0.0
        fo_corr = FrequencyOffset(-cfo, fs)
        rx_cfo_corr = fo_corr.apply(rx_waveform)
        
        # Poravnanje (Time Alignment)
        start_idx = rx_res.tau_hat
        ofdm_demod = OFDMDemodulator(cfg.tx.ndlrb, cfg.tx.normal_cp)
        
        # Izračunaj koliko uzoraka treba vratiti unazad da se dođe na početak subfrejma
        sym_idx_pss = 6 if cfg.tx.normal_cp else 5
        samps_before = 0
        for i in range(sym_idx_pss):
            cp = ofdm_demod.cp_lengths[i % ofdm_demod.n_symbols_per_slot]
            samps_before += (ofdm_demod.fft_size + cp)
            
        start_sf = start_idx - samps_before
        if start_sf < 0: start_sf = 0
        rx_aligned = rx_cfo_corr[start_sf:]
        
        try:
            # Demodulacija cijelog niza
            grid_full = ofdm_demod.demodulate(rx_aligned)
            grid_active = ofdm_demod.extract_active_subcarriers(grid_full)
            
            # Ekstrakcija PBCH simbola
            extracted_syms = []
            for sf_idx in range(min(4, cfg.tx.num_subframes)):
                pbch_indices = pbch_symbol_indices_for_subframes(1, cfg.tx.normal_cp, start_subframe=sf_idx)
                p_cfg = PBCHConfig(cfg.tx.ndlrb, cfg.tx.normal_cp, pbch_indices, 240)
                extractor = PBCHExtractor(p_cfg)
                try:
                    syms = extractor.extract(grid_active)
                    extracted_syms.append(syms)
                except: pass
            
            if extracted_syms:
                results['rx_pbch_symbols'] = np.concatenate(extracted_syms)
        except: pass
            
    return results

# 5. STREAMLIT GUI IMPLEMENTACIJA

st.set_page_config(page_title="LTE Pro Analyzer", layout="wide", page_icon="📡")
st.title("📡 LTE System: TX -> Channel -> RX.")

# --- SIDEBAR (KONTROLE) ---
st.sidebar.header("1. Transmitter (TX)")
ndlrb = st.sidebar.selectbox("Bandwidth (NDLRB)", [6, 15, 25, 50], index=0)
normal_cp = st.sidebar.checkbox("Normal CP", value=True)
num_subframes = st.sidebar.number_input("Broj subfrejmova", 4, 10, 4)
n_id_2 = st.sidebar.selectbox("Cell ID Group (N_ID_2)", [0, 1, 2], index=1)
pbch_on = st.sidebar.checkbox("Pošalji MIB (PBCH)", value=True)

# Logika za MIB unos
mib_mode, mib_man = "random", "0"*24
if pbch_on and st.sidebar.radio("MIB Izvor", ["Random", "Manual"]) == "Manual":
    mib_mode = "manual"
    mib_man = st.sidebar.text_input("24 bita", "1010"*6)

st.sidebar.markdown("---")
st.sidebar.header("2. Kanal (Channel)")
snr_db = st.sidebar.slider("SNR (dB)", -5.0, 35.0, 20.0, 1.0)
st.sidebar.caption(" 5dB za test robusnosti!")
cfo_hz = st.sidebar.number_input("CFO (Hz)", -5000.0, 5000.0, 0.0, step=100.0)
st.sidebar.caption(" 300Hz ili više!")
seed = st.sidebar.number_input("RNG Seed", 0, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.header("3. Receiver (RX)")
rx_corr = st.sidebar.checkbox("CFO Korekcija", value=True)

run_btn = st.sidebar.button("POKRENI SIMULACIJU", type="primary")

# Kreiranje konfiguracijskog objekta
cfg = RunConfig(
    tx=TxConfig(ndlrb, normal_cp, num_subframes, n_id_2, pbch_on, mib_mode, seed, mib_man),
    ch=ChannelConfig(cfo_hz, snr_db, seed, 0.0),
    rx=RxConfig(rx_corr)
)

# --- POKRETANJE SIMULACIJE ---
if run_btn:
    with st.spinner("Simulacija u toku..."):
        try:
            res = run_simulation(cfg)
            st.session_state["res"] = res
            st.session_state["cfg"] = cfg
        except Exception as e:
            st.error(f"Greška tokom simulacije: {e}")

# --- PRIKAZ REZULTATA ---
if "res" in st.session_state:
    res = st.session_state["res"]
    c = st.session_state["cfg"]
    rx_out = res.get('rx_result')

    # Metrics Row
    st.subheader("📊 Rezultati Detekcije")
    col1, col2, col3, col4 = st.columns(4)
    
    # PSS ID Provjera
    pss_ok = (rx_out.n_id_2_hat == c.tx.n_id_2)
    col1.metric("PSS ID (Cell ID)", f"{rx_out.n_id_2_hat}", delta="MATCH" if pss_ok else "MISMATCH")
    
    col2.metric("Timing Offset", f"{rx_out.tau_hat}")
    
    est_cfo = rx_out.cfo_hat if rx_out.cfo_hat else 0.0
    col3.metric("Est. CFO", f"{est_cfo:.1f} Hz", f"Err: {abs(est_cfo - c.ch.freq_offset_hz):.1f} Hz")
    
    col4.metric("MIB CRC", "PASS" if rx_out.crc_ok else "FAIL", 
                delta_color="normal" if rx_out.crc_ok else "inverse")

    # Tabovi za vizualizaciju
    tabs = st.tabs(["Overview", "Grid Inspector (Maps)", "OFDM Bins", "RX: PSS Sync", "RX: EVM & Constellation", "RX: Bits", "Waveform"])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.info(f"Parametri: BW={c.tx.ndlrb} RBs | SNR={c.ch.snr_db} dB | N_ID_2={c.tx.n_id_2}")
        st.write("Dobrodošli u LTE Simulator. Koristite tabove iznad za detaljnu analizu.")
        st.markdown("""
        - **Grid Inspector:** Vizualizacija rasporeda kanala (CRS, PSS, PBCH).
        - **OFDM Bins:** Spektralni prikaz pojedinačnih simbola.
        - **RX EVM:** Kvalitet signala (Error Vector Magnitude).
        """)

    # --- TAB 2: GRID INSPECTOR (NOVO!) ---
    with tabs[1]:
        st.subheader("Resource Grid Inspector")
        grid = res["grid"]
        
        # Radio button za izbor pogleda
        view_mode = st.radio("Tip prikaza", ["Energy (Magnitude)", "Resource Map (Structure)"], horizontal=True)
        
        if view_mode == "Energy (Magnitude)":
            n_sym = grid.shape[1]
            default_sym = int(res.get('pss_index', 0))
            sym_sel = st.slider("Odaberi OFDM Simbol", 0, n_sym - 1, default_sym)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(20*np.log10(np.abs(grid)+1e-12), aspect="auto", origin="lower", cmap='viridis')
            ax.axvline(sym_sel, color='red', linestyle='--', linewidth=2)
            ax.set_title("Resource Grid Magnitude (dB)")
            fig.colorbar(im, ax=ax, label="dB")
            st.pyplot(fig)
            
        else:
            # --- STRUCTURE VIEW ---
            st.caption("Boje označavaju tip kanala (Logička struktura).")
            rmap = generate_resource_map(grid.shape, c.tx.ndlrb, c.tx.n_id_2, c.tx.num_subframes, c.tx.normal_cp)
            
            # Custom mapa boja
            cmap_colors = ['white', '#d62728', '#ff7f0e', '#1f77b4'] # Bijela, Crvena, Žuta, Plava
            cmap = ListedColormap(cmap_colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = BoundaryNorm(bounds, cmap.N)
            
            fig_map, ax_map = plt.subplots(figsize=(12, 5))
            ax_map.imshow(rmap, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation='nearest')
            ax_map.set_title(f"LTE Resource Map (N_ID_2={c.tx.n_id_2})")
            ax_map.set_xlabel("OFDM Symbols")
            ax_map.set_ylabel("Subcarriers")
            
            # Legenda
            patches = [
                mpatches.Patch(color='white', label='Data/Empty', edgecolor='gray'),
                mpatches.Patch(color='#d62728', label='CRS (Pilots)'),
                mpatches.Patch(color='#ff7f0e', label='PSS/SSS (Sync)'),
                mpatches.Patch(color='#1f77b4', label='PBCH (Info)')
            ]
            ax_map.legend(handles=patches, loc='upper right')
            st.pyplot(fig_map)
            st.info("Pokušaj promijeniti 'N_ID_2' u Sidebaru i vidi kako se Crvene tačkice (CRS) pomjeraju!")

    # --- TAB 3: OFDM BINS ---
    with tabs[2]:
        st.subheader("OFDM / IFFT Input")
        grid = res["grid"]
        sym_sel_bin = st.slider("Simbol", 0, grid.shape[1]-1, 0, key="bin_slider")
        bins = build_ifft_input_bins(grid[:, sym_sel_bin], res["Nfft"])
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        # Prikaz magnitude cijelog spektra
        ax[0].plot(np.abs(bins), '.-', linewidth=0.5)
        ax[0].set_title("Magnitude (Whole Band)")
        
        # Prikaz oko DC-a (centralne frekvencije)
        dc = res["Nfft"] // 2
        ax[1].plot(np.arange(dc-15, dc+15), np.abs(bins[dc-15:dc+15]), 'o-', color='red')
        ax[1].set_title("Zoom DC (Subcarrier 0 mora biti Null)")
        plt.tight_layout()
        st.pyplot(fig)

    # --- TAB 4: RX PSS ---
    with tabs[3]:
        st.subheader("PSS Cross-Correlation")
        pss_corr = res['pss_corr']
        fig_pss, ax_pss = plt.subplots(figsize=(10, 4))
        for i in range(3):
            ax_pss.plot(np.abs(pss_corr[i, :]), label=f"Hypothesis N_ID_2={i}", alpha=0.7)
        ax_pss.axvline(rx_out.tau_hat, color='k', linestyle='--', label='Detected Peak')
        ax_pss.legend()
        ax_pss.set_xlabel("Sample Index")
        ax_pss.set_ylabel("Correlation Magnitude")
        # Zoom oko detektovanog pika
        ax_pss.set_xlim(max(0, rx_out.tau_hat-200), rx_out.tau_hat+200)
        st.pyplot(fig_pss)

    # --- TAB 5: RX EVM & CONSTELACIJA ---
    with tabs[4]:
        st.subheader("Kvalitet Signala (EVM & Constellation)")
        
        syms = res.get('rx_pbch_symbols')
        if syms is not None and len(syms) > 0:
            # 1. Blind Phase Correction (za prikaz konstelacije)
            # Normalizacija snage
            p_avg = np.mean(np.abs(syms)**2)
            syms = syms / np.sqrt(p_avg)
            
            corrected_syms = []
            phase_acc = 0.0
            alpha = 0.05 # Faktor učenja za faznu petlju
            
            # QPSK Idealne tačke
            ideals = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            error_vectors = []
            
            for s in syms:
                # Derotacija
                s_rot = s * np.exp(-1j * phase_acc)
                # Hard decision (najbliža idealna tačka)
                dec = (np.sign(s_rot.real) + 1j * np.sign(s_rot.imag))/np.sqrt(2)
                
                # Procjena greške faze
                err = np.angle(s_rot * np.conj(dec))
                phase_acc += alpha * err
                corrected_syms.append(s_rot)
                
                # EVM vektor
                error_vectors.append(np.abs(s_rot - dec)**2)

            # 2. EVM Izračun
            mse = np.mean(error_vectors)
            evm_rms = np.sqrt(mse) * 100 # U postocima
            
            st.metric("EVM (Error Vector Magnitude)", f"{evm_rms:.2f} %", 
                      delta="Loše" if evm_rms > 17.5 else "Dobro", 
                      delta_color="inverse" if evm_rms > 17.5 else "normal")
            
            st.caption("Manji EVM je bolji. Za QPSK, EVM < 17.5% je standard.")

            # 3. Plot Konstelacije
            syms_plot = np.array(corrected_syms)
            fig_const, ax_const = plt.subplots(figsize=(6, 6))
            ax_const.scatter(syms_plot.real, syms_plot.imag, s=20, alpha=0.5, label=f"RX Symbols")
            ax_const.scatter(ideals.real, ideals.imag, c='red', marker='x', s=100, linewidth=2, label="Ideal QPSK")
            ax_const.grid(True, linestyle=':')
            ax_const.set_xlim(-2, 2); ax_const.set_ylim(-2, 2)
            ax_const.axhline(0, color='gray'); ax_const.axvline(0, color='gray')
            ax_const.legend()
            ax_const.set_title("PBCH Constellation (QPSK)")
            st.pyplot(fig_const)
        else:
            st.warning("Nema PBCH simbola za prikaz (vjerovatno neuspješna sinhronizacija).")

    # --- TAB 6 & 7 (BITS & WAVEFORM) ---
    with tabs[5]:
        if c.tx.pbch_enabled and rx_out.mib_bits is not None:
             st.write("Dekodirani bitovi (RX):")
             st.code(f"{rx_out.mib_bits}")
             st.write("Poslani bitovi (TX):")
             st.code(f"{res['mib_tx']}")
        else: st.info("Nema podataka o bitovima.")
        
    with tabs[6]:
        fig_w, ax_w = plt.subplots()
        ax_w.plot(np.real(res['tx_waveform'][:200]), label='TX (Clean)')
        ax_w.plot(np.real(res['rx_waveform'][:200]), label='RX (Noisy)', alpha=0.5)
        ax_w.set_title("Time Domain Waveform (Prvih 200 uzoraka)")
        ax_w.legend()
        st.pyplot(fig_w)