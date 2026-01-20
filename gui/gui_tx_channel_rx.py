# gui_tx_channel.py
"""
LTE TX- Channel-RX 
===============================================================================

Ova skripta pokreće Streamlit aplikaciju koja simulira kompletan LTE komunikacijski lanac:
Transmitter (TX) -> Channel (Kanal) -> Receiver (RX).

Funkcionalnosti:
    - Napredna vizualizacija: Waveform, Spectrum, Grid Inspector, OFDM Bins, Constellation.
    - RX Analiza: Digitalni prikaz bitova sa detekcijom grešaka.
    - Sweeps: Masovna simulacija (SNR/CFO) sa tabelom rezultata i Heatmap.
    - Kontrole: Channel Bypass, RX Disable, Descrambling, Normalization.

Pokretanje:
    $ streamlit run gui_tx_channel.py
"""

from __future__ import annotations

import sys
import time
import io  
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from scipy import signal 

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

#  Funkcija za konverziju slike u bytes (za Download) 
def fig_to_png_bytes(fig):
    """Konvertuje Matplotlib figuru u bytes za download dugme."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def generate_resource_map(grid_shape: Tuple[int, int], 
                          ndlrb: int, 
                          n_id_2: int, 
                          num_subframes: int, 
                          normal_cp: bool) -> np.ndarray:
    """
    Generiše matricu (mapu) resursa koja označava tip logičkog kanala za svaki element grida.

    Koristi se za vizualizaciju strukture okvira (Grid Inspector).

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
        Ako je True, koristi Normal Cyclic Prefix. Ako je False, koristi Extended CP.

    Returns
    -------
    np.ndarray
        2D matrica cijelih brojeva (int) gdje vrijednosti označavaju:
        - 0: Podaci/Prazno
        - 1: CRS (Piloti)
        - 2: PSS/SSS (Sinhronizacija)
        - 3: PBCH (Broadcast)
    """
    n_sc, n_sym = grid_shape
    resource_map = np.zeros(grid_shape, dtype=int)
    
    n_sym_per_slot = 7 if normal_cp else 6
    n_sym_per_sf = n_sym_per_slot * 2
    center_sc = n_sc // 2
    
    # --- 1. CRS (Cell Reference Signals - PILOTI) [ID: 1] ---
    v_shift = n_id_2 % 6
    
    for sf in range(num_subframes):
        sf_start = sf * n_sym_per_sf
        for slot in range(2):
            slot_start = sf_start + slot * n_sym_per_slot
            crs_syms = [0, 4] if normal_cp else [0, 3]
            
            for local_sym in crs_syms:
                abs_sym = slot_start + local_sym
                if abs_sym >= n_sym: continue
                k_shift = v_shift if local_sym == 0 else (v_shift + 3) % 6
                resource_map[k_shift::6, abs_sym] = 1 

    # --- 2. PSS / SSS (Sync Signals) [ID: 2] ---
    pss_width = 62
    sc_start = center_sc - (pss_width // 2)
    sc_end = center_sc + (pss_width // 2)
    
    for sf in range(num_subframes):
        if sf in [0, 5]: 
            sf_start = sf * n_sym_per_sf
            pss_sym = sf_start + (n_sym_per_slot - 1)
            sss_sym = pss_sym - 1
            
            if pss_sym < n_sym:
                resource_map[sc_start:sc_end, pss_sym] = 2 
            if sss_sym < n_sym:
                resource_map[sc_start:sc_end, sss_sym] = 2

    # --- 3. PBCH (Broadcast Channel) [ID: 3] ---
    pbch_width = 72
    pbch_sc_start = center_sc - (pbch_width // 2)
    pbch_sc_end = center_sc + (pbch_width // 2)
    
    if num_subframes > 0:
        slot1_start = n_sym_per_slot 
        pbch_syms = [slot1_start + i for i in range(4)]
        
        for sym in pbch_syms:
            if sym < n_sym:
                current_col = resource_map[pbch_sc_start:pbch_sc_end, sym]
                mask = (current_col != 1)
                current_col[mask] = 3
                resource_map[pbch_sc_start:pbch_sc_end, sym] = current_col

    return resource_map

def safe_parse_mib_bits(bitstr: str) -> Optional[np.ndarray]:
    """
    Sigurno parsira ulazni string bitova u NumPy niz.

    Parameters
    ----------
    bitstr : str
        Ulazni string koji sadrži nule i jedinice (npr. "101010...").

    Returns
    -------
    Optional[np.ndarray]
        Niz integera (0 ili 1) dužine 24 ako je parsiranje uspješno.
        Vraća None ako je string neispravan ili pogrešne dužine.
    """
    s = bitstr.strip().replace(" ", "")
    if len(s) != 24: return None
    if any(ch not in "01" for ch in s): return None
    return np.array([int(ch) for ch in s], dtype=int)

def build_ifft_input_bins(grid_sym: np.ndarray, N: int) -> np.ndarray:
    """
    Mapira aktivne podnosioce iz baseband grida na ulaze IFFT-a.
    
    Vrši pomjeranje frekvencija tako da DC komponenta bude u centru (indeks N/2)
    i umeće nulu na DC nosiocu.

    Parameters
    ----------
    grid_sym : np.ndarray
        Vektor kompleksnih simbola jednog OFDM simbola (aktivni podnosioci).
    N : int
        Veličina IFFT-a (FFT size), npr. 128, 256, 512.

    Returns
    -------
    np.ndarray
        Vektor dužine N spreman za IFFT operaciju, sa umetnutom nulom na DC-u
        i odgovarajućim mapiranjem pozitivnih/negativnih frekvencija.
    """
    num_subcarriers = grid_sym.size
    half = num_subcarriers // 2
    dc = N // 2
    
    ifft_in = np.zeros(N, dtype=np.complex128)
    ifft_in[dc] = 0.0 
    
    pos_freq_bins = np.arange(dc + 1, dc + 1 + half)
    pos_sub = np.arange(half, num_subcarriers)
    
    neg_freq_bins = np.arange(dc - half, dc)
    neg_sub = np.arange(0, half)
    
    ifft_in[pos_freq_bins] = grid_sym[pos_sub]
    ifft_in[neg_freq_bins] = grid_sym[neg_sub]
    
    return ifft_in

def draw_bit_comparison(tx_bits: np.ndarray, rx_bits: np.ndarray, title: str = "Bit Error Visualizer"):
    """
    Crta vizualno poređenje TX i RX bitova koristeći "step" plot (digitalni talasni oblik).
    
    Crta plavu liniju za TX bitove i zelenu isprekidanu za RX bitove.
    Sva nepoklapanja (greške) se automatski označavaju crvenim 'X' markerima.

    Parameters
    ----------
    tx_bits : np.ndarray
        Niz poslatih bitova (Transmitted bits).
    rx_bits : np.ndarray
        Niz primljenih/dekodiranih bitova (Received bits).
    title : str, optional
        Naslov grafika. Default je "Bit Error Visualizer".

    Returns
    -------
    None
        Funkcija direktno iscrtava grafik koristeći `st.pyplot()`.
    """
    # 1. Pretvaranje u numpy nizove (Handling Strings/Lists/Arrays)
    if isinstance(tx_bits, str): tx_bits = [int(x) for x in tx_bits]
    if isinstance(rx_bits, str): rx_bits = [int(x) for x in rx_bits]
    
    tx_bits = np.array(tx_bits).flatten().astype(int)
    rx_bits = np.array(rx_bits).flatten().astype(int)
    
    n_bits = len(tx_bits)
    if len(rx_bits) != n_bits:
        st.warning(f"Dužina TX ({n_bits}) i RX ({len(rx_bits)}) bitova se ne poklapa. Prikazujem presjek.")
        min_len = min(n_bits, len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
        n_bits = min_len

    # Indeksi
    indices = np.arange(n_bits)
    
    # Pronadji greske
    errors = (tx_bits != rx_bits)
    num_errors = np.sum(errors)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 1. Plot TX Bits (Original) - Plava linija
    ax.step(indices, tx_bits, where='mid', label='TX Bits (Sent)', color='#1f77b4', linewidth=3, alpha=0.6)
    
    # 2. Plot RX Bits (Primljeni) - Zelena isprekidana linija
    ax.step(indices, rx_bits, where='mid', label='RX Bits (Received)', color='green', linestyle='--', linewidth=2, alpha=0.9)
    
    # 3. Označavanje Grešaka (Crveni X)
    if num_errors > 0:
        error_indices = indices[errors]
        ax.scatter(error_indices, rx_bits[error_indices], color='red', marker='x', s=150, linewidth=3, zorder=10, label=f'Bit Error ({num_errors})')
        # Vertikalna linija za naglašavanje greške
        for ei in error_indices:
            ax.vlines(ei, rx_bits[ei], tx_bits[ei], colors='red', linestyles=':', alpha=0.5)

    # Uljepsavanje grafa
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0 (Low)', '1 (High)'])
    ax.set_xticks(indices)
    ax.set_xlabel("Bit Index (Time)")
    ax.set_ylabel("Bit Value")
    ax.set_title(f"{title} - Detected Errors: {num_errors}")
    ax.legend(loc='upper right')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Ispis vrijednosti TX bitova na dnu radi lakseg citanja
    if n_bits <= 32:
        for i, val in enumerate(tx_bits):
            # Boja teksta zavisi od toga da li je greska
            c_text = 'red' if errors[i] else 'gray'
            fw = 'bold' if errors[i] else 'normal'
            ax.text(i, -0.25, str(val), ha='center', color=c_text, fontweight=fw, fontsize=9)

    st.pyplot(fig)

# 3. KONFIGURACIJA (DATA CLASSES)
@dataclass
class TxConfig:
    """
    Konfiguracija parametara predajnika (Transmitter).

    Attributes
    ----------
    ndlrb : int
        Broj resursnih blokova (bandwidth), npr. 6, 15, 25, 50.
    normal_cp : bool
        Tip cikličnog prefiksa (True=Normal, False=Extended).
    num_subframes : int
        Trajanje simulacije u subfrejmovima.
    n_id_2 : int
        ID grupe ćelije (0-2) za CRS pomak.
    pbch_enabled : bool
        Da li se generiše PBCH kanal.
    mib_mode : str
        Način generisanja MIB bitova ("random" ili "manual").
    mib_seed : int
        Seed za generator slučajnih brojeva (ako je mib_mode="random").
    mib_manual_bits : str
        String bitova ako je odabran "manual" mod.
    """
    ndlrb: int              
    normal_cp: bool         
    num_subframes: int      
    n_id_2: int             
    pbch_enabled: bool      
    mib_mode: str           
    mib_seed: int           
    mib_manual_bits: str    

@dataclass
class ChannelConfig:
    """
    Konfiguracija kanala.

    Attributes
    ----------
    enabled : bool
        Da li je kanal aktivan (On/Off). Ako je False, koristi se bypass (idealan prenos).
    freq_offset_hz : float
        Frekvencijski ofset (CFO) u Hz.
    snr_db : float
        Odnos signal-šum (SNR) u decibelima.
    seed : int
        Seed za generisanje šuma u kanalu.
    initial_phase_rad : float
        Početna faza (za testiranje faznog pomaka).
    """
    enabled: bool           
    freq_offset_hz: float   
    snr_db: float           
    seed: int               
    initial_phase_rad: float

@dataclass
class RxConfig:
    """
    Konfiguracija prijemnika.

    Attributes
    ----------
    enabled : bool
        Da li je prijemnik aktivan.
    enable_cfo_correction : bool
        Da li primijeniti algoritam za korekciju CFO-a.
    descrambling : bool
        Uključuje/isključuje descrambling korak.
    normalize_pss : bool
        Da li normalizovati signal prije PSS korelacije.
    """
    enabled: bool               
    enable_cfo_correction: bool 
    descrambling: bool          
    normalize_pss: bool         

@dataclass
class RunConfig:
    """
    Objedinjena konfiguracija za cijelu simulaciju.

    Attributes
    ----------
    tx : TxConfig
        Postavke predajnika.
    ch : ChannelConfig
        Postavke kanala.
    rx : RxConfig
        Postavke prijemnika.
    """
    tx: TxConfig
    ch: ChannelConfig
    rx: RxConfig

# 4. GLAVNA LOGIKA SIMULACIJE
@st.cache_data(show_spinner=False)
def run_simulation(cfg: RunConfig) -> Dict[str, Any]:
    """
    Pokreće kompletan LTE komunikacijski lanac: TX -> Channel -> RX.

    Ova funkcija orkestrira generisanje signala, prolazak kroz kanal i dekodiranje.
    Podržava 'Channel Bypass' i 'RX Disable' modove.
    Rezultati se keširaju pomoću @st.cache_data radi performansi.

    Parameters
    ----------
    cfg : RunConfig
        Objedinjen konfiguracijski objekat koji sadrži TX, Channel i RX postavke.

    Returns
    -------
    Dict[str, Any]
        Rječnik sa rezultatima simulacije, uključujući:
        - 'tx_waveform': Generisani signal (vremenski domen).
        - 'rx_waveform': Primljeni signal nakon kanala.
        - 'grid': Originalni TX resursni grid.
        - 'rx_result': Objekat sa rezultatima dekodiranja (LTERxResult).
        - 'pss_corr': Vektor korelacije za PSS sinhronizaciju.
        - 'fs': Frekvencija uzorkovanja.
        - 'mib_tx': Originalni poslani bitovi.
        - 'rx_pbch_symbols': Ekstraktovani simboli za konstelaciju.
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
    
    results['fs'] = fs
    results['tx_waveform'] = tx_waveform
    results['grid'] = tx_chain.grid
    results['mib_tx'] = mib_bits
    
    temp_ofdm = OFDMModulator(tx_chain.grid)
    results['Nfft'] = temp_ofdm.N
    results['pss_index'] = tx_chain._pss_symbol_index()

    # --- 2. CHANNEL (KANAL) ---
    if cfg.ch.enabled:
        channel = LTEChannel(
            freq_offset_hz=float(cfg.ch.freq_offset_hz), 
            sample_rate_hz=fs,
            snr_db=float(cfg.ch.snr_db), 
            seed=int(cfg.ch.seed),
            initial_phase_rad=float(cfg.ch.initial_phase_rad),
        )
        rx_waveform = channel.apply(tx_waveform)
    else:
        # Bypass Mode (Idealan kanal)
        rx_waveform = tx_waveform.copy()
        
    results['rx_waveform'] = rx_waveform

    # --- 3. RECEIVER (RX) PROCESSING ---
    if not cfg.rx.enabled:
        results['rx_result'] = None
        results['pss_corr'] = None
        results['rx_pbch_symbols'] = None
        return results

    rx_chain = LTERxChain(
        sample_rate_hz=fs, 
        ndlrb=cfg.tx.ndlrb, 
        normal_cp=cfg.tx.normal_cp,
        pci=0,
        enable_cfo_correction=cfg.rx.enable_cfo_correction,
        pbch_spread_subframes=cfg.tx.num_subframes
    )
    
    # Primjena descrambling/normalization parametara (ako backend podržava)
    if hasattr(rx_chain, 'descrambling_enabled'):
        rx_chain.descrambling_enabled = cfg.rx.descrambling
    if hasattr(rx_chain, 'normalize_pss'):
        rx_chain.normalize_pss = cfg.rx.normalize_pss
    
    t0 = time.time()
    rx_res = rx_chain.decode(rx_waveform)
    results['rx_result'] = rx_res
    results['process_time'] = time.time() - t0

    # --- 4. ANALITIKA (PSS, EVM) ---
    pss_sync = PSSSynchronizer(fs, ndlrb=cfg.tx.ndlrb, normal_cp=cfg.tx.normal_cp)
    
    # Normalizacija prije PSS-a
    if cfg.rx.normalize_pss:
        rx_for_pss = rx_waveform / (np.sqrt(np.mean(np.abs(rx_waveform)**2)) + 1e-12)
    else:
        rx_for_pss = rx_waveform

    pss_corr = pss_sync.correlate(rx_for_pss)
    results['pss_corr'] = pss_corr

    results['rx_pbch_symbols'] = None
    if rx_res.tau_hat is not None:
        cfo = rx_res.cfo_hat if (cfg.rx.enable_cfo_correction and rx_res.cfo_hat) else 0.0
        fo_corr = FrequencyOffset(-cfo, fs)
        rx_cfo_corr = fo_corr.apply(rx_waveform)
        
        start_idx = rx_res.tau_hat
        ofdm_demod = OFDMDemodulator(cfg.tx.ndlrb, cfg.tx.normal_cp)
        
        sym_idx_pss = 6 if cfg.tx.normal_cp else 5
        samps_before = 0
        for i in range(sym_idx_pss):
            cp = ofdm_demod.cp_lengths[i % ofdm_demod.n_symbols_per_slot]
            samps_before += (ofdm_demod.fft_size + cp)
            
        start_sf = start_idx - samps_before
        if start_sf < 0: start_sf = 0
        rx_aligned = rx_cfo_corr[start_sf:]
        
        try:
            grid_full = ofdm_demod.demodulate(rx_aligned)
            grid_active = ofdm_demod.extract_active_subcarriers(grid_full)
            
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
# Channel ON/OFF
channel_enabled = st.sidebar.checkbox("Channel Active (Uključi smetnje)", value=True, help="Ako je isključeno, signal ide direktno TX->RX (idealan prenos).")

# Promijenjen opseg SNR-a na -25.0 da se omogući razbijanje signala
snr_db = st.sidebar.slider("SNR (dB)", -25.0, 35.0, 20.0, 1.0, disabled=not channel_enabled)
cfo_hz = st.sidebar.number_input("CFO (Hz)", -5000.0, 5000.0, 0.0, step=100.0, disabled=not channel_enabled)
seed = st.sidebar.number_input("RNG Seed", 0, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.header("3. Receiver (RX)")
# RX ON/OFF i dodatne opcije
rx_enabled = st.sidebar.checkbox("Receiver Active", value=True)
rx_corr = st.sidebar.checkbox("CFO Korekcija", value=True, disabled=not rx_enabled)
rx_descramble = st.sidebar.checkbox("Descrambling", value=True, disabled=not rx_enabled, help="Uključi/isključi de-scrambling bitova.")
rx_norm_pss = st.sidebar.checkbox("Normalize before PSS", value=True, disabled=not rx_enabled, help="Normalizuj signal prije korelacije.")

run_btn = st.sidebar.button("POKRENI SIMULACIJU", type="primary")

# Kreiranje konfiguracijskog objekta
cfg = RunConfig(
    tx=TxConfig(ndlrb, normal_cp, num_subframes, n_id_2, pbch_on, mib_mode, seed, mib_man),
    ch=ChannelConfig(channel_enabled, cfo_hz, snr_db, seed, 0.0),
    rx=RxConfig(rx_enabled, rx_corr, rx_descramble, rx_norm_pss) # Koristi rx_descramble
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
    
    if not c.ch.enabled:
        st.info("ℹ️ **Kanal je isključen (Bypass Mode).** Rezultati predstavljaju idealan prenos bez šuma.")

    if rx_out:
        col1, col2, col3, col4 = st.columns(4)
        
        # PSS ID Provjera
        pss_ok = (rx_out.n_id_2_hat == c.tx.n_id_2)
        col1.metric("PSS ID (Cell ID)", f"{rx_out.n_id_2_hat}", delta="MATCH" if pss_ok else "MISMATCH")
        col2.metric("Timing Offset", f"{rx_out.tau_hat}")
        
        est_cfo = rx_out.cfo_hat if rx_out.cfo_hat else 0.0
        target_cfo = c.ch.freq_offset_hz if c.ch.enabled else 0.0
        col3.metric("Est. CFO", f"{est_cfo:.1f} Hz", f"Err: {abs(est_cfo - target_cfo):.1f} Hz")
        
        # --- CRC Logic sa stvarnom provjerom bitova ---
        has_bit_errors = False
        if rx_out.mib_bits is not None and res.get('mib_tx') is not None:
            tx_b_chk = np.array(res['mib_tx']).flatten().astype(int)
            rx_b_chk = np.array(rx_out.mib_bits).flatten().astype(int)
            if len(tx_b_chk) == len(rx_b_chk):
                has_bit_errors = np.any(tx_b_chk != rx_b_chk)
        
        crc_text = "PASS" if rx_out.crc_ok else "FAIL"
        crc_subtext = "OK"
        crc_color = "normal"
        
        if not rx_out.crc_ok:
            crc_subtext = "-Error"
            crc_color = "inverse"
        elif has_bit_errors:
            crc_text = "PASS (?)"
            crc_subtext = "⚠️ Bit Errors"
            crc_color = "inverse"
            
        col4.metric("MIB CRC", crc_text, crc_subtext, delta_color=crc_color)
    else:
        st.warning("Receiver is OFF. Only TX signals available.")

    # --- DEFINICIJA TABOVA ---
    tabs = st.tabs([
        "Overview", 
        "Grid Inspector", 
        "OFDM Bins", 
        "RX: PSS Sync", 
        "RX: EVM & Constellation", 
        "RX: Bits", 
        "Waveform", 
        "Spectrum",    
        "Sweeps"        
    ])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        snr_status = f"{c.ch.snr_db} dB" if c.ch.enabled else "**Bypass (No Noise)**"
        st.info(f"Parametri: BW={c.tx.ndlrb} RBs | SNR={snr_status} | N_ID_2={c.tx.n_id_2}")
        st.write("Dobrodošli u LTE Simulator. Koristite tabove iznad za detaljnu analizu.")

    # --- TAB 2: GRID INSPECTOR (Trenutno ažuriranje) ---
    with tabs[1]:
        st.subheader("Resource Grid Inspector")
        # Koristimo trenutne (selektovane) vrijednosti iz sidebara za Structure View da se azurira odmah
        
        view_mode = st.radio("Tip prikaza", ["Energy (Magnitude)", "Resource Map (Structure)"], horizontal=True)
        
        if view_mode == "Energy (Magnitude)":
            grid = res["grid"]
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
            # Structure View: Generisi na osnovu TRENUTNIH (sidebar) postavki (dinamicki)
            st.caption("Boje označavaju tip kanala. Prikaz se ažurira trenutno.")
            
            curr_ndlrb = cfg.tx.ndlrb
            curr_nid2 = cfg.tx.n_id_2
            curr_nsf = cfg.tx.num_subframes
            curr_cp = cfg.tx.normal_cp
            
            # Pretpostavka za dimenzije
            curr_n_sc = curr_ndlrb * 12
            curr_n_sym = curr_nsf * 14 # Normal CP
            rmap = generate_resource_map((curr_n_sc, curr_n_sym), curr_ndlrb, curr_nid2, curr_nsf, curr_cp)
            
            cmap_colors = ['white', '#d62728', '#ff7f0e', '#1f77b4'] 
            cmap = ListedColormap(cmap_colors)
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = BoundaryNorm(bounds, cmap.N)
            
            fig_map, ax_map = plt.subplots(figsize=(12, 5))
            ax_map.imshow(rmap, aspect="auto", origin="lower", cmap=cmap, norm=norm, interpolation='nearest')
            ax_map.set_title(f"LTE Resource Map (N_ID_2={curr_nid2})")
            ax_map.set_xlabel("OFDM Symbols")
            ax_map.set_ylabel("Subcarriers")
            
            # Legenda
            patches = [
                mpatches.Patch(color='white', label='Data/Empty', edgecolor='gray'),
                mpatches.Patch(color='#d62728', label='CRS (Pilots)'),
                mpatches.Patch(color='#ff7f0e', label='PSS/SSS (Sync)'),
                mpatches.Patch(color='#1f77b4', label='PBCH (Info)')
            ]
            ax_map.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            st.pyplot(fig_map)

            st.markdown("""
            **Objašnjenje boja:**
            - 🔴 **CRS (Crveno):** Piloti.
            - 🟠 **PSS/SSS (Narandžasto):** Sinhronizacija.
            - 🔵 **PBCH (Plavo):** Broadcast kanal.
            """)

    # --- TAB 3: OFDM BINS ---
    with tabs[2]:
        st.subheader("OFDM / IFFT Input")
        st.caption("Prikaz magnitude ulaznih binova u IFFT za odabrani simbol (dB).")
        
        grid = res["grid"]
        sym_sel_bin = st.slider("Simbol", 0, grid.shape[1]-1, 0, key="bin_slider")
        bins = build_ifft_input_bins(grid[:, sym_sel_bin], res["Nfft"])
        
        mag_db = 20 * np.log10(np.abs(bins) + 1e-12)
        
        # Mapiranje tipova (za boje)
        rmap = generate_resource_map(grid.shape, c.tx.ndlrb, c.tx.n_id_2, c.tx.num_subframes, c.tx.normal_cp)
        current_types = rmap[:, sym_sel_bin] 
        
        N = res["Nfft"]
        type_bins = np.full(N, -1, dtype=int) 
        
        num_sc = grid.shape[0]
        half = num_sc // 2
        dc = N // 2
        
        pos_freq = np.arange(dc + 1, dc + 1 + half)
        pos_sub = np.arange(half, num_sc)
        neg_freq = np.arange(dc - half, dc)
        neg_sub = np.arange(0, half)
        
        type_bins[pos_freq] = current_types[pos_sub]
        type_bins[neg_freq] = current_types[neg_sub]
        type_bins[dc] = -2 
        
        freq_idx = np.arange(N) - dc
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        styles = {
            -1: ("Guard", "white", 0.0),
            -2: ("DC", "black", 0.0),    
            0:  ("Data", "gray", 0.5),
            1:  ("CRS (Pilot)", "#d62728", 1.0), 
            2:  ("PSS/SSS", "#ff7f0e", 0.8),     
            3:  ("PBCH", "#1f77b4", 0.8)         
        }
        
        for t_code, (label, color, alpha) in styles.items():
            if t_code < 0: continue 
            
            mask = (type_bins == t_code)
            if np.any(mask):
                markerline, stemlines, baseline = ax.stem(
                    freq_idx[mask], mag_db[mask], 
                    markerfmt='o', basefmt=" ", linefmt='-'
                )
                plt.setp(markerline, markersize=3, color=color, label=label)
                plt.setp(stemlines, linewidth=0.8, color=color, alpha=0.7)

        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1, label='DC (Null)')
        ax.set_title(f"Subcarrier Magnitude Spectrum (Symbol {sym_sel_bin})")
        ax.set_xlabel("Subcarrier Index (Relative to Center Freq)")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_ylim(bottom=-60)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # --- TAB 4: RX PSS ---
    with tabs[3]:
        st.subheader("PSS Cross-Correlation")
        if rx_out:
            pss_corr = res['pss_corr']
            fig_pss, ax_pss = plt.subplots(figsize=(10, 4))
            for i in range(3):
                ax_pss.plot(np.abs(pss_corr[i, :]), label=f"Hypothesis N_ID_2={i}", alpha=0.7)
            ax_pss.axvline(rx_out.tau_hat, color='k', linestyle='--', label='Detected Peak')
            ax_pss.legend()
            ax_pss.set_xlabel("Sample Index")
            ax_pss.set_ylabel("Correlation Magnitude")
            ax_pss.set_xlim(max(0, rx_out.tau_hat-200), rx_out.tau_hat+200)
            st.pyplot(fig_pss)
        else:
            st.warning("Receiver Disabled.")

    # --- TAB 5: RX EVM & CONSTELACIJA ---
    with tabs[4]:
        st.subheader("Kvalitet Signala (EVM & Constellation)")
        if rx_out:
            syms = res.get('rx_pbch_symbols')
            
            if syms is not None and len(syms) > 0:
                p_avg = np.mean(np.abs(syms)**2)
                syms = syms / np.sqrt(p_avg)
                
                corrected_syms = []
                phase_acc = 0.0
                alpha = 0.05 
                ideals = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
                error_vectors = []
                
                for s in syms:
                    s_rot = s * np.exp(-1j * phase_acc)
                    dec = (np.sign(s_rot.real) + 1j * np.sign(s_rot.imag))/np.sqrt(2)
                    err = np.angle(s_rot * np.conj(dec))
                    phase_acc += alpha * err
                    corrected_syms.append(s_rot)
                    error_vectors.append(np.abs(s_rot - dec)**2)

                mse = np.mean(error_vectors)
                evm_rms = np.sqrt(mse) * 100 
                
                st.metric("EVM (RMS)", f"{evm_rms:.2f} %", 
                          delta="Loše" if evm_rms > 17.5 else "Dobro", 
                          delta_color="inverse" if evm_rms > 17.5 else "normal")
                
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
                st.warning("Nema PBCH simbola za prikaz.")
        else:
            st.warning("Receiver Disabled.")

    # --- TAB 6: RX BITS ---
    with tabs[5]:
        if rx_out and c.tx.pbch_enabled:
            # Uvijek imamo TX bitove
            tx_b = np.array(res['mib_tx']).flatten().astype(int)
            
            # Pokušaj dohvatiti RX bitove
            if rx_out.mib_bits is not None:
                 rx_b = np.array(rx_out.mib_bits).flatten().astype(int)
                 sync_failed = False
            else:
                 # FALLBACK: Ako sync ne uspije, generiši nasumične bitove (šum)
                 # Ovo omogućava prikaz "Crvenih X-ova" čak i kad je signal uništen
                 rx_b = np.random.randint(0, 2, size=len(tx_b))
                 sync_failed = True

            st.subheader("Vizualizacija Grešaka u Bitovima")
            st.caption("Poređenje poslatih (TX) i primljenih (RX) bitova MIB poruke.")
            
            if len(tx_b) == len(rx_b):
                ber = np.mean(tx_b != rx_b)
                num_errs = np.sum(tx_b != rx_b)
            else:
                ber = 1.0
                num_errs = len(tx_b)
            
            if num_errs == 0 and not sync_failed:
                st.success(f"Perfect Reception! BER: {ber:.4f}")
            else:
                st.error(f"Errors Detected! BER: {ber:.4f} ({num_errs} grešaka od {len(tx_b)} bitova)")
            
            draw_bit_comparison(tx_b, rx_b, title="MIB Bit Comparison (TX vs RX)")
            
            with st.expander("Prikaži sirove podatke (Raw Bits)"):
                st.write("Poslani bitovi (TX):")
                st.code(f"{res['mib_tx']}")
                if not sync_failed:
                    st.write("Dekodirani bitovi (RX):")
                    st.code(f"{rx_out.mib_bits}")
                else:
                    st.write("RX Bitovi: N/A (Sync Failed - Noise displayed above)")
        else:
            st.info("RX isključen ili PBCH nije omogućen.")


    # Tab 7: Time waveform (TX i RX odvojeno)
    with tabs[6]:
        st.subheader("Waveform")

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

            
            plt.tight_layout()
            st.pyplot(fig)

        

            st.download_button(
                "Download (PNG)",
                data=fig_to_png_bytes(fig),
                file_name="time_tx_rx_separate_subplots.png",
                mime="image/png",
            )

    # Tab 8: Spectrum (subplots)
    with tabs[7]:
        st.subheader("Spectrum ")

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

            axs[0].plot(f, 20*np.log10(np.abs(S_tx)+1e-12), color='#1f77b4')
            axs[0].set_title("TX |FFT| (dB)")
            axs[0].set_ylabel("dB")
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(f, 20*np.log10(np.abs(S_rx)+1e-12), color='#ff7f0e')
            axs[1].set_title("RX |FFT| (dB)")
            axs[1].set_xlabel("frequency [Hz]")
            axs[1].set_ylabel("dB")
            axs[1].grid(True, alpha=0.3)

            
            plt.tight_layout()
            st.pyplot(fig)


            st.download_button(
                "Download spectrum (PNG)",
                data=fig_to_png_bytes(fig),
                file_name="spectrum_subplots.png",
                mime="image/png",
            )

    # --- TAB 9: SWEEPS ---
    with tabs[8]:
        st.subheader("Masovna Simulacija (Sweeps)")
        st.markdown("Ovdje možete pokrenuti simulaciju za niz SNR i CFO vrijednosti i dobiti tabelu rezultata.")
        
        c1, c2 = st.columns(2)
        with c1:
            sw_snr_start = st.number_input("SNR Start (dB)", -20.0, 30.0, -10.0)
            sw_snr_end = st.number_input("SNR End (dB)", -20.0, 30.0, 10.0)
            sw_snr_step = st.number_input("SNR Step", 1.0, 10.0, 2.0)
        with c2:
            sw_cfo_start = st.number_input("CFO Start (Hz)", -2000.0, 2000.0, 0.0)
            sw_cfo_end = st.number_input("CFO End (Hz)", -2000.0, 2000.0, 1000.0)
            sw_cfo_step = st.number_input("CFO Step", 100.0, 1000.0, 500.0)
            
        if st.button("Pokreni Sweep Analizu"):
            snr_vals = np.arange(sw_snr_start, sw_snr_end + 0.1, sw_snr_step)
            cfo_vals = np.arange(sw_cfo_start, sw_cfo_end + 0.1, sw_cfo_step)
            
            results_list = []
            
            progress_bar = st.progress(0)
            total_iter = len(snr_vals) * len(cfo_vals)
            curr_iter = 0
            
            orig_cfg = st.session_state["cfg"]
            
            for s in snr_vals:
                for f_off in cfo_vals:
                    # Kreiramo privremeni config sa omogućenim kanalom za sweep
                    temp_ch = ChannelConfig(True, float(f_off), float(s), 42, 0.0)
                    temp_cfg = RunConfig(orig_cfg.tx, temp_ch, orig_cfg.rx)
                    
                    try:
                        r = run_simulation(temp_cfg)
                        rx_r = r.get('rx_result')
                        
                        crc_status = "OK" if (rx_r and rx_r.crc_ok) else "FAIL"
                        
                        ber = 1.0
                        if rx_r and rx_r.mib_bits is not None:
                             tx_b_sw = res['mib_tx']
                             if isinstance(tx_b_sw, str): tx_b_sw = [int(x) for x in tx_b_sw]
                             tx_b_sw = np.array(tx_b_sw).flatten()
                             
                             rx_b_sw = rx_r.mib_bits
                             if isinstance(rx_b_sw, str): rx_b_sw = [int(x) for x in rx_b_sw]
                             rx_b_sw = np.array(rx_b_sw).flatten()
                             
                             if len(tx_b_sw) == len(rx_b_sw):
                                 ber = np.mean(tx_b_sw != rx_b_sw)
                        
                        pss_peak = 0.0
                        if r.get('pss_corr') is not None:
                            pss_peak = np.max(np.abs(r['pss_corr']))
                        
                        cfo_est_val = rx_r.cfo_hat if (rx_r and rx_r.cfo_hat is not None) else np.nan
                        
                        results_list.append({
                            "SNR (dB)": s,
                            "CFO (Hz)": f_off,
                            "CRC": crc_status,
                            "BER": ber,
                            "PSS Peak": pss_peak,
                            "CFO Est (Hz)": cfo_est_val
                        })
                    except Exception as e:
                        pass
                    
                    curr_iter += 1
                    progress_bar.progress(curr_iter / total_iter)
            
            df_res = pd.DataFrame(results_list)
            st.success("Analiza završena!")
            
            st.write("### 1. Tabela Rezultata")
            try:
                st.dataframe(df_res.style.format({
                    "SNR (dB)": "{:.1f}", 
                    "CFO (Hz)": "{:.1f}", 
                    "BER": "{:.4f}", 
                    "PSS Peak": "{:.2f}",
                    "CFO Est (Hz)": "{:.1f}" 
                }, na_rep="-"))
            except Exception as e:
                st.error(f"Greška pri formatiranju tabele: {e}")
                st.dataframe(df_res) 

            if not df_res.empty and len(snr_vals) > 1 and len(cfo_vals) > 1:
                st.write("### 2. BER Heatmap (Vizualizacija)")
                st.caption("Crveno = Visok BER (Loše), Zeleno/Plavo = Nizak BER (Dobro)")
                
                try:
                    pivot_ber = df_res.pivot(index="SNR (dB)", columns="CFO (Hz)", values="BER")
                    
                    fig_h, ax_h = plt.subplots(figsize=(10, 6))
                    im = ax_h.imshow(pivot_ber, cmap="RdYlGn_r", aspect='auto', vmin=0, vmax=0.5, origin='lower')
                    
                    ax_h.set_xticks(np.arange(len(pivot_ber.columns)))
                    ax_h.set_xticklabels([f"{x:.0f}" for x in pivot_ber.columns], rotation=45)
                    ax_h.set_yticks(np.arange(len(pivot_ber.index)))
                    ax_h.set_yticklabels([f"{x:.1f}" for x in pivot_ber.index])
                    
                    ax_h.set_xlabel("CFO (Hz)")
                    ax_h.set_ylabel("SNR (dB)")
                    ax_h.set_title("Bit Error Rate (BER) Performance")
                    
                    cbar = fig_h.colorbar(im, ax=ax_h)
                    cbar.set_label("BER")
                    
                    st.pyplot(fig_h)
                except Exception as e:
                    st.warning(f"Nije moguće iscrtati heatmap (vjerovatno nedovoljno podataka): {e}")