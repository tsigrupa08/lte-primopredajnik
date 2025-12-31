"""
LTE PBCH End-to-End Simulacija Performansi

Ovaj modul implementira Monte Carlo simulaciju LTE prijemnog lanca (RX chain)
za Physical Broadcast Channel (PBCH). Cilj je evaluacija performansi sistema
u prisustvu aditivnog bijelog Gaussovog šuma (AWGN).

Namjena
-------
Skripta generiše Bit Error Rate (BER) i CRC Success Rate krive u zavisnosti
od odnosa signal-šum (SNR). Koristi se za verifikaciju ispravnosti implementacije
prijemnika (sinkronizacija, demodulacija, dekodiranje).

Struktura simulacije
--------------------
1. **Tx (Predajnik):** Generisanje MIB payload-a, kodiranje i OFDM modulacija.
2. **Kanal:** Dodavanje AWGN šuma za zadati SNR.
3. **Rx (Prijemnik):** Sinkronizacija (PSS), OFDM demodulacija, QPSK demapiranje,
   Viterbi dekodiranje i CRC provjera.

Rezultati
---------
Rezultati se prikazuju numerički u konzoli i grafički putem Matplotlib biblioteke.
Generisani grafikon se automatski spašava u direktorij `examples/results/rx`.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTI MODULA ---
from transmitter.LTETxChain import LTETxChain
from channel.awgn_channel import AWGNChannel
from receiver.LTERxChain import LTERxChain


def run_simulation():
    """
    Pokreće glavnu petlju simulacije za niz SNR vrijednosti.

    Proces:
    -------
    Iterira kroz definisane SNR tačke. Za svaku tačku izvršava zadati broj
    ponavljanja (trials). U svakom trial-u:
      1. Generiše nasumične podatke i LTE waveform.
      2. Dodaje šum.
      3. Pokušava demodulirati i dekodirati podatke.
      4. Računa bitovske greške i provjerava CRC.

    Parametri simulacije (Hardcoded):
    ---------------------------------
    - snr_points : list
        Tačke SNR-a u dB: [0, 2, 4, 6, 8, 10].
    - num_trials : int
        Broj Monte Carlo ponavljanja po tački: 100.
    - NDLRB : int
        Broj resursnih blokova: 6.
    - NUM_SUBFRAMES : int
        Broj subfrejmova: 4 (Neophodno za PBCH TTI od 40 ms).

    Povratna vrijednost
    -------------------
    None. Funkcija ispisuje rezultate i poziva funkciju za plotanje.
    """
    

    # 1. KONFIGURACIJA SIMULACIJE
    snr_points = [0, 2, 4, 6, 8, 10]  
    num_trials = 100                  
    
    # Parametri sistema
    NDLRB = 6
    NORMAL_CP = True
    NUM_SUBFRAMES = 4  # OBAVEZNO >= 4 za PBCH
    
    # Liste za čuvanje rezultata
    ber_results = []
    crc_success_results = []
    
    print(f"Pokrećem simulaciju: {len(snr_points)} SNR tačaka, {num_trials} trial-a po tački.")
    print("-" * 75)
    print(f"{'SNR [dB]':<10} | {'BER':<12} | {'CRC Success':<12} | {'Errors/Total Bits':<20}")
    print("-" * 75)

    
    # 2. GLAVNA PETLJA (SNR LOOP)

    for snr in snr_points:
        
        crc_ok_count = 0
        bit_err = 0
        total_bits = 0
        
        # Inicijalizacija kanala (AWGN)
        channel = AWGNChannel(snr_db=snr, seed=None) 
        
        for trial in range(num_trials):
            # A) TX: Generisanje 24 bita (MIB) i waveform-a
            mib_tx = np.random.randint(0, 2, 24)
            
            tx_chain = LTETxChain(
                n_id_2=0, 
                ndlrb=NDLRB, 
                num_subframes=NUM_SUBFRAMES, 
                normal_cp=NORMAL_CP
            )
            tx_waveform, fs = tx_chain.generate_waveform(mib_bits=mib_tx)
            
            # B) KANAL: Dodavanje šuma
            rx_waveform = channel.apply(tx_waveform)
            
            # C) RX: Procesiranje
            rx_chain = LTERxChain(
                sample_rate_hz=fs, 
                ndlrb=NDLRB, 
                normal_cp=NORMAL_CP
            )
            
            # Dobijanje rezultata
            result = rx_chain.process(rx_waveform)
            
            mib_rx = result['mib_bits']
            is_crc_ok = result['crc_ok']
            
            # D) METRIKE
            # 1. Brojanje CRC uspjeha (Block Error Rate statistika)
            if is_crc_ok:
                crc_ok_count += 1
            
            # 2. Računanje BER-a (Bit Error Rate)
            L_tx = len(mib_tx)
            L_rx = len(mib_rx)
            
            if L_rx == L_tx:
                errors = np.sum(np.abs(mib_tx - mib_rx))
                bit_err += errors
                total_bits += L_tx
            else:
                # U slučaju neuspjeha dimenzija (fallback), brojimo sve kao grešku
                bit_err += L_tx
                total_bits += L_tx

        # Statistika za trenutnu SNR tačku
        ber = bit_err / total_bits if total_bits > 0 else 1.0
        crc_rate = crc_ok_count / num_trials
        
        ber_results.append(ber)
        crc_success_results.append(crc_rate)
        
        print(f"{snr:<10.1f} | {ber:<12.5f} | {crc_rate:<12.2%} | {bit_err}/{total_bits}")

    print("-" * 75)
    print("Simulacija završena.")

    # 3. VIZUALIZACIJA I SPAŠAVANJE
    plot_combined_results(snr_points, ber_results, crc_success_results)


def plot_combined_results(snr_points, ber, crc_success):
    """
    Vizualizira rezultate simulacije na jednom grafikonu sa dvije Y-ose
    i spašava sliku u definisani direktorij.

    Grafikon prikazuje:
    1. BER (Logaritamska skala) na lijevoj osi.
    2. CRC Success Rate (Linearna skala) na desnoj osi.

    Parametri
    ---------
    snr_points : list of floatOrInt
        Vrijednosti SNR-a (x-osa).
    ber : list of float
        Izračunati Bit Error Rate za svaku tačku.
    crc_success : list of float
        Izračunati CRC Success Rate (0.0 - 1.0) za svaku tačku.

    Izlaz
    -----
    Kreira fajl 'ber_crc_vs_snr.png' u folderu 'examples/results/rx'.
    """
    
    # Kreiranje figure
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- LIJEVA OSA (BER - Logaritamska skala) ---
    color_ber = 'blue'
    ax1.set_xlabel('SNR [dB]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bit Error Rate (BER) - Log Skala', color=color_ber, fontsize=12, fontweight='bold')
    
    # BER crtamo logaritamski (semilogy)
    line1 = ax1.semilogy(
        snr_points, ber, 'o--', color=color_ber, linewidth=2, markersize=8, label='BER (Lijeva osa)'
    )
    ax1.tick_params(axis='y', labelcolor=color_ber)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- DESNA OSA (CRC Success - Linearna skala 0-1) ---
    ax2 = ax1.twinx()  
    color_crc = 'red'
    ax2.set_ylabel('CRC Success Rate (0 - 1)', color=color_crc, fontsize=12, fontweight='bold')
    
    line2 = ax2.plot(
        snr_points, crc_success, 's-', color=color_crc, linewidth=2, markersize=8, label='CRC Success (Desna osa)'
    )
    ax2.tick_params(axis='y', labelcolor=color_crc)
    ax2.set_ylim(-0.05, 1.05) # Margine za preglednost

    # --- LEGENDA I NASLOV ---
    # Kombinujemo handle-ove linija za zajedničku legendu
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    
    # Pozicioniranje legende IZNAD grafika
    ax1.legend(
        lines, labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=2, 
        frameon=True, 
        fontsize=10
    )

    # Naslov pomjeramo gore da napravimo mjesto za legendu
    plt.title('End-to-End LTE PBCH Performanse: BER i CRC Success vs SNR', fontsize=14, y=1.15)
    plt.tight_layout()

    # --- SPAŠAVANJE SLIKE ---
    # Definisanje putanje: examples/results/rx
    output_dir = os.path.join("examples", "results", "rx")
    
    # Kreiranje foldera ako ne postoje
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"\nKreiran direktorij: {output_dir}")
        except OSError as e:
            print(f"\nGreška pri kreiranju direktorija: {e}")
            return

    # Ime fajla
    output_filename = "ber_crc_vs_snr.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Spašavanje
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grafik uspješno sačuvan na lokaciji: {output_path}")
    except Exception as e:
        print(f"Greška pri spašavanju slike: {e}")

    # Prikaz na ekranu
    plt.show()


if __name__ == "__main__":
    run_simulation()