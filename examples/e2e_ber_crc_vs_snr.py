"""
End-to-end simulacija BER i CRC success rate vs SNR za BPSK sistem preko AWGN kanala.

Ova skripta demonstrira funkcionalnost RX lanca tako da pokazuje:
- kako BER opada sa rastom SNR
- kako CRC success rate raste sa rastom SNR

Biblioteke:
-----------
numpy : np
    Za generisanje bitova, matematičke operacije i statistiku.
matplotlib.pyplot : plt
    Za crtanje grafova.

Funkcije:
----------
add_awgn_noise(signal, snr_db)
    Dodaje AWGN šum signalu za dati SNR u dB.
    
    Parametri
    ----------
    signal : ndarray
        Ulazni BPSK signal (-1/+1)
    snr_db : float
        SNR u decibelima

    Povratna vrijednost
    ------------------
    ndarray
        Signal sa dodatim AWGN šumom

crc_check(original_bits, decoded_bits)
    Idealizovana CRC provjera.
    Vraća True ako nema nijedne bitne greške.

    Parametri
    ----------
    original_bits : ndarray
        Originalni TX bitovi
    decoded_bits : ndarray
        RX bitovi nakon detekcije

    Povratna vrijednost
    ------------------
    bool
        True ako su svi bitovi identični, False inače

Parametri simulacije:
--------------------
snr_range : ndarray
    Vektor SNR vrijednosti [dB] za koje se vrši simulacija.
num_trials : int
    Broj trial-ova po SNR vrijednosti.
msg_len : int
    Dužina poruke u bitovima po trial-u.

Rezultati:
----------
ber_results : list
    Lista BER vrijednosti po SNR.
crc_success_results : list
    Lista CRC success rate po SNR.

Primjer korištenja:
------------------
Pokretanje skripte iz root foldera projekta:
>>> python examples/e2e_ber_crc_vs_snr.py

Rezultat:
- Dva grafa: BER vs SNR i CRC success rate vs SNR
- Pokazuje da RX bolje radi kad SNR raste

Napomene:
---------
- Hard decision detekcija se koristi za BPSK: u_hat = (rx_signal > 0)
- crc_check je idealizovana, pravi pravi CRC nije implementiran
- Rezultati su prosjek preko num_trials da bi se izbjegla random fluktuacija
"""

import numpy as np
import matplotlib.pyplot as plt

# AWGN channel: dodavanje bijelog Gaussovog šuma

def add_awgn_noise(signal, snr_db):
    """
    Dodaje AWGN šum signalu za dati SNR u dB.
    """
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    return signal + noise



# CRC provjera (idealni model)

def crc_check(original_bits, decoded_bits):
    """
    Idealizovana CRC provjera:
    uspjeh ako nema nijedne bitne greške.
    """
    return np.array_equal(original_bits, decoded_bits)



# Parametri simulacije

snr_range = np.arange(0, 11, 2)   # SNR vrijednosti: 0, 2, ..., 10 dB
num_trials = 200                  # broj ponavljanja po SNR-u
msg_len = 100                     # dužina poruke u bitovima po trial-u

ber_results = []
crc_success_results = []

# (opcionalno) za ponovljive rezultate
np.random.seed(0)



# Glavna SNR sweep petlja

for snr in snr_range:
    bit_errors = 0
    total_bits = 0
    crc_success = 0

    for _ in range(num_trials):
        # Generisanje slučajnih bitova
        u = np.random.randint(0, 2, msg_len)

        # BPSK modulacija (0 -> -1, 1 -> +1)
        tx_signal = 2 * u - 1

        # Kanal: AWGN
        rx_signal = add_awgn_noise(tx_signal, snr)

        # Hard decision detekcija
        u_hat = (rx_signal > 0).astype(int)

        # BER računanje
        bit_errors += np.sum(u != u_hat)
        total_bits += msg_len

        # CRC success
        if crc_check(u, u_hat):
            crc_success += 1

    ber_results.append(bit_errors / total_bits)
    crc_success_results.append(crc_success / num_trials)



# Vizualizacija rezultata

plt.figure(figsize=(12, 5))

# BER vs SNR
plt.subplot(1, 2, 1)
plt.semilogy(snr_range, ber_results, 'o-', label="BER (simulacija)")
plt.xlabel("SNR [dB]")
plt.ylabel("Bit Error Rate")
plt.title("BER vs SNR")
plt.grid(True, which="both")
plt.legend()

# CRC success vs SNR
plt.subplot(1, 2, 2)
plt.plot(snr_range, crc_success_results, 's-', label="CRC success rate")
plt.xlabel("SNR [dB]")
plt.ylabel("CRC success rate")
plt.title("CRC success rate vs SNR")
plt.ylim([0, 1])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

