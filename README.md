# LTE Transceiver

## Project Description
This project implements an LTE transceiver. The goal of the project is to enable
 simulation and/or practical implementation of sending and receiving LTE signals between devices or simulated channels.
 The project is educational in nature and can serve as a foundation for further development of telecommunication systems and experimentation with LTE protocols.

## Features
- Sending and receiving LTE signals
- Channel and interference simulation
- Basic LTE communication functionalities (modulation, demodulation, encoding, and decoding of data)
- Monitoring and logging of transmitted data


## Repository Structure

lte-primopredajnik/
│-channel
│-common
│-docs
│-receiver
│-examples
│-tests
│-transmitter



## Opis projekta
Ovaj projekat implementira LTE primopredajnik. Cilj projekta je omogućiti simulaciju/praktičnu implementaciju slanja 
i prijema LTE signala između uređaja ili simuliranih kanala. 
Projekat je edukativnog karaktera i može poslužiti kao osnova za dalji razvoj telekomunikacionih sistema i eksperimentisanje sa LTE protokolima.

## Šta se implementira
- Slanje i prijem LTE signala 
- Simulacija kanala i interferencije
- Osnovne funkcionalnosti LTE komunikacije (modulacija, demodulacija, enkodiranje i dekodiranje podataka)
- Monitoring i logovanje prenesenih podataka


## Struktura

lte-primopredajnik/
│-channel
│-common
│-docs
│-receiver
│-examples
│-tests
│-transmitter


Automatski generisana tehnička dokumentacija (Doxygen): [docs/html/index.html](https://tsigrupa08.github.io/lte-primopredajnik/html/)


# PSS sinhronizacija (LTE RX)

Ovaj modul implementira obradu Primary Synchronization Signal (PSS)
u LTE prijemniku. Modul vrši detekciju PSS-a, procjenu vremenskog
pomaka (timing), procjenu frekvencijskog ofseta (CFO) i korekciju
frekvencijskog ofseta nad primljenim signalom.

## Funkcionalnosti

- Korelacija primljenog signala sa sve tri LTE PSS sekvence (N_ID_2 = 0, 1, 2)
- Detekcija ispravnog N_ID_2 indeksa
- Procjena vremenskog pomaka (τ̂ – tau_hat)
- Procjena frekvencijskog ofseta (CFO) na osnovu PSS-a
- Korekcija CFO-a korištenjem postojećeg FrequencyOffset modela

## Pozadina

U LTE sistemima, Primary Synchronization Signal (PSS) je Zadoff–Chu
sekvenca dužine 62, definisana u 3GPP TS 36.211 standardu.
PSS se koristi za početnu sinhronizaciju prijemnika, uključujući
detekciju ćelije i grubu vremensku i frekvencijsku sinhronizaciju.

Prijemnik vrši korelaciju primljenog signala sa svim mogućim PSS
sekvencama i bira onu koja daje najveći maksimum korelacije.
Vremenski položaj tog maksimuma daje procjenu početka PSS-a (τ̂).

Frekvencijski ofset (CFO) se procjenjuje iz prosječne fazne rotacije
između uzastopnih uzoraka detektovanog PSS segmenta.

## Primjer korištenja

```python
from rx.pss_sync import PSSSynchronizer

fs = 1.92e6  # sample rate u Hz
sync = PSSSynchronizer(sample_rate_hz=fs)

corr = sync.correlate(rx_waveform)
tau_hat, n_id_2 = sync.estimate_timing(corr)
cfo_hat = sync.estimate_cfo(rx_waveform, tau_hat, n_id_2)
rx_corrected = sync.apply_cfo_correction(rx_waveform, cfo_hat)
