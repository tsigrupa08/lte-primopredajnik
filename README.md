# LTE Transceiver

## Opis projekta
Ovaj projekat implementira LTE primopredajnik. Cilj projekta je omogućiti simulaciju i/ili
praktičnu implementaciju slanja i prijema LTE signala između uređaja ili simuliranih kanala.
Projekat je edukativnog karaktera i može poslužiti kao osnova za dalji razvoj
telekomunikacionih sistema i eksperimentisanje sa LTE protokolima.

---

## Šta se implementira

### Transmitter (TX)
- Generisanje PSS sekvenci i mapiranje u resursni grid  
- PBCH lanac: CRC, konvolucijsko kodiranje, rate matching i QPSK modulacija  
- OFDM modulacija (IFFT i dodavanje cikličkog prefiksa)

### Channel
- Simulacija AWGN kanala  
- Modeliranje frekvencijskog ofseta (CFO)

### Receiver (RX)
- PSS sinhronizacija (detekcija N_ID_2, procjena vremena i frekvencije)  
- OFDM demodulacija (uklanjanje CP i FFT)  
- Ekstrakcija PBCH simbola iz resursnog grida  
- QPSK demapiranje, de-rate matching i Viterbi dekodiranje  
- CRC provjera ispravnosti prijema

### End-to-End LTE sistem
- Integracija TX → Channel → RX  
- Dekodiranje PBCH/MIB informacija  
- Evaluacija ispravnosti prijema (CRC, BER)

### Testiranje i vizualizacije
- Unit testovi za sve ključne blokove (happy i unhappy slučajevi)  
- End-to-end testovi LTE sistema  
- Vizualizacije: PSS korelacija, RX grid, konstelacije i prikaz “od bitova do bitova”

---

## Struktura projekta

lte-primopredajnik/
│-channel
│-common
│-docs
│-examples
│-gui
│-LTE_system_
│-receiver
│-tests
│-transmitter

## Trenutno stanje projekta

Projekat je u funkcionalnoj fazi i implementiran je kompletan osnovni LTE lanac za simulaciju komunikacije.

- Implementiran je predajnik (TX) sa generisanjem PSS sekvenci, PBCH kodiranjem i OFDM modulacijom.
- Implementiran je kanal sa AWGN šumom i frekvencijskim ofsetom (CFO).
- Implementiran je prijemnik (RX) koji obuhvata PSS sinhronizaciju, OFDM demodulaciju, ekstrakciju PBCH simbola, demapiranje, dekodiranje i CRC provjeru.
- Omogućeno je end-to-end testiranje kompletnog sistema (TX → Channel → RX).
- Razvijeni su unit testovi i end-to-end testovi za validaciju funkcionalnosti.
- Dostupne su vizualizacije ključnih faza sistema (PSS korelacija, RX grid, konstelacije i prijem bitova).

Projekat je spreman za dalje proširenje, optimizaciju i analizu performansi.

## Pokretanje projekta

### Kloniranje repozitorija

- git clone https://github.com/tsigrupa08/lte-primopredajnik.git
- cd lte-primopredajnik

### Kreiranje virtualnog okruženja

- python -m venv .venv

### Aktivacija virtualnog okruženja:

- Linux / macOS:
source .venv/bin/activate

- Windows:
.venv\Scripts\activate

### Instalacija zavisnosti

- pip install -r requirements.txt

## Testiranje

Testovi koriste pytest i pokreću se iz terminala naredbom 'pytest'.

## Dokumentacija

Automatski generisana tehnička dokumentacija (Doxygen): [docs/html/index.html](https://tsigrupa08.github.io/lte-primopredajnik/html/).


