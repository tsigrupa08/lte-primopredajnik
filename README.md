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

### Opis direktorija

- `transmitter/` – LTE predajnik: PSS generisanje, PBCH lanac, OFDM modulacija  
- `channel/` – Model kanala (AWGN, CFO)  
- `receiver/` – LTE prijemnik: sinhronizacija, demodulacija, dekodiranje  
- `LTE_system_/` – End-to-end LTE sistem (TX → Channel → RX integracija)  
- `common/` – Zajedničke util funkcije, konstante i pomoćni moduli  
- `tests/` – Unit i end-to-end testovi (pytest)  
- `examples/` – Primjeri korištenja i demonstracione skripte  
- `gui/` – Grafički interfejs za vizualizaciju i interaktivno pokretanje  
- `docs/` – Tehnička dokumentacija i reference

## Tok obrade signala (workflow)

1. Generisanje PBCH/MIB informacija  
2. Kodiranje (CRC + konvolucijsko kodiranje)  
3. QPSK modulacija i mapiranje u resursni grid  
4. OFDM modulacija (IFFT + CP)  
5. Prolazak kroz kanal (AWGN + CFO)  
6. PSS sinhronizacija i procjena frekvencijskog ofseta  
7. OFDM demodulacija (CP removal + FFT)  
8. Ekstrakcija PBCH simbola  
9. Demapiranje i dekodiranje (Viterbi + CRC)  
10. Evaluacija ispravnosti prijema (CRC, BER)


## Trenutno stanje projekta

Projekat je u potpunosti završen i predstavlja kompletno implementiran LTE primopredajni sistem za simulaciju digitalne komunikacije u osnovnom opsegu.
- Implementiran je predajnik (TX) koji obuhvata generisanje PSS sekvenci, PBCH kodiranje, mapiranje simbola i OFDM modulaciju u skladu sa osnovnim LTE principima.
- Implementiran je kanal prenosa, uključujući dodavanje AWGN šuma i modeliranje frekvencijskog ofseta (Carrier Frequency Offset – CFO), čime su obuhvaćeni ključni degradirajući efekti realnog bežičnog kanala.
- Implementiran je potpuno funkcionalan prijemnik (RX) koji uključuje:
1. PSS baziranu vremensku i frekvencijsku sinhronizaciju,
2. OFDM demodulaciju,
3. ekstrakciju PBCH simbola iz primljenog resursnog grida, demapiranje, dekodiranje i CRC provjeru PBCH informacije.

Omogućeno je end-to-end testiranje kompletnog LTE sistema (TX → Channel → RX), pri čemu se potvrđuje ispravna rekonstrukcija prenesenih informacija na prijemu.
Razvijen je sveobuhvatan skup unit testova i end-to-end testova koji validiraju ispravnost pojedinačnih modula, kao i ukupnu funkcionalnost sistema.
Dostupne su vizualizacije ključnih faza prijema i obrade signala, uključujući PSS korelaciju, prijemni resursni grid, konstelacione dijagrame i rekonstrukciju primljenih bitova, što omogućava detaljnu analizu ponašanja sistema.


## Pokretanje projekta

### Kloniranje repozitorija

- git clone https://github.com/tsigrupa08/lte-primopredajnik.git
- cd lte-primopredajnik

### Kreiranje virtualnog okruženja
Radi izolacije zavisnosti projekta i osiguranja reproduktivnog okruženja za razvoj i testiranje, preporučuje se korištenje Python virtualnog okruženja.

Virtualno okruženje omogućava:
- odvajanje projektnih zavisnosti od sistemskog Python-a
- izbjegavanje konflikata između verzija biblioteka
- jednostavno pokretanje projekta na različitim računarima i operativnim sistemima

### Aktivacija virtualnog okruženja:

- Linux / macOS:
source .venv/bin/activate

- Windows:
.venv\Scripts\activate

### Instalacija zavisnosti
Nakon kreiranja i aktivacije virtualnog okruženja, potrebno je instalirati sve Python biblioteke koje projekat koristi. Spisak zavisnosti definisan je u fajlu
equirements.txt.

- pip install -r requirements.txt
  
## Pokretanje GUI aplikacije

GUI aplikacija omogućava interaktivno pokretanje LTE TX → Channel → RX lanca uz grafički prikaz ključnih faza obrade signala.

- python gui/gui_tx_channel.py

Prije pokretanja GUI aplikacije potrebno je aktivirati virtualno okruženje u kojem su instalirane sve zavisnosti projekta.

## Testiranje

### Vrste testova
**Unit testovi**
  - provjera funkcionalnosti pojedinačnih blokova (TX, Channel, RX)
  - testiranje PSS generisanja i detekcije
  - testiranje PBCH kodiranja i dekodiranja
  - validacija OFDM modulacije i demodulacije
  - CRC provjera ispravnosti prijema

**End-to-end testovi**
  - testiranje kompletnog LTE lanca (TX → Channel → RX)
  - provjera ispravnosti dekodiranih PBCH/MIB informacija
  - evaluacija grešaka u prisustvu šuma i frekvencijskog ofseta

Testovi obuhvataju i **ispravne (happy)** i **neispravne (unhappy)** scenarije, uključujući pogrešnu sinhronizaciju, prisustvo šuma i greške u prijemu.

### Pokretanje testova

Pokretanje svih testova iz korijenskog direktorija projekta:
- pytest

## Dokumentacija

Automatski generisana tehnička dokumentacija (Doxygen): [docs/html/index.html](https://tsigrupa08.github.io/lte-primopredajnik/html/).

## Zaključak
U okviru ovog projekta realizovan je kompletan LTE primopredajni sistem, koji obuhvata sve ključne faze digitalne bežične komunikacije – od generisanja signala na predaji, preko modeliranja kanala, do pouzdane rekonstrukcije informacija na prijemu.

Implementacijom predajnika, kanala i u potpunosti funkcionalnog prijemnika omogućena je end-to-end simulacija LTE komunikacionog lanca, čime je potvrđena ispravnost međusobne interakcije svih podsistema. Poseban akcenat stavljen je na proces sinhronizacije i dekodiranja na prijemu, koji predstavlja jednu od najzahtjevnijih faza LTE sistema.

Razvijeni unit testovi i end-to-end testovi dodatno potvrđuju stabilnost i tačnost implementacije, dok dostupne vizualizacije omogućavaju detaljnu analizu ponašanja sistema i olakšavaju razumijevanje pojedinih faza obrade signala.

Ovim projektom demonstrirana je prak­tična primjena teorijskih koncepata LTE standarda, kao i sposobnost projektovanja, implementacije i verifikacije složenog komunikacionog sistema. Implementirani LTE simulacioni lanac predstavlja čvrstu osnovu za dalja istraživanja, analizu performansi i nadogradnju sistema, te ima značajnu edukativnu i istraživačku vrijednost.
