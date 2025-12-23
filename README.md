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

# DeRateMatcher

`DeRateMatcher` je Python klasa za **de-rate-matching primljenih bitova** u LTE prijemniku.  
Vraća bitove u originalni raspored prije rate matchinga, kao soft ili hard vrijednosti.

---

## Primjer korištenja

```python
import numpy as np
from derate_matcher import DeRateMatcher

# Inicijalizacija
matcher = DeRateMatcher(E_rx=1.0, N_coded=100)

# Primljeni bitovi nakon rate matchinga (soft vrijednosti ili 0/1)
bits_rx = np.random.rand(120)

# De-rate-matching: soft vrijednosti
soft_bits = matcher.accumulate(bits_rx, soft=True)
print("Soft bits:", soft_bits)

# De-rate-matching: hard bitovi
hard_bits = matcher.accumulate(bits_rx, soft=False)
print("Hard bits:", hard_bits)
