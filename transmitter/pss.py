import numpy as np

def generate_pss_sequence(nid2):
    """
    Generate LTE Primary Synchronization Signal (PSS) sequence.
    nid2 = 0, 1, or 2
    """

    # Map N2 to u according to 3GPP 36.211 Table 6.11.1.1-1
    if nid2 == 0:
        u = 25
    elif nid2 == 1:
        u = 29
    elif nid2 == 2:
        u = 34
    else:
        raise ValueError("nid2 must be 0, 1, or 2")

    d = np.zeros(62, dtype=complex)

    # First section: n = 0..30
    n1 = np.arange(0, 31)
    d[n1] = np.exp(-1j * np.pi * u * n1 * (n1 + 1) / 63)

    # Second section: n = 31..61
    n2 = np.arange(31, 62)
    d[n2] = np.exp(-1j * np.pi * u * (n2 + 1) * (n2 + 2) / 63)

    return d
