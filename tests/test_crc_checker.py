import numpy as np
import pytest

from receiver.crc_checker import CRCChecker
from transmitter.pbch import PBCHEncoder


class TestCRCChecker:
    """
    Test suite za CRCChecker (PBCH RX).

    Pokriva:
    - happy path (valid payload + CRC)
    - single-bit error
    - multiple-bit error
    - CRC-only corruption
    - minimalne i neispravne ulaze
    """

    def setup_method(self):
        """
        Poziva se prije svakog testa.
        """
        self.checker = CRCChecker()
        self.encoder = PBCHEncoder(verbose=False)

    # ==========================================================
    # HAPPY PATH
    # ==========================================================
    def test_valid_payload_crc_ok(self):
        """
        valid payload + crc -> ok = True
        """
        payload = np.random.randint(0, 2, 24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))
        payload_rx, ok = self.checker.check(bits_with_crc)

        assert ok is True
        assert np.array_equal(payload, payload_rx)

    # ==========================================================
    # SINGLE BIT ERROR
    # ==========================================================
    def test_single_bit_flip_detected(self):
        """
        flip jednog bita -> ok = False
        """
        payload = np.random.randint(0, 2, 24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))

        # flip jedan bit u payloadu
        bits_with_crc[5] ^= 1

        _, ok = self.checker.check(bits_with_crc)
        assert ok is False

    # ==========================================================
    # MULTIPLE BIT ERRORS
    # ==========================================================
    def test_multiple_bit_flips_detected(self):
        """
        flip vise bitova -> ok = False
        """
        payload = np.random.randint(0, 2, 24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))

        # flip vise bitova
        bits_with_crc[[2, 7, 13]] ^= 1

        _, ok = self.checker.check(bits_with_crc)
        assert ok is False

    # ==========================================================
    # CRC PART CORRUPTION
    # ==========================================================
    def test_crc_bits_corrupted(self):
        """
        payload ispravan, CRC dio korumpiran -> ok = False
        """
        payload = np.random.randint(0, 2, 24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))

        # flip jedan CRC bit
        bits_with_crc[-1] ^= 1

        _, ok = self.checker.check(bits_with_crc)
        assert ok is False

    # ==========================================================
    # EDGE CASES
    # ==========================================================
    def test_all_zero_payload(self):
        """
        payload = svi nule, CRC treba proci
        """
        payload = np.zeros(24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))
        payload_rx, ok = self.checker.check(bits_with_crc)

        assert ok is True
        assert np.array_equal(payload_rx, payload)

    def test_all_one_payload(self):
        """
        payload = sve jedinice, CRC treba proci
        """
        payload = np.ones(24, dtype=np.uint8)
        crc = self.encoder.crc16(payload)

        bits_with_crc = np.concatenate((payload, crc))
        payload_rx, ok = self.checker.check(bits_with_crc)

        assert ok is True
        assert np.array_equal(payload_rx, payload)

    # ==========================================================
    # INVALID INPUTS
    # ==========================================================
    def test_input_shorter_than_crc_raises(self):
        """
        ulaz kraci od 16 CRC bitova -> ValueError
        """
        bits = np.random.randint(0, 2, 10, dtype=np.uint8)

        with pytest.raises(ValueError):
            self.checker.check(bits)
