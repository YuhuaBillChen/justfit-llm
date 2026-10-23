"""Safe regression inputs: intercept allocations instead of exhausting memory."""
import io
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
import zlib

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from server.images import prepare, ImageError


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


class ImageSecurity(unittest.TestCase):
    def test_apng_rejects_huge_canvas_before_allocating(self):
        payload = b'\x89PNG\r\n\x1a\n'
        payload += chunk(b'IHDR', struct.pack('>IIBBBBB', 22000, 22000, 8, 6, 0, 0, 0))
        payload += chunk(b'acTL', struct.pack('>II', 1, 0))
        payload += chunk(b'fcTL', struct.pack('>IIIIIHHBB', 0, 1, 1, 0, 0, 1, 10, 1, 0))
        payload += chunk(b'IDAT', zlib.compress(b'\0\0\0\0\xff'))
        payload += chunk(b'IEND', b'')
        with patch.object(Image.core, 'fill', side_effect=AssertionError('unsafe allocation')) as fill:
            with self.assertRaises(ImageError):
                prepare(payload)
            fill.assert_not_called()

    def test_supported_images_still_decode(self):
        for kind in ('PNG', 'GIF', 'JPEG', 'WEBP'):
            with self.subTest(kind=kind):
                stream = io.BytesIO()
                Image.new('RGB', (256, 256), 'red').save(stream, format=kind)
                result = prepare(stream.getvalue())
                self.assertEqual((result.grid_height, result.grid_width), (16, 16))
                self.assertEqual(len(result.pixels), 256 * 256 * 3)

    def test_compromised_ca_removed(self):
        import certifi
        from cryptography import x509
        roots = x509.load_pem_x509_certificates(Path(certifi.where()).read_bytes())
        self.assertFalse(any('emSign ECC Root CA - G3' in cert.subject.rfc4514_string()
                             for cert in roots))
