"""The pixel path between VTK and Tk, and the no-VTK fallback export.

Blitting hands a numpy frame to `tkinter.PhotoImage`, which needs an encoded
image rather than an array. These tests cover the encoder without opening a
window, which is most of what can be checked headless; the rest of the canvas
is event wiring.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from feelmri.gui.view.canvas3d import (_png_chunk, rgb_to_png,
                                       to_photoimage_data)

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def test_the_encoder_produces_a_structurally_valid_png():
  """Signature, an IHDR that agrees with the array, and a terminating IEND."""
  rgb = np.zeros((4, 6, 3), dtype=np.uint8)
  png = rgb_to_png(rgb)

  assert png.startswith(PNG_SIGNATURE)
  assert png.endswith(_png_chunk(b'IEND', b''))

  # IHDR is the first chunk: 4-byte length, tag, payload, CRC.
  length = struct.unpack('>I', png[8:12])[0]
  assert png[12:16] == b'IHDR' and length == 13
  width, height, depth, colour = struct.unpack('>IIBB', png[16:26])
  assert (width, height) == (6, 4)
  assert depth == 8 and colour == 2, 'expected 8-bit RGB'


def test_every_chunk_carries_a_correct_crc():
  """A wrong CRC is the failure a decoder reports as unrecognisable data."""
  png = rgb_to_png(np.zeros((3, 3, 3), dtype=np.uint8))
  offset = len(PNG_SIGNATURE)
  seen = []
  while offset < len(png):
    length = struct.unpack('>I', png[offset:offset + 4])[0]
    tag = png[offset + 4:offset + 8]
    payload = png[offset + 8:offset + 8 + length]
    crc = struct.unpack('>I', png[offset + 8 + length:offset + 12 + length])[0]
    assert crc == zlib.crc32(tag + payload) & 0xFFFFFFFF, f'bad CRC on {tag}'
    seen.append(tag)
    offset += 12 + length
  assert seen == [b'IHDR', b'IDAT', b'IEND']


def test_the_pixels_survive_the_encoding():
  """Decode the IDAT by hand and compare against the array that went in.

  Scanlines carry a leading filter byte, which this encoder always sets to 0,
  so the payload is the rows verbatim once those bytes are dropped.
  """
  rng = np.random.default_rng(0)
  rgb = rng.integers(0, 255, (5, 7, 3), dtype=np.uint8)
  png = rgb_to_png(rgb)

  offset = len(PNG_SIGNATURE)
  idat = None
  while offset < len(png):
    length = struct.unpack('>I', png[offset:offset + 4])[0]
    if png[offset + 4:offset + 8] == b'IDAT':
      idat = png[offset + 8:offset + 8 + length]
      break
    offset += 12 + length
  assert idat is not None

  raw = zlib.decompress(idat)
  height, width = rgb.shape[:2]
  stride = width * 3 + 1
  assert len(raw) == height * stride
  for y in range(height):
    row = raw[y * stride:(y + 1) * stride]
    assert row[0] == 0, 'filter byte is not zero'
    np.testing.assert_array_equal(
      np.frombuffer(row[1:], dtype=np.uint8).reshape(width, 3), rgb[y])


def test_a_float_frame_is_clipped_rather_than_wrapping():
  """VTK can hand back floats; casting without clipping wraps 300 to 44."""
  png = rgb_to_png(np.array([[[300.0, -20.0, 128.0]]]))
  raw = zlib.decompress(png[
    png.index(b'IDAT') + 4:][:struct.unpack(
      '>I', png[png.index(b'IDAT') - 4:png.index(b'IDAT')])[0]])
  assert tuple(raw[1:4]) == (255, 0, 128)


def test_a_wrong_shape_is_refused_by_name():
  for bad in (np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros(4)):
    with pytest.raises(ValueError, match=r'expected \(H, W, 3\)'):
      rgb_to_png(bad)


def test_the_photoimage_payload_is_base64_of_the_png():
  import base64
  rgb = np.zeros((2, 2, 3), dtype=np.uint8)
  data = to_photoimage_data(rgb)
  assert isinstance(data, str)
  assert base64.b64decode(data) == rgb_to_png(rgb)


def test_the_module_imports_without_tkinter_or_vtk():
  """The encoder must be reachable on a machine with neither installed.

  Both are imported inside `Canvas3D.__init__`, never at module scope, so the
  fallback path and these tests keep working.
  """
  import subprocess
  import sys
  import textwrap

  code = textwrap.dedent('''
    import sys
    class Block:
      def find_module(self, name, path=None):
        return self if name.split('.')[0] in ('tkinter', 'pyvista', 'vtk') else None
      def load_module(self, name):
        raise ImportError(name + ' is blocked')
    sys.meta_path.insert(0, Block())
    import numpy as np
    from feelmri.gui.view.canvas3d import rgb_to_png
    assert rgb_to_png(np.zeros((2, 2, 3), np.uint8)).startswith(b'\\x89PNG')
    print('OK')
  ''')
  proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                        text=True, timeout=120)
  assert 'OK' in proc.stdout, proc.stderr[-2000:]
