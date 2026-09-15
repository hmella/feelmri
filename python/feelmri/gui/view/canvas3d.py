"""The 3D panel: VTK renders offscreen, the pixels are blitted into a Canvas.

PyVista has no tkinter embedding, so the render window is never shown. It
draws into an offscreen buffer, the buffer comes back as a numpy array, and
the array is pushed into a `tkinter.PhotoImage`. Mouse events on the canvas
drive `model.camera.Camera`, since without a shown window there is no VTK
interactor to do it.

The pixel hand-off is a PNG built with `zlib`, base64-encoded, which is what
`PhotoImage` accepts. Pillow would be faster, but it is not a declared
dependency and the 3D view is already optional; adding a second optional
dependency to draw the first would be a poor trade.

PNG rather than the simpler PPM because Tk's `-data` option only recognises
base64 for the formats that declare it, and PPM is not one of them: feeding it
a base64 PPM raises "couldn't recognize image data". Compression runs at level
1, since this is a frame buffer and not a file.

**Readback is the cost of this design.** Measured on an X-forwarded session,
a frame at 800x600 took roughly 250 ms end to end, and halving the window
nearly doubled the rate while decimating the mesh tenfold changed nothing:
the bottleneck is pixels, not triangles. Hence `interaction_scale`, which
renders smaller while the mouse is down and full size when it stops.
"""
from __future__ import annotations

import base64
import struct
import zlib
from typing import Callable, Optional

import numpy as np

from ..model.camera import STANDARD_VIEWS, Camera


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
  """One PNG chunk: length, type, payload, CRC of type and payload."""
  return (struct.pack('>I', len(payload)) + tag + payload
          + struct.pack('>I', zlib.crc32(tag + payload) & 0xFFFFFFFF))


def rgb_to_png(rgb: np.ndarray, level: int = 1) -> bytes:
  """A PNG for an `(H, W, 3)` uint8 array, using only the standard library.

  Colour type 2 is 8-bit RGB. Every scanline carries a leading filter byte,
  which is 0 here: filtering would shrink the file, and this one is never
  written to disk.
  """
  rgb = np.asarray(rgb)
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f'rgb_to_png: expected (H, W, 3), got {rgb.shape}')
  if rgb.dtype != np.uint8:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
  height, width = rgb.shape[:2]

  # A zero filter byte in front of each row, done as one array operation.
  rows = np.hstack([np.zeros((height, 1), dtype=np.uint8),
                    np.ascontiguousarray(rgb).reshape(height, width * 3)])
  return (b'\x89PNG\r\n\x1a\n'
          + _png_chunk(b'IHDR',
                       struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
          + _png_chunk(b'IDAT', zlib.compress(rows.tobytes(), level))
          + _png_chunk(b'IEND', b''))


def to_photoimage_data(rgb: np.ndarray) -> str:
  """The base64 string `tkinter.PhotoImage(data=...)` takes."""
  return base64.b64encode(rgb_to_png(rgb)).decode('ascii')


class Canvas3D:
  """A Tk canvas showing an offscreen VTK render of the session's mesh.

  Construction fails loudly when PyVista is missing rather than degrading
  silently; the shell catches that and swaps in the fallback panel.
  """

  def __init__(self, parent, session, width: int = 800, height: int = 600,
               interaction_scale: float = 0.5,
               on_status: Optional[Callable[[str], None]] = None):
    import tkinter as tk

    try:
      import pyvista as pv
    except ImportError as exc:
      raise RuntimeError(
        f'The 3D view needs PyVista and VTK, which are an optional extra: '
        f'pip install "feelmri[gui]" ({exc})') from exc

    self.session = session
    self.width, self.height = int(width), int(height)
    self.interaction_scale = float(interaction_scale)
    self.on_status = on_status or (lambda _: None)

    self.widget = tk.Canvas(parent, width=self.width, height=self.height,
                            highlightthickness=0, background='#1a1a1a')
    self._image_id = None
    self._photo = None            # a live reference, or Tk drops the image
    self._drag = None
    self._camera: Optional[Camera] = None
    self._actor = None
    self._glyph_actor = None
    self._overlay = []

    self._plotter = pv.Plotter(off_screen=True,
                               window_size=(self.width, self.height))
    self._plotter.set_background('#1a1a1a')

    self._bind()
    session.mesh_changed.connect(lambda *_: self.rebuild())
    session.plan_changed.connect(lambda *_: self.refresh_overlay())
    session.view_changed.connect(lambda *_: self.rebuild(keep_camera=True))

  # -- scene ----------------------------------------------------------------

  def rebuild(self, keep_camera: bool = False) -> None:
    """Rebuild the mesh actor from the session and redraw."""
    if not self.session.has_mesh:
      return
    import pyvista as pv

    points = self.session.points
    warp = self.session.warp_vectors()
    if warp is not None and warp.shape == points.shape:
      points = points + self.session.warp_scale * warp

    tris = self.session.surface
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    surface = pv.PolyData(np.ascontiguousarray(points, dtype=float), faces)

    # The session resolves the chosen field, so this backend and the native
    # one cannot disagree about what a label means. They each carried a copy
    # before, which is the duplication that makes one of two checks unable to
    # fail when the other is wrong.
    resolved = self.session.surface_colour_values()
    if resolved is not None:
      values, association = resolved
      if association == 'cell':
        surface.cell_data['field'] = np.asarray(values)[:surface.n_cells]
      else:
        surface.point_data['field'] = np.asarray(values)[:len(points)]

    if self._actor is not None:
      self._plotter.remove_actor(self._actor, render=False)
    self._actor = self._plotter.add_mesh(
      surface, scalars='field' if resolved is not None else None,
      preference='cell' if resolved and resolved[1] == 'cell' else 'point',
      cmap='viridis', show_scalar_bar=resolved is not None,
      # The bar carries the CHOSEN label, so a component and a magnitude of
      # the same field are told apart on the picture rather than only in the
      # combobox the user has since looked away from.
      scalar_bar_args={'title': str(self.session.field or '')},
      opacity=float(self.session.opacity),
      color=None if resolved is not None else '#c8c8c8')

    self._rebuild_glyphs()
    if self._camera is None or not keep_camera:
      lo, hi = points.min(axis=0), points.max(axis=0)
      self._camera = Camera.frame(lo, hi)
    self.refresh_overlay()

  def _rebuild_glyphs(self) -> None:
    """Arrows for the chosen vector field, sampled and scaled by the session."""
    import pyvista as pv

    if self._glyph_actor is not None:
      self._plotter.remove_actor(self._glyph_actor, render=False)
      self._glyph_actor = None
    arrows = self.session.glyph_arrows()
    if arrows is None:
      return
    points, vectors, factor = arrows
    cloud = pv.PolyData(np.ascontiguousarray(points, dtype=float))
    cloud['vectors'] = np.ascontiguousarray(vectors, dtype=float)
    cloud['magnitude'] = np.linalg.norm(vectors, axis=1)
    glyphs = cloud.glyph(orient='vectors', scale='magnitude', factor=factor,
                         geom=pv.Arrow())
    self._glyph_actor = self._plotter.add_mesh(
      glyphs, scalars='magnitude', cmap='plasma', show_scalar_bar=False,
      render=False)

  def refresh_overlay(self) -> None:
    """Redraw the FOV box and the M/P/S arrows, then repaint."""
    import pyvista as pv

    for actor in self._overlay:
      self._plotter.remove_actor(actor, render=False)
    self._overlay = []

    box = self.session.box
    if box is not None:
      corners = box.corners()
      # The 12 edges of a box whose corners are in (-,-,-)...(+,+,+) order.
      edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
               (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
      lines = np.hstack([[2, a, b] for a, b in edges])
      wire = pv.PolyData(corners, lines=lines)
      self._overlay.append(self._plotter.add_mesh(
        wire, color='#ffd24a', line_width=2, render=False))

      # A translucent solid inside the outline, so where the plan cuts the
      # phantom is visible rather than merely outlined. Built axis-aligned and
      # then transformed, since `pv.Box` takes axis-aligned bounds only and an
      # oblique plan would otherwise be drawn square.
      opacity = float(self.session.plan_opacity)
      if opacity > 0 and np.all(box.fov > 0):
        half = 0.5 * box.fov
        solid = pv.Box(bounds=(-half[0], half[0], -half[1], half[1],
                               -half[2], half[2]))
        transform = np.eye(4)
        transform[:3, :3] = box.mps
        transform[:3, 3] = box.loc
        solid.transform(transform, inplace=True)
        self._overlay.append(self._plotter.add_mesh(
          solid, color='#ffd24a', opacity=opacity, show_scalar_bar=False,
          render=False))

      for (origin, direction, _name), colour in zip(
          box.axis_arrows(), ('#ff4d4d', '#4dff88', '#4d9cff')):
        arrow = pv.Arrow(start=origin, direction=direction,
                         scale=float(np.linalg.norm(direction)))
        self._overlay.append(self._plotter.add_mesh(
          arrow, color=colour, render=False))

    self.draw()

  # -- painting -------------------------------------------------------------

  def draw(self, scale: float = 1.0) -> None:
    """Render offscreen and blit the result into the canvas."""
    import tkinter as tk

    if self._camera is None:
      return
    width = max(16, int(self.width * scale))
    height = max(16, int(self.height * scale))
    if tuple(self._plotter.window_size) != (width, height):
      self._plotter.window_size = (width, height)

    self._plotter.camera_position = self._camera.as_vtk()
    # The explicit render() is REQUIRED. `screenshot()` returns the LAST
    # RENDERED frame; it does not render on its own. Measured directly: move
    # the camera and screenshot without rendering and the image is unchanged
    # (mean abs difference 0.000), with a render it moves (4.660). Removing
    # this line as a "duplicate render" leaves a viewport whose camera
    # responds and whose picture never updates, which looks like a fast
    # viewer until frames are compared.
    self._plotter.render()
    rgb = self._plotter.screenshot(return_img=True)

    self._photo = tk.PhotoImage(data=to_photoimage_data(rgb))
    if scale != 1.0:
      # PhotoImage.zoom takes integers, so the reduced frame is scaled by the
      # nearest whole factor. It is a placeholder during a drag, not a result.
      factor = max(1, int(round(1.0 / scale)))
      self._photo = self._photo.zoom(factor, factor)

    if self._image_id is None:
      self._image_id = self.widget.create_image(0, 0, anchor='nw',
                                                image=self._photo)
    else:
      self.widget.itemconfigure(self._image_id, image=self._photo)

  # -- interaction ----------------------------------------------------------

  def _bind(self) -> None:
    w = self.widget
    w.bind('<ButtonPress-1>', self._press)
    w.bind('<B1-Motion>', self._orbit)
    w.bind('<ButtonRelease-1>', self._release)
    w.bind('<ButtonPress-3>', self._press)
    w.bind('<B3-Motion>', self._pan)
    w.bind('<ButtonRelease-3>', self._release)
    w.bind('<MouseWheel>', self._wheel)             # Windows and macOS
    w.bind('<Button-4>', lambda e: self._zoom(0.9))  # X11 scroll up
    w.bind('<Button-5>', lambda e: self._zoom(1.1))  # X11 scroll down
    w.bind('<Configure>', self._resized)

  def _press(self, event) -> None:
    self._drag = (event.x, event.y)

  def _release(self, _event) -> None:
    self._drag = None
    self.draw()                                   # full resolution on release

  def _orbit(self, event) -> None:
    if self._drag is None or self._camera is None:
      return
    dx, dy = event.x - self._drag[0], event.y - self._drag[1]
    self._drag = (event.x, event.y)
    # A drag across the full width is one turn; down tilts up, matching the
    # convention every 3D viewer uses.
    self._camera = self._camera.orbit(-2 * np.pi * dx / max(1, self.width),
                                      -np.pi * dy / max(1, self.height))
    self.draw(self.interaction_scale)

  def _pan(self, event) -> None:
    if self._drag is None or self._camera is None:
      return
    dx, dy = event.x - self._drag[0], event.y - self._drag[1]
    self._drag = (event.x, event.y)
    self._camera = self._camera.pan(dx / max(1, self.width),
                                    dy / max(1, self.height))
    self.draw(self.interaction_scale)

  def _wheel(self, event) -> None:
    self._zoom(0.9 if getattr(event, 'delta', 0) > 0 else 1.1)

  def _zoom(self, factor: float) -> None:
    if self._camera is None:
      return
    self._camera = self._camera.dolly(factor)
    self.draw()

  def _resized(self, event) -> None:
    if event.width > 1 and event.height > 1:
      self.width, self.height = int(event.width), int(event.height)
      self.draw()

  def look(self, name: str) -> None:
    """Jump to a standard view: axial, coronal or sagittal."""
    if self._camera is None:
      return
    if name not in STANDARD_VIEWS:
      raise ValueError(f'look: unknown view {name!r}, '
                       f'expected one of {sorted(STANDARD_VIEWS)}')
    direction, up = STANDARD_VIEWS[name]
    self._camera = self._camera.look_along(direction, up)
    self.draw()

  def reset_view(self) -> None:
    if self.session.has_mesh:
      lo, hi = self.session.points.min(axis=0), self.session.points.max(axis=0)
      self._camera = Camera.frame(lo, hi)
      self.draw()

  def close(self) -> None:
    try:
      self._plotter.close()
    except Exception:
      pass
