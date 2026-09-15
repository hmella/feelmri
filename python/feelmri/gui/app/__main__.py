"""`python -m feelmri.gui`.

`--check` reports what this machine can do and exits without opening a window,
so an install can be verified over ssh with no display.
"""
from __future__ import annotations

import argparse
import sys


def check() -> int:
  """Print what is available. Returns 0 when a window could be opened."""
  import os

  lines = []
  ok_tk = False
  try:
    import tkinter
    try:
      root = tkinter.Tk()
      root.withdraw()
      lines.append(f'tkinter      yes   Tk {root.tk.call("info", "patchlevel")}')
      root.destroy()
      ok_tk = True
    except Exception as exc:
      lines.append(f'tkinter      NO    installed, but no display '
                   f'(DISPLAY={os.environ.get("DISPLAY", "<unset>")!r}): {exc}')
  except ImportError as exc:
    lines.append(f'tkinter      NO    {exc}. On Debian or Ubuntu: '
                 f'sudo apt install python3-tk')

  try:
    import pyvista
    import vtk
    lines.append(f'pyvista/vtk  yes   pyvista {pyvista.__version__}, '
                 f'vtk {vtk.VTK_VERSION}')
    try:
      rw = vtk.vtkRenderWindow()
      rw.SetOffScreenRendering(1)
      rw.SetSize(64, 64)
      rw.AddRenderer(vtk.vtkRenderer())
      rw.Render()
      renderer = next((l.split(':', 1)[1].strip()
                       for l in rw.ReportCapabilities().splitlines()
                       if 'OpenGL renderer string' in l), 'unknown')
      lines.append(f'rendering    yes   {renderer}')
      rw.Finalize()
    except Exception as exc:
      lines.append(f'rendering    NO    {type(exc).__name__}: {exc}. Offscreen '
                   f'still needs a display unless VTK has EGL or OSMesa.')
  except ImportError as exc:
    lines.append(f'pyvista/vtk  NO    {exc}. The 3D view is optional: '
                 f'pip install "feelmri[gui]"')

  print('\n'.join(lines))
  return 0 if ok_tk else 1


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(prog='python -m feelmri.gui',
                                   description='feelmri graphical interface')
  parser.add_argument('phantom', nargs='?',
                      help='mesh to open on startup')
  parser.add_argument('--check', action='store_true',
                      help='report capabilities and exit, opening no window')
  args = parser.parse_args(argv)

  if args.check:
    return check()

  try:
    from .shell import Shell
  except ImportError as exc:
    print(f'The interface needs tkinter: {exc}\n'
          f'On Debian or Ubuntu: sudo apt install python3-tk', file=sys.stderr)
    return 1

  try:
    shell = Shell()
  except Exception as exc:
    print(f'Could not open a window: {exc}\n'
          f'Run with --check to see what is missing.', file=sys.stderr)
    return 1

  if args.phantom:
    shell.session.load_mesh(args.phantom)
  shell.run()
  return 0


if __name__ == '__main__':
  sys.exit(main())
