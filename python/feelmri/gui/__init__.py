"""Graphical interface for feelmri.

Nothing heavy is imported here. The 3D viewer needs PyVista and VTK, which are
an optional extra (`pip install feelmri[gui]`); the shell needs tkinter, which
is in the standard library but is a separate package on some Linux builds. Both
are imported lazily by the modules that use them, so `import feelmri.gui` works
on a headless machine with neither installed.

The `model` subpackage carries no display dependency at all and is the part
worth importing from a script.
"""

__all__ = ["model"]

