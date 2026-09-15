"""Entry point for `python -m feelmri.gui`.

The implementation lives in `app.__main__` alongside the shell it launches;
this is the name the `-m` switch looks for.
"""
import sys

from .app.__main__ import main

if __name__ == '__main__':
  sys.exit(main())
