from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("feelmri")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Ensure basix (and its bundled libbasix) is loaded before feelmri extensions
try:
    import basix  # noqa: F401
except Exception:
    # If basix isn't installed, leave the error to the caller when they import features needing it
    pass