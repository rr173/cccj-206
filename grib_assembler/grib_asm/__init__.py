"""Offline GRIB2 collection assembler.

The package only reads GRIB2 section headers and copies *encoded* messages
byte-for-byte.  It never decodes grid point values, never interpolates and
never re-encodes.
"""

__version__ = "1.0.0"
