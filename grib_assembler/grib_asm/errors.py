"""Typed errors for the assembler."""


class GribAsmError(Exception):
    """Base error."""


class ScannerError(GribAsmError):
    """Raised for structural problems found while scanning a GRIB2 file."""


class BadFileError(GribAsmError):
    """A file cannot participate in assembly.

    ``reason`` is a stable machine code; ``evidence`` carries factual details
    (offsets, expected/actual values) that are retained in the catalog.
    """

    def __init__(self, reason, evidence=None, message=None):
        self.reason = reason
        self.evidence = evidence or {}
        text = message or reason
        super().__init__(text)


class UnsupportedTemplate(GribAsmError):
    """A complete, well-formed message uses a template this profile does not map."""

    def __init__(self, reason, evidence=None):
        self.reason = reason
        self.evidence = evidence or {}
        super().__init__(reason)


class BlockedError(GribAsmError):
    """No complete cycle is available; no success package may be produced."""

    def __init__(self, report):
        self.report = report
        super().__init__("blocked: no complete forecast cycle available")
