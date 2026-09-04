"""Classified loader errors for GarmentCodeData elements.

Each error type maps to the ``loader error`` failure class used by the batch
runner so a malformed element never aborts the rest of the batch.
"""


class LoaderError(Exception):
    """Base class for classified element loader errors."""


class MissingFileError(LoaderError):
    """A required dataset file is absent from the element directory."""


class InvalidElementError(LoaderError):
    """An element contains structurally inconsistent data."""


class SeamPairingError(LoaderError):
    """A stitch label cannot be paired into two boundary chains."""


class ScaleMismatchError(LoaderError):
    """The garment/body bounding boxes are grossly misaligned."""
