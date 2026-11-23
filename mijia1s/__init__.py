"""Bluetooth photo printing for the Mijia Portable Photo Printer 1S."""

from .printer import MijiaPrinterAdapter
from .records import PrintError, PrintResult

__all__ = ["MijiaPrinterAdapter", "PrintError", "PrintResult"]
