# Mijia Pocket Photo Printer 1S

A Windows Bluetooth print client for the Xiaomi Mijia Pocket Photo Printer 1S (`hannto.printer.mint`). I built it for ARTO so a child could take home the picture the installation generated, with the art style and a short description printed on the same card. This repository extracts the printer implementation from the completed ARTO installation. [ARTO project](https://github.com/xauntian/children-art-explorer-mvp)

## What I implemented

- **The 1S print sequence.** I mapped the exchange for a single print, from checking readiness and obtaining a device job ID through JPEG transfer, confirmation and the printer’s final status. The Python client uses Bluetooth Classic RFCOMM on Windows with a paired printer.
- **The physical card.** The layout combines the artwork, a style name and one line of text. It rotates the complete card into the printer’s 1040 × 1560 orientation and encodes a JPEG within 500 KiB. A print with captions also saves a lossless preview next to the input artwork.
- **Recovery after a disconnect.** A receipt on disk ties the source image to the printer and its job ID. The client records progress before sending each critical request. If a connection drops, recovery checks the existing job. Ambiguous device responses ask the operator to check the paper output. Concurrent submissions of the same source path share a lock.

## Run it

Python 3.11 or newer is required. Preview rendering and the test suite work without a printer. Live status and printing require Windows and a paired 1S.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m mijia1s render flower.png card.jpg --title "Post-Impressionism" --caption "Bold strokes give the petals texture."
```

Pair the printer in Windows Bluetooth settings and enter its address when prompted. Use the original artwork path for both the print and recovery commands.

```powershell
$printer = Read-Host "Paired printer Bluetooth address"
.\.venv\Scripts\python -m mijia1s status --address $printer
.\.venv\Scripts\python -m mijia1s print flower.png --address $printer --title "Post-Impressionism" --caption "Bold strokes give the petals texture."
```

To check the same print after an interruption, run the following.

```powershell
.\.venv\Scripts\python -m mijia1s recover flower.png --address $printer
```

Receipts default to `~/.mijia1s/receipts`. If you specify `--receipts`, use the same directory for recovery. Exit code 2 means the device result needs a physical check. If the connection dropped before the device returned a job ID, inspect the printer and the output. Keep the receipt so repeating the same command cannot consume another sheet accidentally.

## Device evidence and tests

Two receipts from the ARTO installation recorded one device-confirmed copy apiece on 23 November 2026. Elapsed time runs from local job creation to device confirmation.

| Artwork | Device job | Copies | Elapsed |
| --- | ---: | ---: | ---: |
| Flower, Post-Impressionism | 56 | 1 | 60.1 s |
| Houses | 57 | 1 | 54.5 s |

The standalone package was tested offline with rendered cards and a simulated printer transport. The hardware measurements above come from the integrated installation. Run the offline tests with this command.

```powershell
.\.venv\Scripts\python -m pytest -q
```

## References and credits

The 1S implementation comes from the ARTO installation. The following GitHub projects cover different models and were consulted when documenting the protocol family.

| Project | Relevance |
| --- | --- |
| [tuat-yate/xiaomi-photo-printer](https://github.com/tuat-yate/xiaomi-photo-printer) | An independent Python implementation for the Xiaomi Portable Photo Printer Pro. |
| [fliptheweb/mi-portable-photo-printer](https://github.com/fliptheweb/mi-portable-photo-printer) | A macOS implementation for the earlier Mi Portable Photo Printer (`hannto.printer.basil`). |
| [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | Image orientation, card composition, font rendering and JPEG encoding. |
| [rsms/inter](https://github.com/rsms/inter) · [Google Fonts distribution](https://github.com/google/fonts/tree/main/ofl/inter) | Inter typeface used for printed captions. The bundled font includes its [SIL Open Font License](mijia1s/assets/OFL.txt). |
