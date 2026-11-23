# Mijia Pocket Photo Printer 1S

The driver is now maintained in its [standalone GitHub repository](https://github.com/xauntian/mijia-pocket-printer-1s). This branch preserves the original ARTO extraction.

A Windows Bluetooth print client for the Xiaomi Mijia Pocket Photo Printer 1S (`hannto.printer.mint`). I built it for ARTO so a child could take home the picture the installation generated, with the art style and a short description printed on the same card. This repository extracts the printer implementation from the completed ARTO installation. [ARTO project](https://github.com/xauntian/children-art-explorer-mvp)

## Earlier projects this driver builds on

On 4 November 2026, we examined two open-source implementations while adapting the print protocol to the Mijia 1S. The links below point to the revisions used in that investigation.

| Earlier implementation | What we used it for |
| --- | --- |
| [fliptheweb/mi-portable-photo-printer · `85335b6`](https://github.com/fliptheweb/mi-portable-photo-printer/tree/85335b639f3d818c3fe9a892e328864fc3d37250) | The original pocket printer (`basil`). Its documented print sequence is the closest match to the 1S: create a job, send the JPEG with the device’s job ID, confirm it, then check the result. |
| [tuat-yate/xiaomi-photo-printer · `d2049ae`](https://github.com/tuat-yate/xiaomi-photo-printer/tree/d2049aeaa9408c205eae1ade780f1c23fa235c01) | The Pro model. We inspected its Python protocol, client and offline tests to compare the Bluetooth frame format and tried the Pro behavior against the 1S during development. |

The model differences affected the implementation. The original pocket printer uses up to 924 JPEG bytes per chunk on image channel 2. The Pro uses 988 on channel 4. The 1S client here uses 892 on channel 2. For ARTO, I implemented the Windows connection to the paired 1S, the printed card layout, and recovery of the original print job after a disconnect. Both earlier projects publish their code under the MIT license. [Upstream copyright and license notices](THIRD_PARTY_NOTICES.md)

## What I implemented

- **The 1S print sequence.** I adapted the single-copy workflow to the 1S, from checking readiness and obtaining a device job ID through JPEG transfer, confirmation and the printer’s final status. The Python client uses Bluetooth Classic RFCOMM on Windows with a paired printer.
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

## Image and type credits

- Artwork preparation, caption rendering and JPEG encoding use [Pillow](https://github.com/python-pillow/Pillow).
- Printed captions use Inter by Rasmus Andersson, distributed through Google Fonts. [Inter](https://github.com/rsms/inter) · [Google Fonts](https://github.com/google/fonts/tree/main/ofl/inter) · [Bundled font license](mijia1s/assets/OFL.txt)
