"""Offline rendering and single-job printing from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .layout import compose_print_image
from .printer import MijiaPrinterAdapter, _encode_jpeg, encode_mijia_image
from .records import PrintError, document_name_for
from .transport import MijiaProtocolError, MijiaSession


def add_captions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--title", help="Printed title, such as Fauvism")
    parser.add_argument("--caption", help="One short line printed under the title")


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--address", required=True, help="Bluetooth address of your paired 1S")


def add_receipts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipts", type=Path, default=Path.home() / ".mijia1s" / "receipts",
                        help="Directory for persistent print receipts")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds to wait for the print")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mijia1s", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="Create a printer-ready JPEG offline")
    render.add_argument("image", type=Path)
    render.add_argument("output", type=Path)
    add_captions(render)

    status = commands.add_parser("status", help="Read the paired printer's status")
    add_device(status)

    send = commands.add_parser("print", help="Send one photo and wait for the printer")
    send.add_argument("image", type=Path)
    add_device(send)
    add_receipts(send)
    add_captions(send)

    recover = commands.add_parser("recover", help="Check an existing print job by its source image")
    recover.add_argument("image", type=Path)
    add_device(recover)
    add_receipts(recover)
    args = parser.parse_args(argv)
    if hasattr(args, "title") and ((args.title is None) != (args.caption is None)):
        parser.error("--title and --caption must be supplied together")
    return args


def caption_snapshot(args: argparse.Namespace) -> dict[str, str] | None:
    if args.title is None:
        return None
    return {"print_name": args.title, "print_description": args.caption}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "render":
            if args.output.resolve() == args.image.resolve():
                raise ValueError("Choose an output path different from the source artwork.")
            if args.output.suffix.lower() not in {".jpg", ".jpeg"}:
                raise ValueError("Use a .jpg or .jpeg extension for the output.")
            style = caption_snapshot(args)
            data = (_encode_jpeg(compose_print_image(args.image, style)) if style is not None
                    else encode_mijia_image(args.image))
            with args.output.open("xb") as out:
                out.write(data)
            print(f"{args.output} ({len(data)} bytes)")
            return 0

        if args.command == "status":
            with MijiaSession(args.address) as session:
                status = session.rpc("mixed_status", {})
            if not isinstance(status, dict) or not isinstance(status.get("category"), str) or type(status.get("error")) is not int:
                raise ValueError("Printer returned an incomplete status.")
            print(json.dumps({"category": status["category"], "error": status["error"]}))
            return 0 if status["error"] == 0 else 1

        adapter = MijiaPrinterAdapter(args.address, receipts_dir=args.receipts)
        if args.command == "print":
            result = adapter.print_image(args.image, style_snapshot=caption_snapshot(args), timeout=args.timeout)
        else:
            document = document_name_for(args.image)
            record = json.loads((args.receipts / f"{document}.json").read_text(encoding="utf-8"))
            job_id = record.get("job_id") if isinstance(record, dict) else None
            if type(job_id) is not int or job_id <= 0:
                raise ValueError("The receipt has no device job ID. Check the printer and the physical output.")
            result = adapter.wait_for_job(job_id, expected_document=document, timeout=args.timeout)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 2 if result.needs_confirmation else 0
    except (PrintError, MijiaProtocolError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
