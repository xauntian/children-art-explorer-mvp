"""Bounded, single-request SPP transport for the Mijia Pocket Printer 1S.

Importing this module does not open Bluetooth. A session opens only on entering
its context; failed requests are never retried. The caller owns job creation,
confirmation, and protection against submitting the same print twice.
"""

from __future__ import annotations

import json
import math
import re
import socket
import struct
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

SOL_RFCOMM = 3
SO_BTH_AUTHENTICATE = -2147483647
SO_BTH_ENCRYPT = 2

CHUNK_SIZE = 892
MAX_JSON_BYTES = 65536
RPC_METHODS = frozenset({"mixed_status", "job_info", "print_job", "confirm_job"})


class MijiaProtocolError(RuntimeError):
    """An invalid response or an explicit printer rejection."""

    def __init__(self, message: str, *, error: Any = None):
        super().__init__(message)
        self.error = error


def frame(
    payload: bytes,
    sequence: int = 1,
    channel: int = 1,
    encoding: int = 3,
    total: int = 1,
    number: int = 1,
    file_transfer: bool = False,
) -> bytes:
    """Build a 1S application frame for the paired RFCOMM connection."""
    payload = bytes(payload)
    if not (len(payload) < 1024 and 1 <= number <= total <= 32767):
        raise ValueError("Invalid frame dimensions")
    if not (0 <= sequence <= 0xFFFFFFFF and 0 <= channel <= 255 and 0 <= encoding <= 255):
        raise ValueError("Invalid frame header")
    header = bytearray([0x7E, 0x64, 0, channel, 6, encoding])
    header.extend(struct.pack("<IIHHH", 0, sequence, total, number,
                              len(payload) | (0x2000 if file_transfer else 0)))
    packet = header + payload
    return bytes(packet + bytes([sum(packet[1:]) & 255, 0x7E]))


def file_frames(data: bytes, job_id: int) -> Iterator[bytes]:
    """Yield 892-byte JPEG chunks, each prefixed with its LE32 job ID."""
    if type(job_id) is not int or not (1 <= job_id <= 0xFFFFFFFF):
        raise ValueError("Invalid printer job ID")
    count = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    if not (1 <= count <= 32767):
        raise ValueError("Image is empty or exceeds the frame count limit")
    for index in range(count):
        payload = struct.pack("<I", job_id) + data[index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE]
        yield frame(payload, index + 1, 2, 2, count, index + 1, True)


class MijiaSession:
    """One synchronous RFCOMM channel-1 connection; use from one thread.

    ``socket_factory`` accepts the same three arguments as ``socket.socket`` and
    permits completely offline tests. ``progress`` receives (sent_chunks, total).
    No pairing changes or application commands are sent on context entry.
    """

    def __init__(
        self,
        address: str,
        connect_timeout: float = 10,
        io_timeout: float = 10,
        *,
        socket_factory: Callable[..., Any] | None = None,
    ):
        if not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", address):
            raise ValueError("Invalid Bluetooth MAC address")
        if not all(math.isfinite(value) and value > 0 for value in (connect_timeout, io_timeout)):
            raise ValueError("Timeouts must be finite and positive")
        self.address = address.upper()
        self.connect_timeout = connect_timeout
        self.io_timeout = io_timeout
        self._socket_factory = socket_factory
        self._socket: Any = None
        self._buffer = bytearray()
        self._request_id = 1

    def __enter__(self) -> MijiaSession:
        if self._socket is not None:
            raise RuntimeError("Session is already connected")
        if self._socket_factory is None and sys.platform != "win32":
            raise OSError("Live Bluetooth printing requires Windows. Offline layout works on other platforms.")
        factory = self._socket_factory or socket.socket
        self._socket = factory(getattr(socket, "AF_BLUETOOTH", 32), socket.SOCK_STREAM,
                               getattr(socket, "BTPROTO_RFCOMM", 3))
        try:
            # Let Windows authenticate and encrypt the paired printer link.
            for option in (SO_BTH_AUTHENTICATE, SO_BTH_ENCRYPT):
                self._socket.setsockopt(SOL_RFCOMM, option, struct.pack("<I", 1))
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.address, 1))
            self._socket.settimeout(self.io_timeout)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        link, self._socket = self._socket, None
        self._buffer.clear()
        if link is not None:
            link.close()

    def _link(self) -> Any:
        if self._socket is None:
            raise RuntimeError("Session is not connected")
        return self._socket

    def _response(self, wanted_id: int) -> dict[str, Any]:
        link = self._link()
        deadline = time.monotonic() + self.io_timeout
        fragments: dict[tuple[int, int, int], dict[int, bytes]] = {}
        fragment_bytes = 0
        try:
            while time.monotonic() < deadline:
                while len(self._buffer) >= 22:
                    if self._buffer[0] != 0x7E or self._buffer[1] not in (1, 0x64):
                        del self._buffer[0]
                        continue
                    attr = int.from_bytes(self._buffer[18:20], "little")
                    length = attr & 1023
                    if len(self._buffer) < length + 22:
                        break
                    raw = bytes(self._buffer[:length + 22])
                    if raw[-1] != 0x7E or raw[-2] != sum(raw[1:-2]) & 255:
                        del self._buffer[0]
                        continue
                    del self._buffer[:length + 22]
                    if raw[3] != 1 or raw[5] != 3 or (attr >> 10) & 7:
                        continue
                    total, number = struct.unpack_from("<HH", raw, 14)
                    if not (1 <= number <= total <= 32767):
                        continue
                    payload = raw[20:-2]
                    if total > 1:
                        # Sequence increments per part; package number defines order.
                        key = (raw[1], int.from_bytes(raw[6:10], "little"), total)
                        parts = fragments.setdefault(key, {})
                        fragment_bytes += len(payload) - len(parts.get(number, b""))
                        if fragment_bytes > MAX_JSON_BYTES:
                            raise MijiaProtocolError("Response exceeds JSON size limit")
                        parts[number] = payload
                        if len(parts) != total:
                            continue
                        payload = b"".join(parts[index] for index in range(1, total + 1))
                        fragment_bytes -= len(payload)
                        del fragments[key]
                    try:
                        value = json.loads(payload)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(value, dict) and value.get("id") == wanted_id:
                        return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                link.settimeout(min(0.5, remaining))
                try:
                    block = link.recv(4096)
                except TimeoutError:
                    continue
                if not block:
                    raise ConnectionError("Printer closed the data channel")
                self._buffer.extend(block)
                if len(self._buffer) > MAX_JSON_BYTES:
                    raise MijiaProtocolError("Response buffer exceeds size limit")
            raise TimeoutError("No matching printer reply before the deadline")
        finally:
            link.settimeout(self.io_timeout)

    def rpc(self, method: str, params: Any) -> Any:
        """Send one whitelisted JSON request; return its result without retries."""
        if method not in RPC_METHODS:
            raise ValueError("Unsupported printer RPC method")
        request_id = self._request_id
        self._request_id += 1
        body = json.dumps({"id": request_id, "method": method, "params": params},
                          separators=(",", ":")).encode("utf-8")
        packet = frame(body)
        link = self._link()
        link.settimeout(self.io_timeout)
        link.sendall(packet)
        value = self._response(request_id)
        if "error" in value:
            raise MijiaProtocolError("Printer rejected RPC request", error=value["error"])
        return value.get("result")

    def send_image(
        self, data: bytes, job_id: int, progress: Callable[[int, int], None] | None = None
    ) -> None:
        """Send JPEG bytes once; this method does not create or confirm a job."""
        # Validate before the first write, including the last frame's dimensions.
        data = bytes(data)
        packets = file_frames(data, job_id)
        first = next(packets)
        link = self._link()
        link.settimeout(self.io_timeout)
        total = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        link.sendall(first)
        if progress is not None:
            progress(1, total)
        for sent, packet in enumerate(packets, 2):
            link.sendall(packet)
            if progress is not None:
                progress(sent, total)
