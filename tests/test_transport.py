"""Wire fixtures and offline socket tests; never create a Bluetooth socket."""

from __future__ import annotations

import json
import struct
from collections import deque

import pytest

from mijia1s import transport
from mijia1s.transport import MijiaProtocolError, MijiaSession, file_frames, frame

ADDRESS = "02:00:00:00:00:01"
STATUS_FIXTURE = bytes.fromhex(
    "7e64000106030000000001000000010001002c00"
    "7b226964223a312c226d6574686f64223a226d697865645f737461747573222c"
    "22706172616d73223a7b7d7dc07e"
)


class FakeSocket:
    def __init__(self, incoming=(), *, connect_error=None, send_error=None):
        self.incoming = deque(incoming)
        self.connect_error = connect_error
        self.send_error = send_error
        self.sent = []
        self.send_timeouts = []
        self.timeouts = []
        self.options = []
        self.closed = 0
        self.connected_to = None

    def setsockopt(self, *args):
        self.options.append(args)

    def settimeout(self, value):
        self.timeouts.append(value)

    def connect(self, target):
        self.connected_to = target
        if self.connect_error:
            raise self.connect_error

    def sendall(self, data):
        self.sent.append(bytes(data))
        self.send_timeouts.append(self.timeouts[-1])
        if self.send_error:
            raise self.send_error

    def recv(self, _size):
        if not self.incoming:
            raise TimeoutError()
        item = self.incoming.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed += 1


def session_for(fake, **kwargs):
    def factory(*_args):
        return fake

    return MijiaSession(ADDRESS, socket_factory=factory, **kwargs)


def reply(payload, *, version=1, total=1, number=1):
    if isinstance(payload, dict):
        payload = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    # Independent reply builder: response interaction=7, sender version=0x01.
    header = struct.pack("<6BIIHHH", 0x7E, version, 0, 1, 7, 3,
                         0, number, total, number,
                         len(payload) | (0x2000 if total > 1 else 0))
    packet = header + payload
    return packet + bytes((sum(packet[1:]) & 255, 0x7E))


def test_status_frame_matches_official_plain_wire_fixture():
    body = b'{"id":1,"method":"mixed_status","params":{}}'
    assert frame(body) == STATUS_FIXTURE


def test_single_file_frame_matches_wire_fixture():
    expected = bytes.fromhex(
        "7e6400020602000000000100000001000100072004030201616263c87e"
    )
    assert list(file_frames(b"abc", 0x01020304)) == [expected]


def test_multiple_file_frames_reassemble_without_padding():
    data = bytes(range(256)) * 7 + b"tail"
    packets = list(file_frames(data, 0x01020304))
    assert len(packets) == 3
    assert b"".join(packet[24:-2] for packet in packets) == data
    for number, packet in enumerate(packets, 1):
        assert packet[:6] == bytes.fromhex("7e6400020602")
        arc, sequence, total, index, attr = struct.unpack_from("<IIHHH", packet, 6)
        assert (arc, sequence, total, index) == (0, number, 3, number)
        assert attr & 0x2000
        assert attr & 1023 == len(packet) - 22
        assert packet[20:24] == b"\x04\x03\x02\x01"
        assert packet[-2] == sum(packet[1:-2]) & 255
        assert len(packet) <= 918


@pytest.mark.parametrize("version", [1, 0x64])
def test_rpc_reads_fragmented_reply_and_restores_timeout(version):
    response = reply({"id": 1, "result": {"category": "idle"}}, version=version)
    fake = FakeSocket([response[:1], response[1:19], response[19:23], response[23:]])
    with session_for(fake, connect_timeout=4, io_timeout=10) as session:
        assert session.rpc("mixed_status", {}) == {"category": "idle"}
        assert fake.timeouts[-1] == 10
    assert fake.sent == [STATUS_FIXTURE]
    assert fake.connected_to == (ADDRESS, 1)
    assert fake.closed == 1


def test_rpc_retains_coalesced_frames_and_ignores_unmatched_id():
    combined = (reply({"id": 99, "result": "unrelated"})
                + reply({"id": 1, "result": "first"})
                + reply({"id": 2, "result": "second"}))
    fake = FakeSocket([combined])
    with session_for(fake) as session:
        assert session.rpc("mixed_status", {}) == "first"
        assert session.rpc("job_info", [51]) == "second"
    assert len(fake.sent) == 2
    assert fake.send_timeouts == [10, 10]


def test_bad_checksum_is_discarded_before_valid_reply():
    broken = bytearray(reply({"id": 1, "result": "bad"}))
    broken[-2] ^= 1
    fake = FakeSocket([b"noise" + bytes(broken) + reply({"id": 1, "result": "good"})])
    with session_for(fake) as session:
        assert session.rpc("mixed_status", {}) == "good"


def test_json_parts_reassemble_by_number_across_utf8_boundary():
    payload = '{"id":1,"result":"打印机"}'.encode()
    split = payload.index("打".encode()) + 1
    fake = FakeSocket([reply(payload[split:], total=2, number=2)
                       + reply(payload[:split], total=2, number=1)])
    with session_for(fake) as session:
        assert session.rpc("mixed_status", {}) == "打印机"


def test_oversized_fragmented_json_is_bounded():
    incoming = [reply(b"x" * 1023, total=65, number=number) for number in range(1, 66)]
    fake = FakeSocket(incoming)
    with session_for(fake) as session, pytest.raises(MijiaProtocolError, match="size limit"):
        session.rpc("mixed_status", {})


def test_timeout_never_resends_print_job_and_uses_total_deadline(monkeypatch):
    ticks = iter(number / 100 for number in range(100))
    monkeypatch.setattr(transport.time, "monotonic", lambda: next(ticks))
    fake = FakeSocket()
    with session_for(fake, io_timeout=0.05) as session:
        with pytest.raises(TimeoutError):
            session.rpc("print_job", {"copies": 1})
        assert fake.timeouts[-1] == 0.05
    assert len(fake.sent) == 1
    assert json.loads(fake.sent[0][20:-2])["method"] == "print_job"
    assert fake.closed == 1


def test_rpc_rejection_retains_error_and_does_not_resend():
    error = {"code": -1, "message": "rejected"}
    fake = FakeSocket([reply({"id": 1, "error": error})])
    with session_for(fake) as session:
        with pytest.raises(MijiaProtocolError) as failure:
            session.rpc("print_job", {"copies": 1})
        assert failure.value.error == error
    assert len(fake.sent) == 1


def test_unsupported_rpc_sends_nothing():
    fake = FakeSocket()
    with session_for(fake) as session:
        with pytest.raises(ValueError, match="Unsupported"):
            session.rpc("reset", {})
    assert not fake.sent


def test_image_send_restores_timeout_and_reports_sent_chunks():
    fake = FakeSocket([reply({"id": 1, "result": {}})])
    progress = []
    data = b"a" * 893
    with session_for(fake) as session:
        session.rpc("mixed_status", {})
        session.send_image(data, 51, lambda sent, total: progress.append((sent, total)))
    assert fake.sent[1:] == list(file_frames(data, 51))
    assert fake.send_timeouts == [10, 10, 10]
    assert progress == [(1, 2), (2, 2)]


def test_context_entry_sends_no_application_bytes_and_close_is_idempotent():
    fake = FakeSocket()
    session = session_for(fake)
    assert fake.connected_to is None
    with session:
        assert fake.sent == []
        assert fake.options == [
            (transport.SOL_RFCOMM, transport.SO_BTH_AUTHENTICATE, b"\x01\x00\x00\x00"),
            (transport.SOL_RFCOMM, transport.SO_BTH_ENCRYPT, b"\x01\x00\x00\x00"),
        ]
    session.close()
    assert fake.closed == 1


def test_failed_connect_closes_socket():
    fake = FakeSocket(connect_error=OSError("unreachable"))
    with pytest.raises(OSError, match="unreachable"), session_for(fake):
        pass
    assert fake.closed == 1


def test_closed_connection_during_reply_is_not_retried():
    fake = FakeSocket([b""])
    with session_for(fake) as session, pytest.raises(ConnectionError):
        session.rpc("print_job", {"copies": 1})
    assert len(fake.sent) == 1
    assert fake.closed == 1


@pytest.mark.parametrize("data,job_id", [(b"", 51), (b"abc", 0), (b"abc", True)])
def test_invalid_image_is_rejected_before_any_write(data, job_id):
    fake = FakeSocket()
    with session_for(fake) as session, pytest.raises(ValueError):
        session.send_image(data, job_id)
    assert not fake.sent
