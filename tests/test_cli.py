from PIL import Image
import pytest

from mijia1s.cli import main
from mijia1s.records import document_lock, document_name_for, PrintError
from mijia1s.printer import MijiaPrinterAdapter


def test_render_is_offline_and_preserves_source(tmp_path):
    source = tmp_path / "flower.png"
    output = tmp_path / "card.jpg"
    Image.new("RGB", (180, 100), "orange").save(source)
    before = source.read_bytes()
    assert main(["render", str(source), str(output), "--title", "Fauvism",
                 "--caption", "Bold colours bring the flowers to life."]) == 0
    with Image.open(output) as card:
        assert card.format == "JPEG" and card.size == (1040, 1560)
    assert output.stat().st_size <= 500 * 1024 and source.read_bytes() == before
    assert main(["render", str(source), str(output)]) == 1


def test_render_requires_both_caption_fields():
    with pytest.raises(SystemExit):
        main(["render", "a.png", "b.jpg", "--title", "Fauvism"])


def test_concurrent_call_never_opens_printer(tmp_path):
    name = tmp_path / "art.png"
    Image.new("RGB", (12, 12)).save(name)
    directory = tmp_path / "receipts"
    adapter = MijiaPrinterAdapter("02:00:00:00:00:01", receipts_dir=directory,
                                 session_factory=lambda _: pytest.fail("Opened the printer"))
    with document_lock(directory / (document_name_for(name) + ".json")):
        with pytest.raises(PrintError, match="Another process"):
            adapter.print_image(name)
