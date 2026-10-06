"""Unit tests for the pure helpers in common.py."""

import hashlib
import json
from pathlib import Path

import pytest
import requests

from common import (
    atomic_write_jsonl,
    cdx_timestamp_to_iso,
    chunk_words,
    get_with_retries,
    is_indexed_warc_crawl,
    is_successful_html_capture,
    make_chunk_id,
    make_page_id,
    make_record_id,
    make_text_hash,
    normalize_date,
    parse_record_header,
    reciprocal_rank_fusion,
    repair_partial_jsonl,
    to_json_line,
)

# --- IDs -------------------------------------------------------------------


def test_record_id_is_sha1_of_concatenated_fields() -> None:
    expected_id = hashlib.sha1(b"CC-MAIN-2016-07http://www.eurosport.fr/2016-02-06T10:00:00Z")
    record_id = make_record_id(
        "CC-MAIN-2016-07", "http://www.eurosport.fr/", "2016-02-06T10:00:00Z"
    )
    assert record_id == expected_id.hexdigest()


def test_ids_are_deterministic_and_distinct() -> None:
    text_hash = make_text_hash("Wawrinka bat Djokovic en finale.")
    assert text_hash == make_text_hash("Wawrinka bat Djokovic en finale.")
    page_id = make_page_id("http://www.eurosport.fr/tennis/", text_hash)
    assert page_id == make_page_id("http://www.eurosport.fr/tennis/", text_hash)
    assert page_id != make_page_id("http://www.eurosport.fr/foot/", text_hash)
    assert make_chunk_id(page_id, 0) != make_chunk_id(page_id, 1)
    assert len(page_id) == 40


# --- CDX filtering -----------------------------------------------------------


@pytest.mark.parametrize(
    ("cdx_row", "expected"),
    [
        ({"status": "200", "mime": "text/html"}, True),
        ({"status": "200", "mime": "unk", "mime-detected": "application/xhtml+xml"}, True),
        ({"status": "200", "mime": "text/html", "mime-detected": "application/pdf"}, False),
        ({"status": "200", "mime": "text/html; charset=utf-8"}, True),
        ({"status": "301", "mime": "text/html"}, False),
        ({"status": "200", "mime": "image/jpeg"}, False),
        ({"status": "200"}, False),
    ],
)
def test_is_successful_html_capture(cdx_row: dict, expected: bool) -> None:
    assert is_successful_html_capture(cdx_row) is expected


def test_crawl_id_filter_excludes_arc_archives() -> None:
    assert is_indexed_warc_crawl("CC-MAIN-2016-07")
    assert not is_indexed_warc_crawl("CC-MAIN-2012")
    assert not is_indexed_warc_crawl("CC-MAIN-2009-2010")


def test_cdx_timestamp_to_iso() -> None:
    assert cdx_timestamp_to_iso("20150607154512") == "2015-06-07T15:45:12Z"


def test_normalize_date() -> None:
    assert normalize_date("2015-06-07") == "2015-06-07"
    assert normalize_date("2015-06-07T10:00:00") == "2015-06-07"
    assert normalize_date("2015-13-45") is None
    assert normalize_date(None) is None


# --- Raw record files --------------------------------------------------------


def test_parse_record_header_skips_html_even_with_tricky_content() -> None:
    record = {"record_id": "abc", "url": 'http://x.fr/?q=", "html": ', "html": '<p>"html": x</p>'}
    header = parse_record_header(to_json_line(record))
    assert header == {"record_id": "abc", "url": 'http://x.fr/?q=", "html": '}


def test_repair_partial_jsonl_drops_incomplete_last_line(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard.jsonl.tmp"
    shard_path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "html": "trunc', encoding="utf-8")
    repair_partial_jsonl(shard_path)
    assert shard_path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n'


def test_atomic_write_jsonl_leaves_no_temporary_file(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    assert atomic_write_jsonl(output_path, [{"b": 1}, {"a": "é"}]) == 2
    assert [json.loads(line) for line in output_path.read_text().splitlines()] == [
        {"b": 1},
        {"a": "é"},
    ]
    assert not (tmp_path / "out.jsonl.tmp").exists()


# --- HTTP retries -------------------------------------------------------------


def test_get_with_retries_retries_invalid_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = []
    for body in ('{"value": "truncated', '{"value": "complete"}'):
        response = requests.Response()
        response.status_code = 200
        response._content = body.encode()
        responses.append(response)

    class StubSession:
        def get(self, *args: object, **kwargs: object) -> requests.Response:
            return responses.pop(0)

    monkeypatch.setattr("common.time.sleep", lambda _: None)
    response = get_with_retries(
        StubSession(),  # type: ignore[arg-type]
        "https://example.test",
        max_attempts=2,
        response_validator=lambda item: (
            None if item.text.endswith('complete"}') else "malformed JSON"
        ),
    )

    assert response.json() == {"value": "complete"}
    assert responses == []


# --- Chunking ------------------------------------------------------------------


def test_chunk_words_overlap_and_coverage() -> None:
    words = [f"w{index}" for index in range(450)]
    chunk_texts = chunk_words(" ".join(words), title=None, chunk_size_words=200, overlap_words=40)
    chunk_word_lists = [chunk_text.split() for chunk_text in chunk_texts]
    assert [len(chunk) for chunk in chunk_word_lists] == [200, 200, 130]
    assert chunk_word_lists[0][-40:] == chunk_word_lists[1][:40]
    assert chunk_word_lists[-1][-1] == "w449"


def test_chunk_words_prepends_title_and_handles_short_text() -> None:
    assert chunk_words("court texte", title="Titre") == ["Titre\n\ncourt texte"]
    assert len(chunk_words(" ".join(["mot"] * 200), title=None)) == 1


def test_chunk_words_rejects_overlap_larger_than_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_words("a b c", title=None, chunk_size_words=10, overlap_words=10)


# --- Reciprocal Rank Fusion ------------------------------------------------------


def test_reciprocal_rank_fusion_scores_and_sources() -> None:
    fused = reciprocal_rank_fusion({"keyword": ["a", "b"], "semantic": ["b", "c"]})
    fused_by_id = {item_id: (score, sources) for item_id, score, sources in fused}
    assert fused[0][0] == "b"
    assert fused_by_id["b"][0] == pytest.approx(1 / 62 + 1 / 61)
    assert fused_by_id["b"][1] == ["keyword", "semantic"]
    assert fused_by_id["a"][0] == pytest.approx(1 / 61)
    assert fused_by_id["c"][1] == ["semantic"]


def test_reciprocal_rank_fusion_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion({"keyword": ["z"], "semantic": ["y"]})
    assert [item_id for item_id, _, _ in fused] == ["y", "z"]
