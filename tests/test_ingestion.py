import io
import pytest
from unittest.mock import patch, MagicMock
from llama_index.core import Document


def make_mock_pdf_reader(pages_text):
    mock_reader = MagicMock()
    mock_pages = []
    for text in pages_text:
        page = MagicMock()
        page.extract_text.return_value = text
        mock_pages.append(page)
    mock_reader.pages = mock_pages
    return mock_reader


class TestLoadPdf:
    def test_returns_one_document_per_page(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Page one content", "Page two content"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].text == "Page one content"
        assert docs[1].text == "Page two content"

    def test_skips_empty_pages(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Real content", "   ", ""])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert len(docs) == 1
        assert docs[0].text == "Real content"

    def test_metadata_includes_source_and_page(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["Content"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert docs[0].metadata["source"] == "pdf"
        assert docs[0].metadata["page"] == 1

    def test_skips_none_text_pages(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader([None, "Real content"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert len(docs) == 1
        assert docs[0].metadata["page"] == 2

    def test_page_metadata_is_original_page_number(self):
        from ingestion import load_pdf
        mock_reader = make_mock_pdf_reader(["First", "   ", "Third"])
        with patch("ingestion.PdfReader", return_value=mock_reader):
            docs = load_pdf(b"fake-pdf-bytes")
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 3  # original index preserved, not re-sequenced


class TestLoadYoutube:
    def test_parses_standard_url(self):
        from ingestion import _parse_video_id
        assert _parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_parses_short_url(self):
        from ingestion import _parse_video_id
        assert _parse_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_raises_on_invalid_url(self):
        from ingestion import _parse_video_id
        with pytest.raises(ValueError, match="Could not parse video ID"):
            _parse_video_id("https://example.com/video")

    def test_returns_single_document_with_full_transcript(self):
        from ingestion import load_youtube
        snippet1 = MagicMock(); snippet1.text = "Hello world"
        snippet2 = MagicMock(); snippet2.text = "This is a test"
        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value = [snippet1, snippet2]
        mock_list = MagicMock()
        mock_list.find_transcript.return_value = mock_transcript
        mock_api = MagicMock()
        mock_api.list.return_value = mock_list
        with patch("ingestion.YouTubeTranscriptApi", return_value=mock_api):
            docs = load_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert "This is a test" in docs[0].text
        assert docs[0].metadata["source"] == "youtube"
        assert docs[0].metadata["lang"] == "en"

    def test_falls_back_to_first_available_when_english_missing(self):
        from ingestion import load_youtube
        from youtube_transcript_api._errors import NoTranscriptFound

        vi_snippet = MagicMock(); vi_snippet.text = "Xin chào"
        vi_transcript = MagicMock()
        vi_transcript.language_code = "vi"
        vi_transcript.fetch.return_value = [vi_snippet]
        mock_list = MagicMock()
        mock_list.find_transcript.side_effect = NoTranscriptFound("vid", ["en"], mock_list)
        mock_list.__iter__.return_value = iter([vi_transcript])
        mock_api = MagicMock()
        mock_api.list.return_value = mock_list
        with patch("ingestion.YouTubeTranscriptApi", return_value=mock_api):
            docs = load_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert docs[0].text == "Xin chào"
        assert docs[0].metadata["lang"] == "vi"


class TestLoadGithub:
    def test_parses_github_url(self):
        from ingestion import _parse_github_owner_repo
        owner, repo = _parse_github_owner_repo("https://github.com/openai/openai-python")
        assert owner == "openai"
        assert repo == "openai-python"

    def test_raises_on_invalid_github_url(self):
        from ingestion import _parse_github_owner_repo
        with pytest.raises(ValueError, match="Could not parse owner/repo"):
            _parse_github_owner_repo("https://gitlab.com/user/repo")

    def test_fetches_and_returns_documents(self):
        from ingestion import load_github

        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.raise_for_status = MagicMock()
        tree_response.json.return_value = {
            "tree": [
                {"type": "blob", "path": "README.md"},
                {"type": "blob", "path": "main.py"},
                {"type": "blob", "path": "image.png"},  # should be excluded
                {"type": "tree", "path": "src"},         # should be excluded
            ]
        }

        file_response = MagicMock()
        file_response.status_code = 200
        file_response.text = "file content"

        def mock_get(url, **kwargs):
            if "git/trees" in url:
                return tree_response
            return file_response

        with patch("ingestion.requests.get", side_effect=mock_get):
            docs = load_github("https://github.com/owner/repo")

        assert len(docs) == 2
        paths = [d.metadata["file"] for d in docs]
        assert "README.md" in paths
        assert "main.py" in paths

    def test_skips_files_with_empty_content(self):
        from ingestion import load_github

        tree_response = MagicMock()
        tree_response.status_code = 200
        tree_response.raise_for_status = MagicMock()
        tree_response.json.return_value = {"tree": [{"type": "blob", "path": "README.md"}]}

        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.text = "   "

        with patch("ingestion.requests.get", side_effect=lambda url, **kw: tree_response if "git/trees" in url else empty_response):
            docs = load_github("https://github.com/owner/repo")

        assert len(docs) == 0


class TestIngest:
    def test_combines_all_sources(self):
        from ingestion import ingest
        pdf_doc = Document(text="pdf content", metadata={"source": "pdf", "page": 1})
        yt_doc = Document(text="youtube content", metadata={"source": "youtube", "url": "u"})
        gh_doc = Document(text="github content", metadata={"source": "github", "file": "f"})

        with patch("ingestion.load_pdf", return_value=[pdf_doc]) as mock_pdf, \
             patch("ingestion.load_youtube", return_value=[yt_doc]) as mock_yt, \
             patch("ingestion.load_github", return_value=[gh_doc]) as mock_gh:
            docs = ingest(pdf_bytes=b"x", youtube_url="url", github_url="ghurl")

        assert len(docs) == 3
        mock_pdf.assert_called_once_with(b"x")
        mock_yt.assert_called_once_with("url")
        mock_gh.assert_called_once_with("ghurl")

    def test_skips_none_sources(self):
        from ingestion import ingest
        pdf_doc = Document(text="pdf content", metadata={"source": "pdf", "page": 1})
        with patch("ingestion.load_pdf", return_value=[pdf_doc]) as mock_pdf, \
             patch("ingestion.load_youtube") as mock_yt, \
             patch("ingestion.load_github") as mock_gh:
            docs = ingest(pdf_bytes=b"x")
        assert len(docs) == 1
        mock_yt.assert_not_called()
        mock_gh.assert_not_called()
