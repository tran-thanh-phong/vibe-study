import io
import re
import requests
from pypdf import PdfReader
from llama_index.core import Document
from youtube_transcript_api import YouTubeTranscriptApi


def load_pdf(file_bytes: bytes) -> list[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(text=text, metadata={"source": "pdf", "page": i + 1}))
    return docs


def _parse_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not parse video ID from URL: {url}")
    return match.group(1)


def load_youtube(url: str) -> list[Document]:
    video_id = _parse_video_id(url)
    api = YouTubeTranscriptApi()
    transcripts = api.list(video_id)
    try:
        transcript = transcripts.find_transcript(["en"])
    except Exception:
        transcript = next(iter(transcripts))
    fetched = transcript.fetch()
    text = " ".join(snippet.text for snippet in fetched)
    return [Document(text=text, metadata={"source": "youtube", "url": url, "lang": transcript.language_code})]


def _parse_github_owner_repo(url: str) -> tuple[str, str]:
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not match:
        raise ValueError(f"Could not parse owner/repo from URL: {url}")
    return match.group(1), match.group(2)


_ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".md", ".java", ".go"}
_MAX_GITHUB_FILES = 30


def load_github(url: str) -> list[Document]:
    owner, repo = _parse_github_owner_repo(url)
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    resp = requests.get(tree_url, headers={"Accept": "application/vnd.github.v3+json"})
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    docs = []
    for item in tree:
        if len(docs) >= _MAX_GITHUB_FILES:
            break
        if item["type"] != "blob":
            continue
        path = item["path"]
        ext = f".{path.rsplit('.', 1)[-1]}" if "." in path else ""
        if path != "README.md" and ext not in _ALLOWED_EXTENSIONS:
            continue
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        content_resp = requests.get(raw_url)
        if content_resp.ok and content_resp.text.strip():
            docs.append(Document(text=content_resp.text, metadata={"source": "github", "file": path}))
    return docs


def ingest(
    pdf_bytes: bytes | None = None,
    youtube_url: str | None = None,
    github_url: str | None = None,
) -> list[Document]:
    docs = []
    if pdf_bytes:
        docs.extend(load_pdf(pdf_bytes))
    if youtube_url:
        docs.extend(load_youtube(youtube_url))
    if github_url:
        docs.extend(load_github(github_url))
    return docs
