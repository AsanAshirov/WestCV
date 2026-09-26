"""Собирает docs/WestCV_plan_RU.pdf из docs/plan/WestCV_plan_RU.html.

Два прохода: первый печатает PDF с закладками, по закладкам находим страницы
разделов, вписываем их в оглавление и печатаем окончательный PDF.

usage: python docs/plan/build.py
Нужны: node + playwright (Chromium), pip install pypdf.
"""
import html
import re
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
SRC = HERE / "WestCV_plan_RU.html"
OUT = HERE.parent / "WestCV_plan_RU.pdf"


def npm_root() -> str:
    return subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True).stdout.strip()


def render(src: Path, dst: Path) -> None:
    env = {"NODE_PATH": npm_root(), "PATH": subprocess.os.environ["PATH"]}
    env.update({k: v for k, v in subprocess.os.environ.items() if k.startswith("PLAYWRIGHT")})
    subprocess.run(["node", str(HERE / "render_pdf.cjs"), str(src), str(dst)], check=True, env=env)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", html.unescape(s)).lower()


def heading_ids(text: str) -> dict[str, str]:
    """id -> нормализованный текст заголовка h2/h3."""
    out = {}
    for m in re.finditer(r'<h[23] id="([^"]+)"[^>]*>(.*?)</h[23]>', text, re.S):
        out[m.group(1)] = norm(re.sub(r"<[^>]+>", "", m.group(2)))
    return out


def outline_pages(pdf: Path) -> list[tuple[str, int]]:
    reader = PdfReader(str(pdf))
    found = []

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                found.append((norm(it.title), reader.get_destination_page_number(it) + 1))

    walk(reader.outline)
    return found


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    ids = heading_ids(text)
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(tmp) / "pass1.pdf"
        render(SRC, first)
        pages = outline_pages(first)

    # Chromium иногда дублирует текст заголовка в закладке, поэтому ищем по вхождению
    # текста без номера, двигаясь по закладкам в порядке документа.
    id_to_page, pos = {}, 0
    for hid, title in ids.items():
        core = re.sub(r"^[\d.]+", "", title)
        for j in range(pos, len(pages)):
            if core and core in pages[j][0]:
                id_to_page[hid], pos = pages[j][1], j + 1
                break
    missing = [i for i in ids if i not in id_to_page]
    if missing:
        print("WARNING: no bookmark for", missing)

    def fill(m):
        return f'{m.group(1)}{id_to_page.get(m.group(2), "")}</span>'

    filled = re.sub(r'(<span class="pg" data-toc="([^"]+)">)</span>', fill, text)
    tmp_src = HERE / ".build_toc.html"          # рядом с исходником, чтобы работали относительные пути к шрифтам
    tmp_src.write_text(filled, encoding="utf-8")
    try:
        render(tmp_src, OUT)
    finally:
        tmp_src.unlink()
    print("OK:", OUT, len(PdfReader(str(OUT)).pages), "pages")


if __name__ == "__main__":
    main()
