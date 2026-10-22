"""Build research-update.pdfpc (pdfpc presenter notes) from speaker-notes.json by frame title."""
import json, re, sys
tex = open("research-update.tex").read()
notes = {n["title"]: n["notes"] for n in json.load(open("speaker-notes.json"))["slides"]}
frames = re.findall(r"\\begin\{frame\}(\[.*?\])?(.*?)\\end\{frame\}", tex, re.S)
pages = []
for i, (_, body) in enumerate(frames):
    m = re.search(r"\\frametitle\{(.+?)\}", body)
    title = m.group(1) if m else "title"
    pages.append(dict(idx=i, label=str(i + 1), overlay=0, note=notes.get(title, "")))
meta = dict(pdfpcFormat=2, disableMarkdown=True, noteFontSize=16, pages=pages)
open("research-update.pdfpc", "w").write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
print(len(pages), "pages,", sum(1 for p in pages if p["note"]), "with notes")
