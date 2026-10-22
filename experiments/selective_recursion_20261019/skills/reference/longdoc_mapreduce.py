"""Map-reduce reading for multiple-choice questions over a long document (generic reference library, E32).

answer_longdoc(request) -> str    # brief relevant notes from every part of the document (helper calls in parallel), then one choice over the notes; "ANSWER: <letter>"
take_notes(body, qblock, chunk=24000) -> list    # notes per part, in document order; parts with nothing relevant are dropped
Typical use: FINAL_TEXT(answer_longdoc(request)).
"""
import re


def split_request(text):
    body, rest = text.rsplit("\n\nQUESTION:", 1)
    return body, "QUESTION:" + rest


def take_notes(body, qblock, chunk=24000):
    parts = [body[i:i + chunk] for i in range(0, len(body), chunk)] or [""]
    sys_n = ("You read one part of a long document to help answer a multiple-choice question. Quote or summarize "
             "only the facts in this part that bear on the question or on any of the options; if nothing in it is "
             "relevant, reply exactly 'nothing relevant'. At most 120 words.")
    prompts = [f"{qblock}\n\nPart {k + 1} of {len(parts)}:\n{p}\n\nRelevant facts:" for k, p in enumerate(parts)]
    outs = ask_batch(prompts, system=sys_n, options={"temperature": 0, "max_output_tokens": 250}).texts
    return [f"[part {k + 1}] {o.strip()}" for k, o in enumerate(outs)
            if o and "nothing relevant" not in o.strip().lower()[:40]]


def answer_longdoc(request):
    text = request["input"] if isinstance(request["input"], str) else request["input"][-1]["content"]
    body, qblock = split_request(text)
    notes = "\n".join(take_notes(body, qblock))[:150000] or "(no relevant facts were found)"
    sys_r = ("Answer the multiple-choice question using notes taken, part by part, from a long document. Reason "
             "briefly, then finish with one line 'ANSWER: <letter>'.")
    out = ask(f"Notes from the document, in order:\n{notes}\n\n{qblock}", system=sys_r,
              options={"temperature": 0, "max_output_tokens": 1500})
    m = re.findall(r"ANSWER:\s*\(?([ABCD])\b", out or "")
    if not m:  # the reasoning ran out before the ANSWER line: ask once for the letter
        out2 = ask(f"{qblock}\n\nReasoning so far:\n{(out or '')[-3000:]}\n\nReply with one line 'ANSWER: <letter>'.",
                   options={"temperature": 0, "max_output_tokens": 20})
        m = re.findall(r"\b([ABCD])\b", out2 or "")
    return "ANSWER: " + (m[-1] if m else "unknown")
