person, item = re.search(r"did (\w+) buy an? (\w+)\?", question).groups()
hits = [l for l in lines if person in l and item in l]
reply = ask(
    "Lines:\n" + "\n".join(hits) + f"\n\nOn which day did {person} buy a {item}? Count only a "
    "completed purchase (bought, purchased, paid for, picked up per a receipt, a logged sale, or an "
    "order that went through); ignore complaints, reviews, wishlists, considered or declined "
    "purchases, and returns. Finish with one line of the form ANSWER: <day number>.",
    options={"temperature": 0, "max_output_tokens": 512},
)
m = re.search(r"ANSWER:\s*(\d+)", reply)
nums = re.findall(r"\d+", reply)
FINAL_TEXT("ANSWER: " + (m.group(1) if m else (nums[-1] if nums else reply.strip())))
