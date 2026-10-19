import re, json
from datetime import datetime
text = request["input"]
rows = []
for line in text.split("\n"):
    m = re.match(r"Date: (.*?) \|\| User: (\d+) \|\| Instance: (.*)$", line)
    if m:
        rows.append({"date": datetime.strptime(m.group(1).strip(), "%b %d, %Y").date(), "user": m.group(2), "text": m.group(3)})
question = [l for l in text.split("\n") if l.strip()][-1].strip()
header = text.split("\nDate:", 1)[0]
m = re.search(r"classified as (\w+) or (\w+)", header)
labels = list(m.groups()) if m else re.findall(r"'([^']+)'", header.split("categories:", 1)[1].split("\n", 1)[0]) if "categories:" in header else []
print(len(rows), "rows;", "labels:", labels)
print(question)
