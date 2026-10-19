import re, json
text = request["input"]
log = text.split("LOG (", 1)[1].split("\n", 1)[1].split("\n\nQUESTION:", 1)[0]
lines = log.split("\n")
question = text.split("QUESTION:", 1)[1].split("\n", 1)[0].strip()
print(len(lines), "lines")
print(question)
