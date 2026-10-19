reply = model_complete(request)
FINAL_TEXT("".join(c["text"] for o in reply["output"] if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text"))
