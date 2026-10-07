"""Reads a JSON answer of the service on stdin and prints it for reading: base64 strings cut to 24 characters,
only the first N detections kept (the rest counted), floats rounded to 3 decimals, short objects on one line.
usage: curl -s ... | python3 docs/examples/pretty.py [N=2]"""
import json
import sys

KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def prepare(v, key=None):
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, str) and key in ("image", "result_image_b64"):
        return v[:24] + "..."
    if isinstance(v, dict):
        return {k: prepare(x, k) for k, x in v.items()}
    if isinstance(v, list):
        items = [prepare(x) for x in v]
        if key == "detections" and len(items) > KEEP:
            return items[:KEEP] + [f"... {len(items) - KEEP} more"]
        return items
    return v


def show(v, depth=0):
    pad = "  " * depth
    flat = json.dumps(v, ensure_ascii=False)
    if not isinstance(v, (dict, list)) or len(flat) + len(pad) <= 100:
        return flat
    if isinstance(v, dict):
        body = ",\n".join(f'{pad}  {json.dumps(k)}: {show(x, depth + 1)}' for k, x in v.items())
        return "{\n" + body + "\n" + pad + "}"
    return "[\n" + ",\n".join(pad + "  " + show(x, depth + 1) for x in v) + "\n" + pad + "]"


print(show(prepare(json.load(sys.stdin))))
