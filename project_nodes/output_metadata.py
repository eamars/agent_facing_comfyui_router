"""Remove input image names from the prompt metadata ComfyUI embeds in saved PNGs.

SaveImage writes the executed graph as a `prompt` text chunk. Prompts, seeds and
models stay useful for reproducing a result, but LoadImage file names identify
the uploaded inputs (and, for content-hashed uploads, the photo itself). Only
those names are replaced; every other chunk, including the pixel data, is copied
byte for byte.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REDACTED_INPUT = "[input image]"


def _redact_graph(text: str) -> str | None:
    try:
        graph = json.loads(text)
    except ValueError:
        return None
    if not isinstance(graph, dict):
        return None
    changed = False
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        # Loader filenames are the only plain-string "image" inputs; linked images are [node, slot] lists.
        if isinstance(inputs, dict) and isinstance(inputs.get("image"), str) and inputs["image"] != REDACTED_INPUT:
            inputs["image"] = REDACTED_INPUT
            changed = True
    return json.dumps(graph) if changed else None


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _rewrite_text_chunk(kind: bytes, data: bytes) -> bytes | None:
    keyword, _, rest = data.partition(b"\x00")
    if keyword != b"prompt":
        return None
    if kind == b"tEXt":
        redacted = _redact_graph(rest.decode("latin-1"))
        if redacted is None:
            return None
        try:
            return _chunk(b"tEXt", keyword + b"\x00" + redacted.encode("latin-1"))
        except UnicodeEncodeError:
            return _chunk(b"iTXt", keyword + b"\x00\x00\x00\x00\x00" + redacted.encode("utf-8"))
    if kind == b"zTXt":
        redacted = _redact_graph(zlib.decompress(rest[1:]).decode("latin-1"))
        if redacted is None:
            return None
        return _chunk(b"iTXt", keyword + b"\x00\x01\x00\x00\x00" + zlib.compress(redacted.encode("utf-8")))
    # iTXt: compression flag, method, language\0, translated keyword\0, text
    flag, method, tail = rest[0], rest[1], rest[2:]
    language, _, tail = tail.partition(b"\x00")
    translated, _, text = tail.partition(b"\x00")
    redacted = _redact_graph((zlib.decompress(text) if flag else text).decode("utf-8"))
    if redacted is None:
        return None
    body = redacted.encode("utf-8")
    return _chunk(b"iTXt", keyword + b"\x00" + bytes([flag, method]) + language + b"\x00" + translated + b"\x00"
                  + (zlib.compress(body) if flag else body))


def redact_input_names(path: Path) -> bool:
    """Rewrite `path` in place when its prompt metadata names input images. Returns True if it changed."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        return False
    out = [PNG_SIGNATURE]
    position, changed = len(PNG_SIGNATURE), False
    while position + 8 <= len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        end = position + 12 + length
        replacement = _rewrite_text_chunk(kind, data[position + 8:position + 8 + length]) if kind in {b"tEXt", b"zTXt", b"iTXt"} else None
        out.append(replacement if replacement is not None else data[position:end])
        changed |= replacement is not None
        position = end
    if not changed:
        return False
    temporary = path.with_name(path.name + ".redacting")
    temporary.write_bytes(b"".join(out))
    os.replace(temporary, path)
    return True
