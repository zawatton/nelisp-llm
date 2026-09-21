#!/usr/bin/env python3
"""Write the donor Qwen byte-level vocabulary as an Emacs Lisp table."""

import json
import os
import sys


def bytes_to_surface_table():
    """Return the GPT-2 byte-to-unicode alphabet as byte -> character."""
    direct = (list(range(33, 127)) + list(range(161, 173))
              + list(range(174, 256)))
    out = {b: chr(b) for b in direct}
    next_cp = 256
    for byte in range(256):
        if byte not in out:
            out[byte] = chr(next_cp)
            next_cp += 1
    return out


def surface_bytes(text, reverse):
    try:
        return bytes(reverse[ch] for ch in text)
    except KeyError as exc:
        raise SystemExit(f"token {text!r} is outside the byte-level alphabet") from exc


def elisp_string(value):
    return json.dumps(value, ensure_ascii=False)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: qwen-token-table.py OUT")
    out = sys.argv[1]
    donor = os.path.join("build", "donor", "qwen3-0.6b")
    with open(os.path.join(donor, "tokenizer.json"), encoding="utf-8") as fh:
        tokenizer = json.load(fh)
    with open(os.path.join(donor, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)

    alphabet = bytes_to_surface_table()
    reverse = {surface: byte for byte, surface in alphabet.items()}
    entries = []
    for token, token_id in tokenizer["model"]["vocab"].items():
        raw = surface_bytes(token, reverse)
        surface = "".join(alphabet[b] for b in raw)
        entries.append((surface, token_id))
    for added in tokenizer.get("added_tokens", []):
        entries.append((added["content"], added["id"]))
    entries.sort(key=lambda item: item[1])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(";; -*- lisp-data -*-  Qwen byte-level token table\n")
        fh.write(f"(:vocab-size {config['vocab_size']} :tokens [")
        fh.write(" ".join(f"({elisp_string(surface)} . {token_id})"
                          for surface, token_id in entries))
        fh.write("])\n")


if __name__ == "__main__":
    main()
