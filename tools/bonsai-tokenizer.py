#!/usr/bin/env python3
"""Vocabulary in and out of the GGUF, so a predicted id can be read.

Greedy longest-match over the stored vocabulary rather than the model's real
BPE merges.  That is enough for what it is for -- putting a prompt in and
reading a continuation out -- and it is honest about being an approximation:
it never invents a token that is not in the vocabulary.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

SPACE = "\u0120"   # byte-level BPE renders a leading space as this
NEWLINE = "\u010a"


def load_vocab(path):
    with open(path, "rb") as f:
        info = gguf.parse(f.read(512 << 20))
    kv = info["kv"]
    toks = kv["tokenizer.ggml.tokens"]
    return toks, kv


def render(tok):
    return tok.replace(SPACE, " ").replace(NEWLINE, "\n")


def encode(toks, text):
    index = {}
    for i, t in enumerate(toks):
        index.setdefault(t, i)
    raw = text.replace(" ", SPACE)
    out, i, longest = [], 0, max(len(t) for t in index)
    while i < len(raw):
        for n in range(min(longest, len(raw) - i), 0, -1):
            piece = raw[i:i + n]
            if piece in index:
                out.append(index[piece])
                i += n
                break
        else:
            raise SystemExit("no vocabulary entry covers %r" % raw[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--encode")
    ap.add_argument("--decode", nargs="*", type=int)
    ap.add_argument("--dump", help="write id<TAB>token to this file")
    args = ap.parse_args()
    toks, kv = load_vocab(args.gguf)
    print("vocab %d entries" % len(toks), file=sys.stderr)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for i, t in enumerate(toks):
                f.write("%d\t%s\n" % (i, t.replace("\t", "\\t").replace("\n", "\\n")))
        print("wrote %s" % args.dump, file=sys.stderr)
    if args.encode is not None:
        ids = encode(toks, args.encode)
        print(" ".join(str(i) for i in ids))
        print("  -> %s" % "|".join(render(toks[i]) for i in ids), file=sys.stderr)
    if args.decode:
        print("".join(render(toks[i]) for i in args.decode))


if __name__ == "__main__":
    main()
