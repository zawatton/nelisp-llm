#!/usr/bin/env python3
"""Real byte-level BPE for the Bonsai vocabulary, from the GGUF's own merges.

The greedy longest-match encoder in tools/bonsai-tokenizer.py is fine for
putting a prompt in and reading a token out, and wrong for measuring anything:
it produces sequences the model was never trained on, which inflates the cross
entropy by an unknown amount.  The positive control used the donor's real
tokenizer, so comparing the two without this was comparing a model on its own
tokenisation against a model on an approximation of one.

Pre-tokenizer regex is the GPT-4 / Qwen split; merges and vocabulary come from
the file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

PAT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
       r"|[^\r\n\p{L}\p{N}]?\p{L}+"
       r"|\p{N}{1,3}"
       r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
       r"|\s*[\r\n]+"
       r"|\s+(?!\S)"
       r"|\s+")


def byte_encoder():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    def __init__(self, path):
        with open(path, "rb") as f:
            info = gguf.parse(f.read(512 << 20))
        kv = info["kv"]
        self.tokens = kv["tokenizer.ggml.tokens"]
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        merges = kv["tokenizer.ggml.merges"]
        self.ranks = {}
        for i, mm in enumerate(merges):
            a, _, b = mm.partition(" ")
            self.ranks[(a, b)] = i
        self.benc = byte_encoder()
        import regex
        self.pat = regex.compile(PAT)

    def _bpe(self, word):
        parts = list(word)
        if len(parts) < 2:
            return parts
        while True:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                return parts
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]

    def encode(self, text):
        out = []
        for piece in self.pat.findall(text):
            word = "".join(self.benc[b] for b in piece.encode("utf-8"))
            for sym in self._bpe(word):
                i = self.ids.get(sym)
                if i is None:
                    raise SystemExit("symbol %r is not in the vocabulary" % sym)
                out.append(i)
        return out

    def decode_one(self, i):
        return self.tokens[i]


def main():
    bpe = BPE(sys.argv[1])
    text = sys.argv[2]
    ids = bpe.encode(text)
    print(" ".join(str(i) for i in ids))
    print("%d tokens: %s" % (len(ids),
                             "|".join(bpe.decode_one(i).replace("Ġ", " ")
                                      for i in ids[:24])), file=sys.stderr)


if __name__ == "__main__":
    main()
