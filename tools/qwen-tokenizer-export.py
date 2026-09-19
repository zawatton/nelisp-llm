#!/usr/bin/env python3
"""Export a HuggingFace byte-level BPE tokenizer to the nl-llm binary table.

Doc 08 (docs/design/08-weight-import.org) Phase 1.  This is an OFFLINE tool, not
runtime: it turns a donor `tokenizer.json' into a compact binary that the pure
Elisp loader in `lisp/nl-llm-qwen-tokenizer.el' can read without parsing 11 MB
of JSON.

The table also carries the Unicode letter/number ranges the pre-tokenizer
needs, and those ranges are obtained by ASKING THE REFERENCE PRE-TOKENIZER,
one code point at a time, rather than by consulting a Unicode database.  Both
alternatives are wrong in the same quiet way: the host's tables (Emacs,
NeLisp) and Python's `unicodedata' each track their own Unicode version, and a
letter that one of them has not heard of yet tokenizes differently with no
error anywhere.  Probing the reference is the only source that is by
construction the version the donor's tokenizer actually uses; a full sweep
costs a few seconds.  This tool reports how far `unicodedata' disagrees so the
skew stays visible instead of becoming folklore.

Binary format `nl-llm-tok-v2' (little-endian u32 throughout):

    magic    16 bytes  "nl-llm-tok-v2" + three NULs
    n_vocab  u32       number of token entries with known bytes
    n_merge  u32       number of merge rules
    n_added  u32       number of added/special tokens
    vocab_sz u32       the model's padded vocabulary size (config vocab_size)
    n_lrange u32       number of letter code-point ranges
    n_nrange u32       number of number code-point ranges
    offs     u32 * (n_vocab + 1)   byte offsets into blob, id order from 0
    merges   u32 * (n_merge * 3)   (left_id, right_id, result_id), rank order
    added    u32 * (n_added * 2)   (id, index into offs) for added tokens
    lranges  u32 * (n_lrange * 2)  inclusive (lo, hi), ascending
    nranges  u32 * (n_nrange * 2)  inclusive (lo, hi), ascending
    blob     bytes                 concatenated raw token bytes

Every option that would change the BPE algorithm is refused rather than
silently exported, because a table that encodes differently from the donor is
the one failure mode here that produces fluent-looking garbage instead of an
error.

Usage:
    PYTHONPATH=build/donor/pylibs \\
      python3 tools/qwen-tokenizer-export.py DONOR_DIR OUT_DIR
"""

import json
import os
import struct
import sys
import unicodedata

from tokenizers import Tokenizer

MAGIC = b"nl-llm-tok-v2" + b"\0" * 3
MAX_CP = 0x110000

# Unicode White_Space, fixed since Unicode 4.1, so it needs no probing and is
# duplicated in the Elisp loader rather than exported.
WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0),
               (0x1680, 0x1680), (0x2000, 0x200A), (0x2028, 0x2029),
               (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]

SURROGATES = (0xD800, 0xDFFF)


def bytes_to_unicode():
    """GPT-2's reversible byte <-> printable-codepoint map, char -> byte."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes(text, decoder):
    """Raw bytes of a byte-level-encoded vocabulary TEXT."""
    try:
        return bytes(decoder[ch] for ch in text)
    except KeyError as exc:
        raise SystemExit(f"token {text!r} has a char outside the byte-level "
                         f"alphabet: {exc}") from exc


def to_ranges(flags):
    """Inclusive (lo, hi) ranges of the set positions in FLAGS."""
    ranges, lo = [], None
    for cp, hit in enumerate(flags):
        if hit and lo is None:
            lo = cp
        elif not hit and lo is not None:
            ranges.append((lo, cp - 1))
            lo = None
    if lo is not None:
        ranges.append((lo, len(flags) - 1))
    return ranges


def probe_categories(pre):
    """Classify every code point by asking the reference pre-tokenizer PRE.

    The donor's pattern gives two observable signatures:

      "a" + X  is one piece  <=>  X is a letter, because the letter-run branch
                                  extends over both characters
      X + X    is two pieces <=>  X is a number, because the number branch
                                  matches exactly one character (a symbol run
                                  would swallow both)

    Lone surrogates are skipped: they cannot cross into the reference's Rust
    string type, and they cannot occur in decoded text either.
    """
    ws = set()
    for lo, hi in WHITE_SPACE:
        ws.update(range(lo, hi + 1))
    is_l = bytearray(MAX_CP)
    is_n = bytearray(MAX_CP)
    for cp in range(MAX_CP):
        if SURROGATES[0] <= cp <= SURROGATES[1]:
            continue
        ch = chr(cp)
        if len(pre.pre_tokenize_str("a" + ch)) == 1:
            is_l[cp] = 1
        elif cp not in ws and len(pre.pre_tokenize_str(ch + ch)) == 2:
            is_n[cp] = 1
    return is_l, is_n


def report_unicodedata_skew(is_l, is_n):
    """Print how far Python's Unicode tables disagree with the probe."""
    skew = 0
    example = None
    for cp in range(MAX_CP):
        if SURROGATES[0] <= cp <= SURROGATES[1]:
            continue
        init = unicodedata.category(chr(cp))[0]
        if (init == "L") != bool(is_l[cp]) or (init == "N") != bool(is_n[cp]):
            skew += 1
            if example is None:
                example = cp
    if skew:
        print(f"  note: python unicodedata {unicodedata.unidata_version} "
              f"disagrees with the reference on {skew} code points "
              f"(first U+{example:04X}); the probed values are authoritative")
    return skew


def main(donor, outdir):
    tok = json.load(open(os.path.join(donor, "tokenizer.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(donor, "config.json"), encoding="utf-8"))
    model = tok["model"]

    if model["type"] != "BPE":
        raise SystemExit(f"unsupported tokenizer model: {model['type']}")
    for flag, want in (("byte_fallback", False), ("ignore_merges", False),
                       ("fuse_unk", False), ("dropout", None),
                       ("continuing_subword_prefix", ""),
                       ("end_of_word_suffix", "")):
        if model.get(flag) != want:
            raise SystemExit(f"unsupported BPE option {flag}={model.get(flag)!r}")
    norm = (tok.get("normalizer") or {}).get("type")
    if norm not in (None, "NFC"):
        raise SystemExit(f"unsupported normalizer: {norm}")

    decoder = bytes_to_unicode()
    vocab = model["vocab"]                      # byte-level text -> id
    n_plain = max(vocab.values()) + 1
    entries = [None] * n_plain
    for text, tid in vocab.items():
        entries[tid] = token_bytes(text, decoder)
    missing = [i for i, e in enumerate(entries) if e is None]
    if missing:
        raise SystemExit(f"vocabulary has {len(missing)} holes, first {missing[:5]}")

    # Added tokens carry literal text, not byte-level text.
    added = []
    for a in tok.get("added_tokens", []):
        for flag in ("lstrip", "rstrip", "single_word"):
            if a.get(flag):
                raise SystemExit(f"added token {a['content']!r} sets {flag}, "
                                 "which the Elisp matcher does not implement")
        added.append((a["id"], a["content"].encode("utf-8")))

    # Merges resolve to ids at export time so the Elisp BPE never touches strings.
    merges = []
    for pair in model["merges"]:
        left, right = pair if isinstance(pair, list) else pair.split(" ")
        lid, rid = vocab.get(left), vocab.get(right)
        mid = vocab.get(left + right)
        if lid is None or rid is None or mid is None:
            raise SystemExit(f"merge {left!r}+{right!r} is not resolvable in vocab")
        merges.append((lid, rid, mid))

    reference = Tokenizer.from_file(os.path.join(donor, "tokenizer.json"))
    if reference.pre_tokenizer is None:
        raise SystemExit("donor has no pre_tokenizer to probe")
    is_l, is_n = probe_categories(reference.pre_tokenizer)
    lranges, nranges = to_ranges(is_l), to_ranges(is_n)
    if not lranges or not nranges:
        raise SystemExit("category probe found no letters or no numbers, "
                         "which means the probe itself is broken")

    # One flat id-ordered table: plain vocabulary first, then added tokens.
    table = list(entries) + [b for _, b in added]
    offs, blob, pos = [], bytearray(), 0
    for b in table:
        offs.append(pos)
        blob += b
        pos += len(b)
    offs.append(pos)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "tokenizer.bin")
    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<6I", len(table), len(merges), len(added),
                             cfg.get("vocab_size", n_plain),
                             len(lranges), len(nranges)))
        fh.write(struct.pack(f"<{len(offs)}I", *offs))
        for lid, rid, mid in merges:
            fh.write(struct.pack("<3I", lid, rid, mid))
        for i, (tid, _) in enumerate(added):
            fh.write(struct.pack("<2I", tid, n_plain + i))
        for lo, hi in lranges + nranges:
            fh.write(struct.pack("<2I", lo, hi))
        fh.write(blob)

    print(f"wrote {out}: {len(table)} tokens, {len(merges)} merges, "
          f"{len(added)} added, blob {len(blob)} bytes, "
          f"{os.path.getsize(out)} bytes total")
    print(f"  probed from the reference pre-tokenizer: "
          f"{len(lranges)} letter ranges, {len(nranges)} number ranges")
    report_unicodedata_skew(is_l, is_n)
    print(f"  normalizer={norm} vocab_size={cfg.get('vocab_size')}")
    print("  pre_tokenizer=" + json.dumps(tok.get("pre_tokenizer"),
                                          ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
