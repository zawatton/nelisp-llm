# Native file-action generation

The opt-in artifact grammar `file-actions-v1` lets one native model choose
between a `read` tool call, an `edit` tool call, and `DONE`. Unlike a fixed
template, it does not contain candidate paths, search strings, replacements,
or answers. This is a file-tool profile, not arbitrary Lisp generation or
general support for every MCP schema.

```elisp
(:type "file-actions-v1" :max-field 128)
```

`:max-field` is required, from 1 to 1024 decoded characters per field.
Optional `:allow` supplies one shared content alphabet of 1 to 512 printable
Unicode scalar characters. The default is printable ASCII. Configure this
alphabet independently of held-out task answers; a per-case answer alphabet
would leak information into evaluation. Non-ASCII content also requires a
matching `utf8-byte-v1` model.

The model selects `DONE` versus a fenced tool call, `read` versus `edit`, the
field characters, and when each field ends. Fixed scaffolding supplies only
the existing runtime's syntax:

````text
```tool
(:name "read" :arguments (:path "notes.txt"))
```
````

An edit has `:path`, `:search`, and `:replace` string arguments. Path and search
must be nonempty; replacement and the single-line DONE answer may be empty.
Quoted tool fields support escaped quotation marks, backslashes, newline,
carriage return, and tab. At the configured field bound, the grammar closes
the field; it never leaves an unfinished escape. DONE ends with a newline.

Grammar validity is not authorization or task correctness. Generated paths
and edits still cross the ordinary tool registry, workspace confinement, and
permission checks. This profile does not expose shell or arbitrary Elisp.
Context capacity must cover both the complete prompt/history and generated
action; the grammar does not truncate history or expand model context.

`nl-llm-agent-grammar-file-actions` constructs the callback directly. Artifact
normalization, JSON publication, and reload preserve the same data-only
profile. Legacy done/message/template profiles remain available unchanged.

Run `make test-agent-actions` for syntax and artifact tests. Scripted logits
can verify that every branch is reachable through the same grammar, but are
not evidence that a trained model knows which branch or content is correct.
For capability measurement, use real model logits and the same fixed grammar
before and after [completion-only training](completion-training.md), then
score actual files with the agent's task evaluator. No self-training,
publication, or provider switch is enabled merely by selecting this grammar.
