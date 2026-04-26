# Embedding, Descriptor Construction, Thresholds, And Validation

## Environmental Taxonomy

The corpus was organized around six environmental macro-topics:

| Code | Macro-topic |
|---|---|
| T1 | Clean energy transition |
| T2 | Operational sustainability and circular production |
| T3 | Sustainable products, services and consumption |
| T4 | Climate strategy, carbon governance and disclosure |
| T5 | Climate risk, adaptation and resilience |
| T6 | Ecosystems, pollution and environmental stewardship |

Each macro-topic was represented by short phrase-level descriptors. The
descriptor catalog included a topic definition, issue elements, boundary notes,
and false-positive notes. The operational matching step used the individual
descriptor expressions, not a single pooled paragraph for each macro-topic.

The final descriptor catalog is represented in the source pipeline by
`catalog/six_topic_discourse_catalog.json`.

## Embedding Model

The final pipeline used the sentence-transformer model:

```text
BAAI/bge-large-en-v1.5
```

The embedding dimensionality was 1024. Text chunks and descriptor expressions
were embedded in the same vector space.

## Domain Assignment

Each text chunk was compared with the descriptor-expression embeddings by
cosine similarity. For each macro-topic, the domain score was the strongest
cosine similarity between the chunk and any descriptor expression belonging to
that macro-topic. A chunk received one provisional macro-topic assignment only
when the strongest macro-topic score exceeded the configured threshold:

```text
similarity_threshold = 0.65
```

Chunks below the threshold were assigned to `OUTSIDE_SCOPE` and were not carried
into environmental topic modeling.

## Binary Validation

The provisional domain assignments were then submitted to a binary
language-model validation step. This step was designed to remove generic
sustainability language, generic ESG/risk language, and nearby but substantively
different matches.

The validation prompt instructed the model to answer `yes` only when the text
substantively discussed the assigned topic or a clearly equivalent concept. It
instructed the model to answer `no` when the match was vague, generic, lexical
only, or merely adjacent.

Prompt template:

- `prompts/binary_domain_validation_prompt.txt`

## Audit Logic

Manual audit focused on the risk of false positives introduced by broad
sustainability language. The audit design prioritized difficult cases, including
model rejections and harder positives within source-topic cells.

The audit principle was conservative: when the text was only loosely related,
the assignment was treated as not substantively valid.
