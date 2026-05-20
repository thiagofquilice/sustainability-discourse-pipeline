# Corporate-Centered Candidate Selection And Final Review

## Corporate Anchors

The final comparison used a corporate-centered design. All reviewed and merged
corporate microtopics retained for the paper were treated as anchors.

Final corporate anchor count:

```text
27 corporate microtopics
```

Academic and media microtopics were assessed according to their relationship
with those corporate anchors.

## Candidate Selection

External candidate topics entered the corporate-centered review by two routes.

### 1. Direct Similarity Route

Academic and media microtopics were compared with corporate anchors within the
same macro-topic domain. Candidate links were based on cosine similarity between
microtopic profiles.

Final corporate-focus matching threshold:

```text
cosine similarity >= 0.65
```

This threshold-based route selected:

```text
170 non-corporate microtopics
```

### 2. Manual Recovery Route

Some external topics were considered substantively relevant even though they did
not enter through the direct similarity route. These were recovered manually for
review to avoid reducing the comparison to semantic proximity alone.

Manual recovery added:

```text
44 non-corporate microtopics
```

Together, the two routes produced:

```text
214 non-corporate microtopics for substantive adjudication
```

## Final Review Criteria

Each reviewed non-corporate microtopic was classified into one of three
outcomes:

| Review outcome | Meaning |
|---|---|
| Aligned counterpart | The academic or media microtopic referred to the same or closely adjacent issue as a corporate anchor |
| Relevant unpaired external topic | The topic was substantively relevant to corporate sustainability disclosure but did not form a direct corporate counterpart |
| Excluded topic | The topic was not substantively useful for the corporate-centered comparison |

Final outcomes:

```text
152 aligned external counterparts
18 relevant unpaired external topics
44 excluded topics
```

## Sample Counts

The pre-review corporate-focused subset contained:

```text
140,339 unique documents
123,060 academic
11,305 media
5,974 corporate
```

After final review, the post-review subset contained:

```text
116,356 unique documents
99,355 academic
11,027 media
5,974 corporate
```

## Interpretation

The corporate-centered review was not a simple topic-matching exercise. It was
designed to distinguish:

- issues incorporated into corporate disclosure;
- relevant external issues that remained outside the corporate frame;
- topics that were not substantively useful for the final comparison.

This is the basis for interpreting alignment, external spillover, and corporate
blind spots in the paper.
