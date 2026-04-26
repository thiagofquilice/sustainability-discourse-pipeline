# Microtopic Modeling, Review, And Merge

## Source-Domain BERTopic Models

BERTopic was used after environmental validation as a taxonomy-anchored
microtopic layer. Rather than fitting one topic model over the full corpus,
separate models were estimated for each source-by-domain subgroup:

```text
3 sources x 6 macro-topics = 18 BERTopic subgroups
```

This design preserved institutional differences in how academic, media, and
corporate texts discussed the same broad environmental domain.

## Representation

The final microtopic pipeline used multi-aspect BERTopic representations,
including KeyBERT-inspired, part-of-speech, and MMR-based representations. The
same embedding model, `BAAI/bge-large-en-v1.5`, was used for representation
support where needed.

## Merge Review

The review stage consolidated semantically overlapping or over-fragmented
microtopics. Candidate merge groups were constructed from mutual topic-profile
similarity inside each source-domain subgroup.

Final merge-review threshold:

```text
profile_similarity >= 0.70
```

The final multiaspect merge-review workbook covered:

- 785 retained non-outlier microtopics in scope.
- 586 filtered mutual candidate edges.
- 275 connected components.
- 104 non-singleton proposed groups for review.
- 171 singleton topics requiring no group review.

## Review Criteria

The merge review prioritized:

- semantic similarity;
- agreement across main, part-of-speech, and secondary/aspect representations;
- hierarchical proximity;
- representative evidence.

Temporal resemblance was not treated as merge evidence. This was important
because two topics could rise and fall together while still representing
different substantive issues.

The reviewer could:

- accept a proposed merge group;
- reject a proposed merge group;
- split a proposed group into manually specified subgroups.

The materialized outputs assigned every retained microtopic to a final
`final_merge_group_id`, which became the unit used in the corporate-centered
analysis.

## Temporal Summaries

For retained topics, yearly topic evidence was summarized with a language-model
prompt using annual topic words and representative snippets. A second prompt
then condensed yearly evidence into 2-4 contiguous phases of topic evolution.

Prompt templates:

- `prompts/year_summary_prompt.txt`
- `prompts/evolution_summary_prompt.txt`
