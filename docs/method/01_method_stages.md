# Method Stages

| Stage | Purpose | Main Procedure | Paper-Facing Output |
|---|---|---|---|
| 1. Corpus construction | Build a three-source corpus of sustainability discourse | Academic abstracts were collected from Semantic Scholar, media articles from The Guardian, and corporate disclosures from SEC Form 10-K filings in oil and gas and metal mining | Raw source corpus before segmentation |
| 2. Text segmentation | Make heterogeneous source documents comparable | Academic abstracts were retained as compact units, Guardian articles were split into paragraphs, and 10-K sections were split into sentence-grouped chunks | Source-specific text chunks with document metadata |
| 3. Environmental taxonomy | Define a common issue vocabulary | Six macro-topics were constructed from business-facing sustainability issue areas informed by the SDG Industry Matrix | Six environmental macro-topic domains |
| 4. Embedding assignment | Assign texts to environmental domains | Chunk embeddings were compared with phrase-level descriptor embeddings using cosine similarity; only texts above the domain threshold were retained | Provisional source-domain assignments |
| 5. Relevance validation | Remove generic or weak matches | A binary language-model validation prompt assessed whether each text substantively matched the assigned domain | Validated environmental corpus |
| 6. BERTopic microtopic modeling | Identify issue structures within each source-domain cell | Separate BERTopic models were fitted for each of the 18 source-by-domain subgroups | Source-domain microtopics |
| 7. Topic review and merge | Improve interpretability and reduce fragmentation | Candidate merge groups were reviewed using semantic evidence, topic labels, representations, and representative examples | Consolidated microtopics |
| 8. Corporate-centered sample construction | Build the final comparative sample | Corporate microtopics were treated as anchors; academic and media topics were selected by similarity or manually recovered for review | Aligned, unpaired, and excluded external topics |
| 9. Longitudinal measurement | Compare topic trajectories across sources | Annual document prevalence was calculated as topic documents divided by retained source-domain documents in the same year | Comparable annual time series |
| 10. Qualitative temporal synthesis | Interpret how topics evolved | Yearly topic words, representative snippets, and LLM-assisted summaries were condensed into phase narratives | Topic-level evolution summaries |
| 11. Source-relation diagnostics | Describe co-movement and timing between sources | Same-year Spearman, first active-year gaps, and peak-year gaps were calculated | Diagnostic tables for source relations |

## Key Counts

- Raw analytical corpus: 1,118,137 unique source documents.
- Text chunks after segmentation: 3,237,463 chunks.
- Validated environmental corpus: 353,971 chunk-domain rows, corresponding to
  291,034 unique documents.
- Topic-modeling design: 18 source-by-domain BERTopic subgroups.
- Final corporate anchors: 27 consolidated corporate microtopics.
- Non-corporate topics reviewed in the corporate-centered workbook: 214.
- Final external review outcomes: 151 aligned counterparts, 19 relevant
  unpaired external topics, and 44 excluded topics.
