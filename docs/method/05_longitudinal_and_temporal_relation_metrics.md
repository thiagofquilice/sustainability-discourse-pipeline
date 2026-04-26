# Longitudinal And Temporal-Relation Metrics

## Annual Document Prevalence

The main longitudinal metric is annual document prevalence:

```text
annual_document_prevalence =
topic documents in source s, macro-topic m, year y
/
all retained documents in source s, macro-topic m, year y
```

This denominator corrects for annual changes in corpus size. It is especially
important for corporate disclosure, where increases in the number of annual
filings could otherwise be mistaken for increased issue salience.

## Longitudinal Panels

For each corporate anchor, the panels compare:

- the corporate anchor series;
- the aggregate of aligned academic counterparts, when present;
- the aggregate of aligned media counterparts, when present.

Unpaired external topics are shown separately by macro-topic because they are
substantively relevant but do not have a retained corporate counterpart.

## Same-Year Spearman

Same-year Spearman correlation measures contemporaneous co-movement between a
corporate series and an external series.

```text
spearman_r > 0: same-direction movement
spearman_r < 0: opposite-direction movement
spearman_r near 0: little monotonic association
```

The `spearman_p` value indicates whether the observed correlation is
statistically distinguishable from no association. It does not imply causality.

## First Active-Year Gap

```text
first_active_year_gap =
corporate_first_active_year - external_first_active_year
```

Positive values mean the external topic appeared before the corporate topic.
Negative values mean the corporate topic appeared first.

## Peak-Year Gap

```text
peak_year_gap =
corporate_peak_year - external_peak_year
```

Positive values mean the external topic peaked before the corporate topic.
Negative values mean the corporate topic peaked first.

No plus/minus one-year window is applied in the summary table.

## Interpretation

These diagnostics are descriptive. Same-year Spearman summarizes whether two
annual prevalence series move together, while the first-active-year and
peak-year gaps summarize timing. They are useful for describing alignment,
delay, or contrast across sources, but they should not be interpreted as causal
tests of influence.
