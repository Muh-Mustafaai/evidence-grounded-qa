# SciFact Dataset Notes

## Dataset
- **Name:** SciFact (BEIR benchmark)
- **Split:** Test
- **Corpus documents:** 5,183
- **Queries:** 300
- **Queries with relevance judgments:** 300

## Data format
- **Corpus:** Document IDs with scientific abstract text and titles.
- **Queries:** Query IDs paired with scientific claims.
- **Relevance judgments:** Query IDs mapped to relevant document IDs and relevance scores.

## Example
- **Query:** `0-dimensional biomaterials show inductive properties.`
- **Relevant document ID:** `31715818`
- **Relevance score:** `1`

## Research notes
SciFact provides a larger, labeled benchmark for comparing BM25, semantic retrieval, and hybrid retrieval. The test split will be used consistently across the methods.

The current eight-document starter dataset remains useful for debugging. Results from that starter dataset must be reported separately from SciFact results.

## Limitations to address
- SciFact queries are scientific claims rather than ordinary questions.
- The benchmark's relevance judgments define which retrieved documents count as relevant.
- Results on this dataset may not generalize to other domains or query types.
