# MedRAG Replication Study Results

Total experiments: 11


## MEDQA

Best accuracy: 67.3% (bm25_statpearls)

| Experiment      | k    | Accuracy   | Correct   |
|:----------------|:-----|:-----------|:----------|
| bm25_statpearls | 16.0 | 67.3%      | 202/300   |
| bm25_statpearls | 8.0  | 67.0%      | 201/300   |
| test_baseline   | N/A  | 66.7%      | 2/3       |
| bm25_statpearls | 4.0  | 66.0%      | 198/300   |
| baseline_cot    | N/A  | 66.0%      | 198/300   |
| bm25_pubmed     | 4.0  | 65.7%      | 197/300   |
| bm25_statpearls | 32.0 | 65.0%      | 195/300   |
| bm25_pubmed     | 16.0 | 64.0%      | 192/300   |
| bm25_pubmed     | 8.0  | 63.7%      | 191/300   |
| bm25_pubmed     | 32.0 | 63.0%      | 189/300   |



## PUBMEDQA

Best accuracy: 22.0% (baseline_cot)

| Experiment   | k   | Accuracy   | Correct   |
|:-------------|:----|:-----------|:----------|
| baseline_cot | N/A | 22.0%      | 110/500   |



---

# Comparison with Original MedRAG Paper

Note: Original uses full datasets; our replication uses subsets.


## MEDQA

| Metric | Original Paper | Our Replication | Difference |
|--------|----------------|-----------------|------------|
| Baseline | 50.6% | 66.0% | +15.4% |
| Best RAG | 53.0% | 67.3% | +14.3% |

Original best config: MedCPT + Textbooks

## PUBMEDQA

| Metric | Original Paper | Our Replication | Difference |
|--------|----------------|-----------------|------------|
| Baseline | 71.0% | 22.0% | -49.0% |

Original best config: MedCPT + PubMed