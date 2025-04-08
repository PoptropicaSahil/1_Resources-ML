# Questions and Answers from NL Pandey (FODO AI)

### Explain Hit Rate in Information Retrieval (or recommendation system) and when it fails?

Hit Rate (HR) measures how often at least one relevant document appears in the top-K retrieved results. 

It is defined as:
 HR@K = (Queries with ≥1 relevant doc in top K) / (Total queries)

Interpretation:
- HR@K = 1: Every query has a relevant document in top-K
- HR@K = 0: No query has a relevant document in top-K

When Hit Rate Fails ?

1. Ignores Ranking Within Top-K
 - **Doesn't differentiate between rank 1 and rank K**
 - Critical for ranking-sensitive applications

2. Fails to Capture Partial Relevance
 - Gives full credit even if only one relevant document is retrieved
 - Doesn't distinguish between retrieving one or all relevant documents

3. Binary Relevance Assumption
 - Assumes documents are either relevant or not
 - Ignores graded relevance

4. Fails for Precision-Sensitive Tasks
 - **Doesn't penalize irrelevant documents in top-K**
 - Same score for 1 relevant + 99 irrelevant vs. 10 relevant + 0 irrelevant

5. Not Suitable for Long-Tail Queries
 - May give low scores for rare or highly specific relevant documents

Alternatives to Hit Rate - 
Precision@K → Measures how many retrieved documents are relevant.
Recall@K → Measures how many relevant documents were retrieved.
NDCG → Accounts for graded relevance and ranking order.
