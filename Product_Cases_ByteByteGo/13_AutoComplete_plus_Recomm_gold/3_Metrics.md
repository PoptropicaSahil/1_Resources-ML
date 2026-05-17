# FROM GEMINI. Clearing MAJOR doubts

> Comprehension is Compression!

## METRICS

* **Offline Metrics** (used during model training)
* **Online Metrics** (business KPIs during live A/B testing)

---

### 1. E-Commerce / Audiobooks (Amazon Books/Audible)

> MAIN idea: Monetization

**Stage 1: Retrieval (Two-Tower)**

* Recall@K
* Item Coverage: `The percentage of your total catalog that actually gets retrieved over a given day. Neural retrievers can suffer from popularity bias, ignoring the long tail`

**Stage 2: Reranking (DCN-v2 + MMoE)**

* **NDCG@K (Normalized Discounted Cumulative Gain):** `The gold (GOAT) standard for listwise ranking`. It heavily penalizes the model for putting a highly relevant book at rank 10 instead of rank 1

```math
NDCG@K = \frac{DCG@K}{IDCG@K} = \frac{\sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}}{\text{Ideal } DCG@K}
```

* **Multi-Task AUC-ROC:** For the CTR gate; Book CVR gate; Audible CVR gate

**Online (A/B Testing)**

* **Revenue per Session / ARPU** 
* **Conversion Rate (CVR):** Split by format (physical vs. audio)
* **Diversity:** Are we just recommending the same top 10 bestsellers? (Measured via Gini coefficient of recommended items)

---

### 2. Music Streaming (Spotify)

> Since the cost of consumption is low, the MAIN IDEA shifts from "purchases" to "continuous engagement" and "habituation"

**Stage 1: Retrieval (Session-Based Two-Tower)**

* **Hit Rate@K / Next-Track Recall:** Did the sequence-based model retrieve the actual next track the user chose to listen to?

**Stage 2: Reranking (Contextual DCN-v2 + MMoE)**

* **Intra-List Diversity (ILD):** Measures the `average distance between items in the recommended list.` A high ILD means the user is getting a good mix of genres/artists, preventing listener fatigue.
* **Log-Loss / BCE (Binary Cross-Entropy):** Specifically for the `skip-prediction task`. We want the model to be highly calibrated on predicting negative sentiment.

**Online (A/B Testing)**

* **`Skip Rate`:** The percentage of tracks skipped before 30 seconds. Must go down
* **Session Length / Total Playtime**
* **Save Rate**

---

### 3. Autocomplete / Query Prediction

> MAIN IDEA: Latency and cognitive load. ML system acts as a *mind reader*

**Stage 1: Retrieval (Prefix-to-Query)**

* **Mean Reciprocal Rank (MRR):** Evaluates where the target query appeared in the retrieved candidate list

```math
MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
```

**Stage 2: Reranking (Query MMoE)**

* **`Keystroke Savings`:** A specialized metric for typeahead systems. It measures the percentage of characters the user *didn't* have to type because the system guessed correctly.

```math
KS = 1 - \frac{\text{Keystrokes Typed}}{\text{Total Target Query Length}}
```

* **Expected Search Success Rate:** The model's predicted probability that the top-ranked query will actually yield clicks on the subsequent search page.

**Online (A/B Testing)**

* **`Query Formulation Time`:** The time delta between the first keystroke and hitting "search." Faster is better
* **Zero-Result Rate / Defect Rate:** How often the autocomplete system recommended a query that led to a "No results found" page. This is a critical failure and must be minimized
* **`Abandonment Rate`:** The user types a prefix, sees the suggestions, and just closes the app without searching

---