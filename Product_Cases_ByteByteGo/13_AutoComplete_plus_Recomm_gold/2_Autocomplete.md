# FROM GEMINI. Clearing MAJOR doubt

> Comprehension is Compression!

## LTR for AUTOCOMPLETE

Exact dense Two-Tower and DCN+MMoE architecture

> Brutal reality: item reranker budget ~500ms; but **autocomplete system** ~50ms per keystroke

---

### 1. Problem Formulation & Latency Budget

* **Input:** A sequential keystroke prefix (e.g., "har", "harry p"), user context, and environmental context
* **Output:** Top $K$ (usually 5-10) suggested full queries (e.g., "harry potter book 1", "harry potter audiobook")
* **Latency Budget (~50ms total):**
* Network Round Trip: ~15ms
* Stage 1 (Retrieval): ~10ms
* Stage 2 (Reranking): ~20ms
* Post-processing/UI render: ~5ms

### 2. Training Data: The "Prefix-to-Query" Logs

* **Positive Pairs:** *Logs*. If a user types `h` -> `ha` -> `har` -> selects "harry potter", generate *three* positive pairs: `("h", "harry potter")`, `("ha", "harry potter")`, `("har", "harry potter")`
* **Hard Negatives:** Queries that were suggested on the screen but the user ignored and kept typing
* **Random Negatives:** Unrelated queries

* **Downstream Context:** `Did the query result in a "dead end" (no clicks) or a "conversion" (bought/listened to a book)? This will feed our MMoE tasks`

---

### 3. Stage 1: Candidate Generation (Prefix-to-Query Two-Tower)

> Trie (Prefix Tree) data structure

**Two-Tower Retrieval:** Train the model to map short, messy prefixes into the ***SAME EMBEDDING SPACE*** as full, well-formed queries

**Architecture:**

* **Prefix Tower (Query + Context):** `Ingests the current keystrokes.` `Because prefixes are character-heavy and often misspelled, we use a fast Character-CNN or a lightweight Transformer (e.g., a tiny ALBERT) over sub-word tokens.` `Also inject the user's demographic/historical embeddings.` $\rightarrow$ prefix embedding $E_p \in \mathbb{R}^d$
* **Query Tower (well-formed query):** Ingests the full historical queries from our catalog $\rightarrow$ query embedding $E_q \in \mathbb{R}^d$

**Training & Inference:**

* **InfoNCE contrastive loss**: *Push the embedding of "harry p" closer to "harry potter" and further from "hardware"* ``LOL``
* **Offline:** Pre-compute $E_q$ for millions of historical queries $\rightarrow$ index in FAISS - ANN 
* **Online:** Pure semantic ANN on prefixes can sometimes retrieve completely different words that mean the same thing (e.g., typing "fast" might return "quick").` In production, you *must* fuse this Two-Tower ANN retrieval with a standard text-based Trie retrieval to ensure users still see exact literal matches.`

---

### 4. Stage 2: Reranking (DCN-v2 + MMoE for Queries)

Have ~200 candidate queries retrieved

**DCN Inputs (Feature Crosses):**

* **Features:** `Prefix length, candidate query length, edit distance between prefix and query, user's past 5 searched queries, candidate query historical CTR`
* **Explicit Crosses Learned:** 
  * *Prefix Length is Short (<3 chars)* $\times$ *Query is highly popular* (Favor broad hits early on)
* *User prefers Audiobooks* $\times$ *Candidate query contains "mp3" or "audio"* (Personalized intent)

**The MMoE Tasks (Output Gates):**

> `Not all clicked queries are good. Sometimes a user clicks an autocomplete suggestion, realizes the search results are terrible, and reformulates. MMoE prevents us from optimizing for bad clicks`

* **P(Autocomplete CTR)**
* **P(Search Success Rate)** If the user executes this query, they will actually click a book/audiobook in the resulting search page
* **P(Downstream Conversion):** Query ultimately leads to a purchase or a >30-minute listen

**Final Sorting**

```math
Score = P(\text{Select}) \times [w_1 \cdot P(\text{Search Success}) + w_2 \cdot P(\text{Convert})]
```

---

### 5. Architectural Adjustments for 50ms Latency

Cut corners mathematically

1. **Edge Compute / Session Caching:** `User embeddings and historical context vectors *cannot* be fetched per keystroke`. They must be fetched once when the app opens or the search bar is focused, and cached locally or at the nearest edge node.
2. **`Shallow Towers:`** Max 2 dense layers per task
3. **Prefix Caching:** If a user types `h`, `ha`, `har`, the reranker shouldn't re-calculate the entire DCN from scratch. `Caching intermediate computations of previous keystrokes is critical.`
