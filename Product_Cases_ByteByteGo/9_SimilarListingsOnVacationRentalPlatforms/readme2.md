# ML System Design: Airbnb Similar Listings (Two-Tower Architecture)

> **Interview format:** ~50 minutes | Senior / Staff ML Engineer | FAANG-level

---

## 0. Architecture Choice: Why Two-Tower Over Listing2Vec

This is the first thing to reason through out loud — interviewers want to see that you pick architectures deliberately, not by default.

### The core question

Is the query just a listing ID, or is it richer than that?

> **If "similar" = listings like this listing (pure item-item)** → word2vec / Listing2Vec, single encoder, symmetric problem.
>
> **If "similar" = listings right for this user, in this session, viewing this listing** → Two-tower, because the query is now `(listing + user + session_context)`, which is structurally different from a candidate listing.

At Airbnb's production scale the query side is richer. A solo budget traveller and a family of 6 viewing the same villa should receive different "similar" recommendations. That contextual asymmetry is exactly what two-tower handles.

### When two-tower fits vs. doesn't

| Condition | Better choice |
|---|---|
| Query = item only, no user context | Listing2Vec / single encoder |
| Query = item + user + session context | Two-tower |
| Training signal is purely sequential co-occurrence | Listing2Vec |
| Training signal includes user features + engagement labels | Two-tower |
| Serving requires real-time query personalisation | Two-tower (pre-compute item side only) |
| Symmetric item-item distance is the objective | Listing2Vec |

**Decision:** Use two-tower. The query tower encodes `(current listing + user features + session context)`. The item tower encodes candidate listings. Item embeddings are pre-computed; only the query embedding is computed at request time.

**Retain session co-occurrence signal** by using Listing2Vec-trained embeddings as an *input feature* to both towers, rather than discarding that signal entirely.

---

## 1. Problem Statement & Clarifying Questions (5 min)

**The ask:** Build a "Similar Listings" module on the listing detail page surfacing relevant vacation rentals to the currently-viewing user.

### Clarifying questions

| Question | Why it matters |
|---|---|
| Primary objective: engagement (CTR) or conversion (bookings)? | Determines label definition and loss function |
| Listing corpus size? | Shapes index design and sharding |
| Latency SLA? | Determines online vs. offline computation budget |
| Can we use the user's session history? | Enables query-side personalisation |
| Are trip dates / guest count available in context? | Real-time features for query tower |
| Cold-start requirement for new listings? | Item tower must handle unseen items via features |

### Scope assumptions

- ~10M active listings globally
- ~100M daily search sessions
- Latency SLA: < 100ms p99
- User context available: session clicks, trip dates, guest count, origin country
- Labels: click and booking events from search/listing page sessions

---

## 2. Two-Tower Architecture Overview

```
Query (runtime, personalised)          Item (offline, pre-computed)
─────────────────────────────          ─────────────────────────────
Current listing features               Candidate listing features
+ User features                        (price, type, amenities, location,
+ Session context                       review rating, capacity, ...)
+ Real-time signals
        │                                        │
   Query Tower                            Item Tower
   (MLP / Transformer)                    (MLP)
        │                                        │
   Query embedding q ──── dot product ──── Item embedding e
                             │
                       similarity score
                             │
                    ANN retrieval → re-rank
```

Both towers project to the **same d-dimensional space** (d = 64–256). At serving time, all item embeddings are pre-computed and stored in an ANN index. Only the query embedding is computed on the fly.

---

## 3. Query Tower

The query tower takes as input everything known at the moment the user views a listing.

### Input features

**Current listing features** (what the user is viewing)

| Feature | Type |
|---|---|
| Pre-trained Listing2Vec embedding of query listing | Dense 64d vector |
| Price (normalised) | Float |
| Listing type (entire / private / shared) | Embedding |
| Bedrooms, bathrooms | Int |
| Location (lat/lon → geo hash bucket) | Embedding |
| Review rating | Float |
| Number of reviews | Log-normalised float |

**User features** (from user profile and history)

| Feature | Type |
|---|---|
| Past booked listing embeddings (pooled mean) | Dense 64d vector |
| Price range of past bookings | Float |
| Preferred listing type from history | Embedding |
| Origin country | Embedding |
| Device / browser language | Embedding |
| Account age bucket | Ordinal |

**Session context** (real-time, within current session)

| Feature | Type |
|---|---|
| Listings clicked in this session (mean-pooled embeddings) | Dense 64d vector |
| Price range of clicked listings in session | Float |
| Average rating of clicked listings | Float |
| Trip dates (if entered) — duration, season | Int, Ordinal |
| Number of guests (if entered) | Int |
| Minutes elapsed in session | Float |

### Query tower architecture

```
[all features concatenated → ~400d raw input]
         │
    LayerNorm
         │
    Linear(400 → 256) + ReLU
         │
    Linear(256 → 128) + ReLU
         │
    Linear(128 → d)          ← d = 64–128 output embedding
         │
    L2-normalise             ← required for cosine similarity via dot product
```

Depth (2–3 layers) keeps inference latency low. Layer norm stabilises training with heterogeneous input types.

---

## 4. Item Tower

The item tower encodes each candidate listing into the same embedding space as the query tower.

### Input features

| Feature | Type |
|---|---|
| Listing2Vec embedding (pre-trained from session co-occurrence) | Dense 64d |
| Price bucket | Ordinal embedding |
| Listing type | Embedding |
| Bedrooms, bathrooms, max guests | Int |
| Geo hash (neighbourhood-level) | Embedding |
| Average review rating | Float |
| Number of reviews | Log-normalised |
| Amenity set (WiFi, pool, kitchen, parking, …) | Multi-hot → embedding |
| Cancellation policy | Ordinal |
| Host response rate | Float |
| Photo quality score (from CV model) | Float |
| Booking rate (last 30 days) | Float |

### Item tower architecture

```
[listing features → ~300d raw input]
         │
    LayerNorm
         │
    Linear(300 → 256) + ReLU
         │
    Linear(256 → 128) + ReLU
         │
    Linear(128 → d)
         │
    L2-normalise
```

The item tower is **intentionally simpler** than the query tower. Item embeddings are computed offline (batch inference over all 10M listings), so serving cost is not a concern here.

---

## 5. Training

### Labels

| User action | Label value | Notes |
|---|---|---|
| Impression only | Ignore (or 0 in some setups) | Too noisy |
| Click | Soft positive (0.1–0.3 in utility loss) | Weak intent signal |
| Wishlist | Medium positive (0.5) | Stronger intent |
| Booking | Hard positive (1.0) | Ground truth |
| High-quality booking (>4.9 rating, returned) | Boosted positive (1.2) | Aligns with retention goal |

For binary contrastive training, use **booking as the positive label** and same-session non-booked clicks as negatives.

### Loss Function: Contrastive with In-Batch and Hard Negatives

For a batch of N (query, positive item) pairs, compute similarity scores against all other items in the batch (in-batch negatives) plus explicitly sampled hard negatives.

**Sampled softmax (standard two-tower loss):**

```
L = -log( exp(q · p / τ) / [exp(q · p / τ) + Σⱼ exp(q · nⱼ / τ)] )
```

Where:
- `q` = query embedding (L2-normalised)
- `p` = positive item embedding (booked listing)
- `nⱼ` = negative item embeddings
- `τ` = temperature (learnable or fixed, e.g. 0.05–0.1)

**In-batch negatives:** Other items in the batch serve as implicit negatives. Efficient but can include "false negatives" — popular listings that appear as negatives even though the user might have booked them. Apply **debiasing correction** by down-weighting high-frequency items as negatives.

**Hard negatives (critical for quality):**
- Same city / neighbourhood listings (force model to distinguish fine-grained differences)
- Same price tier listings
- Listings from similar sessions (co-browsed but not booked)

Mix: ~80% in-batch + ~20% explicit hard negatives.

### Training Setup

| Parameter | Value |
|---|---|
| Batch size | 2048–8192 (larger = more in-batch negatives) |
| Embedding dimension d | 64–128 |
| Temperature τ | 0.05 (start), learnable |
| Optimiser | Adam, lr=1e-4 with warmup |
| Epochs | 5–20 (early stop on validation MRR) |
| Negative mining | Mix in-batch + hard negatives (same city) |
| Regularisation | L2 on item tower weights (prevents embedding collapse) |
| Listing2Vec pre-training | Warm-start item tower embeddings |

### Key training pitfalls

**Embedding collapse:** All embeddings converge to similar vectors, dot products become uniform. Detect via monitoring embedding variance. Fix: L2 regularisation, temperature tuning, batch norm.

**False negatives in-batch:** A popular listing appears as a negative for many queries even though users do book it. Fix: subtract `log(freq(item))` from logits (popularity correction).

**Query-item feature leakage:** Do not include future signals in the query tower when training on past sessions. Reconstruct query features as of the *timestamp of the session*, not today.

**Label sparsity:** Bookings are rare (1–3% of sessions). Use curriculum training — start with click labels to warm up, then fine-tune on booking labels.

---

## 6. Cold-Start Handling

### New listings (no Listing2Vec embedding yet)

The item tower still works because it takes explicit features — price, type, location, amenities — not just the Listing2Vec embedding. Set Listing2Vec embedding to the cluster centroid of similar listings (by price + type + geo) as initialisation.

### New users (no booking or session history)

The query tower degrades gracefully: user feature vectors become zero-vectors, session context is empty. The model falls back to query-listing-only features. Explicitly train on such "cold query" examples to ensure the model learns a robust fallback representation.

### Mitigation strategies

| Signal missing | Fallback |
|---|---|
| No user history | Use only current listing features in query tower |
| No session clicks | Use only listing + user profile features |
| New listing (no co-occurrence embedding) | Use geo + type cluster centroid as Listing2Vec init |
| New user + new listing | Pure content similarity via item features |

---

## 7. Serving Architecture

### Three-pipeline system

**Training pipeline (offline, weekly or daily fine-tune):**

```
Session logs (Kafka → S3)
     → Data prep (pair construction, negative mining)
     → Two-tower training (distributed PyTorch, multi-GPU)
     → Model checkpoint → Model registry
```

**Indexing pipeline (offline, after training / daily refresh):**

```
Model checkpoint + all active listings
     → Batch item tower inference (GPU batch, ~10M listings)
     → Item embeddings → FAISS IVF-HNSW index build
     → Index deployed to serving fleet
     → Item embeddings written to KV store (Redis)
```

**Prediction pipeline (online, real-time):**

```
User views listing page
     → Feature fetcher (listing features, user features, session context)
     → Query tower inference (GPU: ~5ms)
     → ANN search on FAISS index: top-K candidates (K=100–500)
     → Availability / price / geo filter
     → Re-ranker (lightweight LightGBM or small MLP)
     → Top-N results rendered
```

### Component responsibilities

| Component | Technology | Latency budget |
|---|---|---|
| Feature fetcher | Redis + Feature Store (Feast) | ~10ms |
| Query tower inference | TorchServe (GPU) | ~5–10ms |
| ANN retrieval (FAISS IVF-HNSW) | C++ service, multiple replicas | ~10–20ms |
| Re-ranker | LightGBM served CPU | ~5ms |
| Total end-to-end | | < 60ms (leaves headroom for p99) |

### ANN index design at 10M listings

Use **FAISS IVF-HNSW** (Inverted File + Hierarchical Navigable Small World):

- Shard index by geography (e.g., continent) to reduce search space per query
- 1024 IVF clusters, probe 64 clusters per query (recall/latency tradeoff)
- Recall target: 95%+ at top-100 retrieval
- Index update: full nightly rebuild + incremental updates for new/changed listings

---

## 8. Re-Ranking Layer

The ANN retrieval stage gets you top-K candidates efficiently. Re-ranking adds a more expensive model that can use features that aren't captured in the embedding distance alone.

### Re-ranker features (beyond embedding similarity)

| Feature | Notes |
|---|---|
| Cosine similarity (query, item embeddings) | Primary retrieval signal, carried forward |
| Price delta (query listing vs. candidate) | Users prefer similar price range |
| Distance (query listing location vs. candidate) | Proximity preference |
| Availability overlap with trip dates | Filtering + ranking signal |
| Guest count fit | Does candidate capacity match? |
| Language match (browser language vs. listing language) | Strong personalisation signal |
| Review rating + count | Quality signal |
| Days since last booking (recency) | Freshness |
| CTR of this listing in similar contexts | Historical engagement signal |

### Re-ranker model

LightGBM (GBDT) trained on same click/booking labels. Lightweight enough for CPU serving at < 5ms.

For higher-quality ranking, replace with a small MLP (2 hidden layers, 128d) — adds ~10ms but improves NDCG.

---

## 9. Evaluation Metrics

### Offline

| Metric | Notes |
|---|---|
| **MRR (Mean Reciprocal Rank)** | Primary: where does the booked listing rank in retrieved candidates? |
| **Hit Rate @ K** (K = 10, 50) | Did the booked listing appear in top-K? |
| **NDCG @ K** | Graded relevance — bookings > wishlists > clicks |
| **ANN Recall @ K** | Does the index actually retrieve the right candidates? |
| **Embedding cosine similarity distribution** | Monitor for embedding collapse |

### Online (A/B test)

| Metric | Direction |
|---|---|
| Booking rate from similar listings module | ↑ |
| CTR on similar listings | ↑ |
| Session depth (pages per session) | ↑ |
| Revenue per session | ↑ |
| Listing page bounce rate | ↓ |

### Guardrails

- Query tower inference p99 latency < 20ms
- Total response p99 < 100ms
- New listing booking rate (cold-start coverage not harmed)
- Diversity index: Gini coefficient of host IDs in top-10 results

---

## 10. Business Rules as Constraints (Secondary Objectives)

| Rule | Implementation |
|---|---|
| Quality promotion | Up-weight high-rated, high-review-count listings in training loss |
| New listing discovery | Score boost for new listings with complete profiles in re-ranker |
| Price range diversity | Post-ranking: ensure top-10 spans ≥ 2 price tiers |
| Geographic diversity | Max 3 listings from same neighbourhood in top-10 |
| Host diversity | Max 2 listings from same host in top-10 |
| Language preference | Query tower language-match feature; strong positive weight in re-ranker |

---

## 11. Monitoring & Debugging

### Embedding health

| Signal | Alert |
|---|---|
| Mean cosine similarity of random pairs | > 0.8 suggests collapse |
| Embedding variance (per dimension) | < 0.01 suggests dead dimensions |
| JS divergence vs. prior week embeddings | > 0.15 triggers retraining |

### Serving health

| Signal | Alert |
|---|---|
| Query tower p99 latency | > 20ms |
| ANN recall rate (spot check sample) | < 90% |
| % requests hitting cold-start fallback | > 25% |
| Re-ranker feature null rate | > 10% for any feature |

### Explainability

SHAP on re-ranker outputs provides per-listing score attribution to share with host-facing tooling. Example:

```
Listing score = 0.72
  + 0.38  embedding_similarity       (most similar to query listing)
  + 0.17  same_price_range
  + 0.14  proximity_km
  + 0.08  high_review_rating
  - 0.05  no_availability_overlap
```

---

## 12. Key Tradeoffs to Discuss

### Single encoder vs. two-tower

| | Single encoder (Listing2Vec) | Two-tower |
|---|---|---|
| Best for | Pure item-item similarity | Item + user + context similarity |
| Training signal | Session co-occurrence | Engagement labels + features |
| Personalisation | Implicit, post-hoc | Native (query tower) |
| Serving complexity | Low (one embedding lookup) | Moderate (query inference + ANN) |
| Cold start | Hard | Easier via item tower features |
| **Choose when** | No user context needed | Real-time personalisation needed |

**In practice:** Use two-tower, but warm-start the item tower from Listing2Vec embeddings — you get the best of both signals.

### Shared vs. separate towers

Using a Siamese network (shared weights, same architecture for query and item) collapses back toward Listing2Vec. Use separate towers to allow the query side to be more complex (deeper, more features) than the item side.

### Retrieval recall vs. latency

FAISS IVF-HNSW parameters (`nprobe`) directly trade recall for latency. Default recommendation: `nprobe=64` for ~95% recall at ~15ms. Increase to `nprobe=128` for ~98% recall at ~25ms. Test empirically at your index size.

---

## 13. Interview Timeline Guide (50 min)

| Time | Section | What to say |
|---|---|---|
| 0–5 min | Clarifying questions | Goals, scale, SLA, user context availability |
| 5–10 min | Architecture choice | Reason through two-tower vs. Listing2Vec; commit to two-tower because query is richer than just a listing ID |
| 10–20 min | Model design | Query tower inputs, item tower inputs, shared embedding space, loss function |
| 20–28 min | Training | Contrastive loss, in-batch negatives, hard negatives, cold start, label design |
| 28–36 min | Serving architecture | 3 pipelines, ANN index, query tower inference latency budget |
| 36–42 min | Evaluation | MRR, NDCG, A/B metrics, guardrails |
| 42–48 min | Tradeoffs | Single vs. two-tower, recall vs. latency, re-ranker depth |
| 48–50 min | Extensions | Bring up Listing2Vec warm-start, real-time embedding updates, diversity constraints |

---

## 14. Common Follow-Up Questions

**Q: Why not just use a single encoder with symmetric similarity (like Listing2Vec)?**
> Single encoder is ideal when query = item. The moment your query is `(listing + user + session)`, the two sides are asymmetric. Two-tower handles this asymmetry natively, and its operational win — pre-compute item embeddings offline — becomes a genuine serving advantage.

**Q: How do you avoid embedding collapse?**
> Monitor mean pairwise cosine similarity across random pairs — if it approaches 1.0, embeddings have collapsed. Fix: L2 regularisation on item tower weights, temperature tuning (lower τ sharpens gradients), larger batch size for more diverse negatives.

**Q: How do you handle false negatives in in-batch training?**
> Apply frequency-based correction: subtract `log(freq(item))` from the logit for each in-batch negative. High-frequency items would otherwise be penalised unfairly.

**Q: How do you keep item embeddings fresh when listing attributes change (price updates, new reviews)?**
> Two tiers: (1) daily full re-indexing for stable features; (2) trigger incremental re-embedding for listings with significant feature changes (>X% price change, new review batch) — push directly to KV store and index without full retraining.

**Q: How do you evaluate whether two-tower is actually better than Listing2Vec for this problem?**
> Run an interleaving experiment or A/B test. Primary offline metric: MRR on held-out booking sessions. Expect two-tower to win on personalised metrics (MRR for logged-in users) but possibly lose on pure item similarity metrics. That outcome would validate the architectural choice.

---

*References: Airbnb Engineering — "Real-time Personalization using Embeddings for Search Ranking at Airbnb" · Google's "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (YouTube two-tower) · Facebook's "Embedding-Based Retrieval in Facebook Search"*