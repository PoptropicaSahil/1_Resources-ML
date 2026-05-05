# ML System Design: Airbnb Similar Listings

> **Interview format:** ~50 minutes | Senior / Staff ML Engineer | FAANG-level

---

## 1. Problem Statement & Clarifying Questions (5 min)

**The ask:** Build a "Similar Listings" recommendation module that surfaces relevant vacation rental listings to a user currently viewing a specific listing page.

### Key clarifying questions to ask

| Question | Why it matters |
|---|---|
| What is the primary goal - engagement (clicks) or conversion (bookings)? | Drives choice of loss function and labels |
| How many listings exist globally? | Shapes nearest-neighbour index design |
| Is the module on the listing detail page only, or also search results? | Determines latency & serving SLA |
| Do we care about diversity (not showing 10 near-identical listings)? | Adds post-ranking re-rank step |
| Are cold-start listings (new hosts) a hard requirement? | Introduces fallback strategy |
| Real-time or batch embeddings? | Determines infra investment |

### Scope assumptions (state these explicitly)

- **Input:** A currently-viewed listing (the "query listing")
- **Output:** Ranked list of 10–20 visually/semantically similar listings
- **Primary signal:** User browsing sessions (co-click / co-view behavior)
- **Scale:** ~10M active listings worldwide, ~100M daily search sessions
- **Latency SLA:** < 100 ms p99

---

## 2. Why Traditional CF Doesn't Work Here

This is a key conceptual point interviewers probe.

| Dimension | Traditional Collaborative Filtering | Session-Based (this problem) |
|---|---|---|
| User signal used | Long-term user-item ratings | Short browsing sessions (30–60 min) |
| Interest model | Stable user preferences | Rapidly evolving session intent |
| Item representation | User × item matrix | Item co-occurrence in sessions |
| Cold-start for users | Major problem | Less critical (session is the context) |
| Personalization | Per-user | Per-session (current listing as context) |

`**Core insight:** A user browsing a 3BR beachfront villa in Malibu right now has a *very different* intent than their booking history would suggest. The *current listing* is the best proxy for their session intent.`

---

## 3. Model: Listing2Vec (word2vec Analogy)

### The Word2Vec Analogy

```
NLP:        sentence  →  words  →  word2vec  →  word embeddings
Airbnb:     session   →  listings  →  listing2vec  →  listing embeddings
```

Sessions are "sentences." Listings are "words." Co-occurrence in a session ≈ semantic similarity.

### Training Data Construction

**Sessions:** Anonymised sequences of listing IDs from a single user's search session within a time window (e.g., 30 minutes, same location searched).

```
Session example:
[listing_A, listing_B, listing_C, listing_D (booked), listing_E]
```

**Sliding window:** A window of size `w` (typically 5) slides across the session to generate (central, context) pairs.

| Central listing | Context (positive pairs) |
|---|---|
| listing_B | listing_A, listing_C |
| listing_C | listing_B, listing_D |
| listing_D (booked) | listing_C, listing_E |

**Negative sampling:** For each positive (central, context) pair, sample `k` random listings from the entire corpus as negatives. These should be dissimilar listings.

### Standard Loss Function (Click Objective)

For each central listing `c`, context listing `p` (positive), and `k` negative listings `{n₁..nₖ}`:

```
L = -log σ(v_c · v_p) - Σᵢ log σ(-v_c · v_nᵢ)
```

Where:

- `v_c · v_p` = dot product (cosine similarity proxy)
- `σ(x)` = sigmoid function
- This is **binary cross-entropy** treating the problem as "is this a co-clicked pair?"

**Problem:** This loss only optimises for co-click, not booking intent.

### Enhanced Loss: Global Booking Context

**Key innovation:** Add the *eventually-booked listing* as a **global context** that creates positive pairs with *every* listing in the session.

```
For session: [A, B, C, D_booked, E]

Standard pairs: (A,B), (B,C), (C,D), etc.
Global booking pairs added: (A, D_booked), (B, D_booked), (C, D_booked), (E, D_booked)
```

This pulls all listings in the same session *closer* to the booked listing in embedding space.

**Hard negatives from same region:** Add listings from the **same city/region** as extra negatives. This forces the model to learn fine-grained differences (not just "Paris vs. Tokyo") and distinguish similar-sounding but functionally different listings.

```
Enhanced loss:
L = -log σ(v_c · v_p)                     ← standard click context
    - log σ(v_c · v_booked)                ← global booking context
    - Σᵢ log σ(-v_c · v_nᵢ)               ← random negatives
    - Σⱼ log σ(-v_c · v_region_nⱼ)        ← hard regional negatives
```

### Embedding Dimensions & Training Setup

| Parameter | Typical value | Notes |
|---|---|---|
| Embedding dimension | 32–64d | Higher = more expressive, more memory |
| Window size | 5 | ±2 listings around central |
| Negative samples per pair | 5–10 | More = sharper boundaries |
| Training epochs | 3–10 | Until validation loss plateaus |
| Optimiser | SGD or Adam | SGD works well for this scale |
| Batch size | 512–2048 | Depends on GPU memory |

---

## 4. Feature Engineering (Beyond Pure Embeddings)

For a re-ranking layer on top of embedding similarity, additional features can be used:

### Listing Features (for re-ranking)

| Feature | Type | Notes |
|---|---|---|
| Price | Numeric | Absolute & relative to query listing's price |
| Listing type | Categorical | Private room, entire place, shared room |
| Bedrooms/bathrooms | Numeric | Match to query listing capacity |
| Amenities similarity | Jaccard | Pool, kitchen, WiFi overlap |
| Cancellation policy | Ordinal | Flexible → Strict |
| Avg review rating | Numeric | 1–5 stars |
| Number of reviews | Numeric | Cold-start signal |
| Distance from query | Numeric | Geographic proximity (km) |
| Availability overlap | Binary | Does it overlap user's trip dates? |
| Host response rate | Numeric | Quality proxy |

### User/Session Features (for personalization)

| Feature | Type | Notes |
|---|---|---|
| Price range of viewed listings | Numeric | Session-level preference |
| Categories of clicked listings | Categorical | Inferred interest |
| Trip length (if known) | Numeric | Affects price sensitivity |
| Number of guests | Numeric | Affects listing type |
| Device / browser language | Categorical | For language-matched listings |
| Country of origin | Categorical | Cultural preference signal |

---

## 5. Serving Architecture

### Three Pipelines

```
┌─────────────────────────────────────────────────────┐
│ TRAINING PIPELINE (offline, periodic)               │
│  Session logs → Data prep → Listing2Vec training    │
│  → Fine-tuned model artefact                        │
└─────────────────────┬───────────────────────────────┘
                      │ trained model
                      ▼
┌─────────────────────────────────────────────────────┐
│ INDEXING PIPELINE (offline, after training)         │
│  All active listings → Embedding inference          │
│  → ANN Index (FAISS / ScaNN) → Deployed index       │
└─────────────────────┬───────────────────────────────┘
                      │ index + embeddings
                      ▼
┌─────────────────────────────────────────────────────┐
│ PREDICTION PIPELINE (online, real-time)             │
│  Query listing → Embedding fetch → ANN search       │
│  → Candidate pool → Re-ranking → Results            │
└─────────────────────────────────────────────────────┘
```

### Prediction Pipeline Detail

**Step 1 - Embedding fetch service:**

- If query listing has been seen during training → look up pre-computed embedding from key-value store (Redis / DynamoDB)
- If query listing is new (cold start) → fall back to content-based embedding (see §6)

**Step 2 - Nearest neighbour search:**

- Use Approximate Nearest Neighbour (ANN) with FAISS (IVF + HNSW) or ScaNN
- Retrieve top-K candidates (K = 100–500 before re-ranking)
- Filter by availability, price range, geography if query context available

**Step 3 - Re-ranking:**

- Apply lightweight GBM (LightGBM/XGBoost) or MLP re-ranker
- Features: embedding similarity score + listing features + session context
- Re-rank to final top-N (N = 10–20)

### Infrastructure Components

| Component | Technology choice | Reason |
|---|---|---|
| Session data collection | Kafka → S3 | High-throughput stream, cheap storage |
| Training platform | Spark + PyTorch | Distributed data prep + DL training |
| Embedding store | Redis / DynamoDB | Low-latency key-value lookup |
| ANN Index | FAISS (IVF-HNSW) | Best recall/latency tradeoff at 10M scale |
| Re-ranker serving | Torchserve / TF Serving | GPU-accelerated model serving |
| Feature store | Feast / custom | Pre-computed listing features |
| Monitoring | Prometheus + Grafana | Real-time metric tracking |

---

## 6. Cold-Start Handling

New listings have no session co-occurrence data → no learned embedding.

### Strategy 1: Content-Based Embedding Proxy

Train a separate model mapping listing *features* → embedding space, supervised by existing listing embeddings.

```
Input: [price, location, type, amenities, photos_embedding]
  → MLP
  → 32-64d embedding (same space as Listing2Vec)
```

### Strategy 2: Cluster Assignment

Find the nearest cluster centroid in the embedding space using listing content features, and assign the new listing's embedding to that centroid.

### Strategy 3: Promote New Listings via Business Rules

- Override ranking for new high-quality listings (good photos, full profile)
- A/B test their placement at position 3–5 to collect interaction data fast
- Once enough interactions → retrain to get real embedding

---

## 7. Evaluation Metrics

### Offline Metrics

| Metric | Formula | Notes |
|---|---|---|
| **Average rank of booked listing** | Mean position of booked listing in candidate set | Primary offline metric; lower is better |
| **Hit Rate @ K** | % of sessions where booked listing appears in top-K | K = 10, 50, 100 |
| **MRR** | Mean reciprocal rank | Rewards top positions more |
| **NDCG @ K** | Normalised Discounted Cumulative Gain | Handles graded relevance |
| **Embedding quality** | Cosine similarity distribution | Same-price/type listings should cluster |

### Online Metrics (A/B Test)

| Metric | Direction | Notes |
|---|---|---|
| **Booking rate** on similar listings module | ↑ | Primary business KPI |
| **CTR** on similar listings | ↑ | Engagement proxy |
| **Session depth** (pages viewed) | ↑ | Indicates relevance |
| **Bounce rate** from listing page | ↓ | Negative signal |
| **Revenue per session** | ↑ | Ultimate business metric |

### Guardrail Metrics (must not regress)

- Page load latency (p99 < 100ms)
- New listing booking rate (cold start not harmed)
- Inventory diversity index (not showing same hosts repeatedly)

---

## 8. Handling Business Rules (Secondary Objectives)

Following the Airbnb Experiences approach, pure relevance optimisation alone is insufficient.

| Business Rule | Implementation |
|---|---|
| **Quality promotion** | Upweight high-rated listings in loss function during training |
| **Price diversity** | Post-ranking filter: ensure price range spread in top-10 |
| **New listing promotion** | Boost score for new listings with strong content (cold start coverage) |
| **Geographic diversity** | Enforce max 3 listings from same neighbourhood in top-10 |
| **Listing type diversity** | Ensure mix of entire place / private room |
| **Host diversity** | Cap same host at max 2 listings in results |

---

## 9. Monitoring & Explainability

### Feature Attribution for Re-ranker

Use SHAP values on the re-ranking GBM to understand per-listing score:

```
Similar listing score decomposition:
  + 0.42  embedding_cosine_similarity
  + 0.21  same_price_range
  + 0.15  proximity_km
  + 0.09  similar_amenities
  - 0.08  no_availability_overlap
  - 0.03  lower_review_count
```

### Data Drift Monitoring

| Signal | Alert threshold |
|---|---|
| Embedding distribution shift (JS divergence) | > 0.1 vs. last week |
| CTR on similar listings module | Drop > 5% vs. 7-day average |
| % queries hitting cold-start fallback | > 20% (suggests index staleness) |
| ANN index recall rate | < 90% (index degradation) |
| Model serving latency p99 | > 100ms |

---

## 10. Scaling Considerations

### Training Scale

| Data scale | Recommended approach |
|---|---|
| < 1M sessions | Single-machine PyTorch, full-batch updates |
| 1M – 100M sessions | Distributed training (PyTorch DDP on multi-GPU) |
| > 100M sessions | Asynchronous SGD, parameter server architecture |

### Serving Scale (10M listings, 100M sessions/day)

- **Index update frequency:** Daily full re-index + incremental embedding updates for new/changed listings
- **Index sharding:** Shard by geography (e.g., continent) to reduce search space
- **Caching:** Cache top-N similar listings per popular listing in Redis (invalidate on listing update)
- **Read replicas:** Multiple ANN index replicas behind load balancer for horizontal scale

---

## 11. Tradeoffs & Discussion Points

### Embedding Approach vs. Feature-Based Re-ranker

| | Pure embeddings | Hybrid (embedding + re-ranker) |
|---|---|---|
| **Latency** | Lower (single lookup + ANN) | Higher (two-stage) |
| **Freshness** | Batch (hours lag) | Re-ranker can use real-time features |
| **Interpretability** | Black box | Re-ranker offers SHAP attribution |
| **Cold start** | Hard | Easier (content features available) |
| **Accuracy** | Good for surface-level similarity | Better personalised ranking |

**Recommendation:** Ship pure embedding retrieval first (simpler, faster), add re-ranker in v2.

### Real-Time vs. Batch Embeddings

| | Batch (offline) | Real-time (online) |
|---|---|---|
| **Latency** | ~1ms (cache lookup) | ~20-50ms (inference) |
| **Freshness** | Stale by hours/days | Up-to-date |
| **Infrastructure cost** | Low | High (GPU serving) |
| **Good for** | Established listings | New listings, price changes |

**Recommendation:** Batch for established listings, real-time content-based for cold-start.

---

## 12. Interview Timeline Guide (50 min)

| Time | Section | Key points to cover |
|---|---|---|
| 0–5 min | Problem clarification | Clarify goals, scale, SLA, scope |
| 5–10 min | Why session-based | Traditional CF fails; session = intent |
| 10–20 min | Listing2Vec model | word2vec analogy, loss function, booking context, hard negatives |
| 20–30 min | Serving architecture | 3 pipelines, ANN index, cold start |
| 30–38 min | Evaluation | Offline (avg rank, NDCG) + online (A/B) |
| 38–44 min | Scaling & business rules | Sharding, diversity, quality promotion |
| 44–50 min | Tradeoffs & follow-ups | Embeddings vs. re-ranker, real-time vs. batch |

---

## 13. Common Follow-Up Questions

**Q: How do you handle the position bias in training data?**
> Listings shown at the top of search results get clicked more regardless of quality. Use Inverse Propensity Scoring (IPS) - downweight clicks on top positions, upweight clicks on lower positions.

**Q: How would you personalise similar listings per user?**
> Two approaches: (1) train user embeddings alongside listing embeddings (like AirBnb's real-world2vec), then bias ANN retrieval towards listings similar to user's past bookings. (2) Add user features to the re-ranker.

**Q: How do you keep the index fresh without full daily retraining?**
> Use a two-tier system: (1) incremental fine-tuning on last 24h session data (fast, cheap), (2) weekly full retraining. New listings get content-based embeddings injected into the live index without retraining.

**Q: What if a user is viewing a listing but hasn't searched with dates - how do you filter candidates?**
> Relax the availability filter and rank by similarity only. Surface a "check availability" CTA instead of blocking the recommendations.

**Q: How do you prevent filter bubbles (only showing similar-priced listings)?**
> Enforce diversity constraints post-ranking: e.g., top-10 must span at least 2 price tiers and 2 listing types. Can also use MMR (Maximal Marginal Relevance) to balance similarity and diversity.

---

*Reference: Airbnb Engineering Blog - "Listing Embeddings in Search Ranking" and "ML-Powered Search Ranking of Airbnb Experiences"*
