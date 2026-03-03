# ML System Design Interview: Similar Listings on Vacation Rental Platforms

**Role:** Senior Data Scientist | **Level:** FAANG L6/L7 | **Duration:** 45–60 min

---

## Phase 1: Problem Formulation & Requirement Clarification (~8 min)

**Interviewer:** Let's say you're working at a vacation rental platform — think Airbnb or Vrbo. When a user is viewing a specific listing, we want to show a "Similar Listings" carousel below the listing details. Design the ML system that powers this feature. Take a moment to think about it, and start wherever you'd like.

**Candidate:** Great problem. Before I jump into the architecture, let me clarify the scope and constraints so we're aligned on what "similar" means in this context and what the business goals are.

**First — what surface is this appearing on?** Is this the listing detail page (LDP) only, or also search results, emails, push notifications?

**Interviewer:** Let's focus on the listing detail page. The user is looking at Listing A, and we show K similar listings underneath.

**Candidate:** Perfect. And what's the primary business objective? Are we optimizing for:

- **Engagement** — clicks on similar listings to keep users browsing?
- **Conversion** — bookings that originate from this carousel?
- **Session depth** — reducing bounce rate from listing pages?

**Interviewer:** Primarily conversion — we want users who might not book *this* listing to find one they will book. But engagement is a secondary metric.

**Candidate:** Understood. Let me also pin down a few constraints:

1. **Scale.** How many active listings are we talking about? I'll assume on the order of **7M+ listings** globally (Airbnb-scale).
2. **Latency.** Since this is on the listing detail page, the carousel should load as part of or shortly after the page — I'd target **< 200ms p99** for the ML inference path.
3. **K.** We're showing maybe **12–20 similar listings** in the carousel, but generating a candidate set that's much larger.
4. **Two-sided marketplace.** "Similar" needs to respect the guest's perspective (I want something like this listing) but also host availability and acceptance patterns.
5. **Cold start.** New listings with no interaction history need to appear in similar listing results too.

**Interviewer:** Those are good assumptions. Let's proceed with those.

**Candidate:** Let me also define what "similar" means precisely, because this is the crux. Two listings can be similar along multiple axes:

| Dimension | Example |
|-----------|---------|
| **Location** | Same neighborhood or comparable area |
| **Price** | Similar nightly rate for similar capacity |
| **Listing type** | Entire home vs. private room vs. shared |
| **Amenities** | Pool, WiFi, parking, kitchen |
| **Style/aesthetic** | Modern loft vs. rustic cabin (visual similarity) |
| **Guest behavior** | Users who viewed/booked A also viewed/booked B |
| **Availability** | Similar open date ranges |

The key insight is that **behavioral similarity often captures latent dimensions** that content features miss — like "vibe" or "trustworthiness signals" from review quality. So I'll design a system that combines both content-based and behavioral signals.

---

## Phase 2: Metrics Design (~5 min)

**Interviewer:** Before we get to the model, walk me through how you'd measure success.

**Candidate:** I'll structure metrics at three levels:

### Offline Metrics (Model Quality)

- **Recall@K:** Of the listings a user eventually books or clicks in a session, what fraction appeared in our top-K similar listings? This directly measures candidate quality.
- **NDCG@K (Normalized Discounted Cumulative Gain):** Measures ranking quality — are the best similar listings placed highest?
- **MRR (Mean Reciprocal Rank):** How high does the first "relevant" (clicked/booked) listing appear?
- **Hit Rate@K:** Binary — did at least one relevant listing appear in the top K?

For embeddings specifically:

- **Embedding coverage:** What percentage of active listings have a valid embedding? Target > 99%.
- **Intra-cluster cosine similarity:** For listings we know are similar (same neighborhood, same type), are embeddings close?

### Online Metrics (A/B Test)

- **Primary: Booking Conversion Rate from Carousel** — `bookings originating from similar listings / impressions of carousel`
- **Secondary:**
  - Click-Through Rate (CTR) on the carousel
  - Session depth after viewing the carousel (pages/session)
  - Bounce rate reduction on the listing detail page
  - Time-to-book (should decrease if we surface better alternatives)

### Guardrail Metrics

- **Revenue per session** (shouldn't drop)
- **Host-side fairness** — similar listings shouldn't always funnel traffic to the same top-rated listings; we need exposure equity
- **Listing diversity in carousel** — we don't want 12 nearly identical listings; some variety in price/type/neighborhood helps

**Interviewer:** Good. What about the tension between CTR and booking rate?

**Candidate:** Great callout. CTR can be gamed by clickbait-y thumbnails or misleadingly low prices. I'd weight **booking rate as the north star** and use CTR as a diagnostic. If CTR goes up but bookings don't, we're showing attractive-looking but mismatched listings — that's worse than the baseline.

Also, I'd track **downstream host rejection rate** — if guests inquire on a similar listing but the host rejects them at a higher rate, our similarity model might be matching listings but not matching guest-host compatibility.

---

## Phase 3: High-Level Architecture (~7 min)

**Candidate:** The standard architecture for recommendation at this scale follows a **multi-stage funnel**:

```
┌─────────────────────────────────────────────────────────────┐
│                   USER VIEWS LISTING A                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   STAGE 1: RETRIEVAL   │  ~1000 candidates
              │  (Candidate Generation)│  Latency: < 50ms
              │                        │
              │  • ANN on Embeddings   │
              │  • Geo-based filter    │
              │  • Co-engagement graph │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   STAGE 2: PRE-FILTER  │  ~200 candidates
              │  (Business Rules)      │  Latency: < 10ms
              │                        │
              │  • Availability check  │
              │  • Host active status  │
              │  • Blocked listings    │
              │  • Minimum review score│
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   STAGE 3: SCORING     │  ~200 → ranked
              │  (Ranking Model)       │  Latency: < 80ms
              │                        │
              │  • Pointwise / Pairwise│
              │    GBDT or DNN ranker  │
              │  • Features: listing,  │
              │    user, cross features│
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  STAGE 4: RE-RANKING   │  Top K (12-20)
              │  (Post-processing)     │  Latency: < 10ms
              │                        │
              │  • Diversity injection  │
              │  • Price spread control│
              │  • Dedup (same host)   │
              │  • Explore/exploit     │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │    SIMILAR LISTINGS    │
              │      CAROUSEL (K)      │
              └────────────────────────┘
```

**Interviewer:** Walk me through the reasoning for this funnel shape.

**Candidate:** This is the classic retrieval-then-ranking paradigm, and the reason is computational budget.

With 7M listings, we can't run a heavy neural ranker over all of them per request. So:

- **Retrieval** uses cheap operations (ANN vector search) to narrow 7M → ~1000 candidates. The goal is **high recall** — don't miss good candidates. Precision can be low.
- **Scoring** uses an expensive model on ~200 candidates. The goal is **high precision in ranking** — put the best at the top.
- **Re-ranking** applies business logic and diversity constraints that are hard to encode in a model loss function.

The total latency budget of 200ms is split roughly: 50ms retrieval + 10ms filtering + 80ms scoring + 10ms re-ranking + 50ms network/serialization overhead.

---

## Phase 4: Retrieval — Embedding Design (Deep Dive) (~12 min)

**Interviewer:** Let's go deep on the retrieval stage. How would you build the embeddings?

**Candidate:** This is the core of the system. I'll describe two complementary embedding approaches, following what Airbnb published in their KDD 2018 paper.

### Approach 1: Listing Embeddings from Click Sessions (Short-Term)

**Intuition:** Treat a user's click session like a "sentence" in NLP. Each listing the user clicks is a "word." Listings that appear in similar contexts (same sessions) should have similar embeddings — just like Word2Vec.

**Training data construction:**

From search logs, extract click sessions:

```
Session 1: [L_23, L_105, L_42, L_87, L_booked]
Session 2: [L_5, L_91, L_32]           ← no booking
Session 3: [L_412, L_7, L_200, L_booked]
```

A session is a temporally contiguous sequence of listing clicks by the same user. We apply rules: sessions break after 30 min of inactivity, and we discard sessions with only 1 click.

**Model: Modified Skip-Gram with Negative Sampling**

The standard Word2Vec skip-gram objective maximizes the log-probability of observing context listings given a center listing. For a session `s = (l₁, l₂, ..., l_M)`:

$$\mathcal{L} = \sum_{s \in S} \sum_{i=1}^{|s|} \left( \sum_{\substack{j=1 \\ j \neq i}}^{|s|} \log \frac{1}{1 + e^{-\mathbf{v}_{l_i}^{\top} \mathbf{v}'_{l_j}}} + \sum_{k=1}^{n_{\text{neg}}} \mathbb{E}_{l_k \sim P_n} \left[ \log \frac{1}{1 + e^{\mathbf{v}_{l_i}^{\top} \mathbf{v}'_{l_k}}} \right] \right)
$$

Where:
- $\mathbf{v}_{l_i} \in \mathbb{R}^d$ is the embedding of listing $l_i$ (center)
- $\mathbf{v}'_{l_j}$ is the context embedding of listing $l_j$
- $P_n$ is the negative sampling distribution
- $n_{\text{neg}}$ is the number of negative samples (typically 5–10)

**Airbnb's three key modifications:**

**Modification 1 — Booked listing as global context:**

For sessions that end in a booking, the booked listing $l_b$ is added as a "global context" that pairs with every listing in the session, not just those within the sliding window. The intuition: if a user booked listing B after viewing A, C, D — then A, C, D are all contextually related to B, regardless of window distance.

The additional term added to the loss:

$$+ \log \frac{1}{1 + e^{-\mathbf{v}_{l_i}^{\top} \mathbf{v}'_{l_b}}}$$

This upweights the booking signal, which is the strongest positive signal we have.

**Modification 2 — Same-market negative sampling:**

Standard Word2Vec samples negatives from the global listing distribution $P_n(l) \propto f(l)^{3/4}$ (frequency-dampened unigram). But in a marketplace, most random negatives are geographically irrelevant — a listing in Tokyo is trivially different from one in Paris.

Airbnb adds **hard negatives from the same market** (city/region). For each positive pair, they add negatives sampled from listings in the same city as the center listing. This forces the model to learn *within-market* distinctions (price tier, neighborhood, style) rather than just "these are in the same country."

$$+ \sum_{k=1}^{n_{\text{hard}}} \mathbb{E}_{l_k \sim P_{\text{market}(l_i)}} \left[ \log \frac{1}{1 + e^{\mathbf{v}_{l_i}^{\top} \mathbf{v}'_{l_k}}} \right]
$$

**Modification 3 — Congregated search (adapting to marketplace search):**

In vacation rental search, a session tends to cluster in one market/city. Rather than treating each click as a flat "word," they leverage the fact that users compare similar listings in the same locale.

**Embedding dimensionality:** d = 32. This is surprisingly low but sufficient — Airbnb found 32 floats captured location, price, type, architecture, and style. Higher dimensions didn't improve recall significantly but increased storage and ANN latency.

**Training details:**
- ~800M click sessions
- SGD with learning rate 0.025 (linearly decayed)
- Window size: entire session (not fixed context window) — because sessions are short (avg ~5–6 clicks)
- Training: distributed across machines, daily retraining

### Approach 2: User-Type & Listing-Type Embeddings (Long-Term)

**Problem:** Booking data is much sparser than click data. Most users book 1–2 times per year. We can't learn per-user or per-listing embeddings from booking sequences.

**Solution:** Learn embeddings at the **type level** rather than the ID level.

Map each listing to a **listing type** using a rule-based bucketing:

```
listing_type = f(country, listing_category, capacity_bucket, 
                 price_bucket, num_rooms, has_reviews)
```

For example: `US_entire_home_4guests_$150-200_2bed_reviewed`

Similarly, map each user to a **user type:**

```
user_type = f(country, device_type, is_superguest, 
              avg_price_booked, language, booking_count_bucket)
```

Now construct booking sessions per user — ordered sequence of listing *types* they booked over time. Train embeddings in the **same vector space** as user types — so we can compute `cosine(user_type_embedding, listing_type_embedding)` for cross-entity similarity.

This gives us a long-term preference signal: a user who historically books "modern 2BR apartments in European cities" will have their user-type embedding close to those listing-type embeddings.

### Cold-Start Embedding Strategy

**Interviewer:** How do you handle new listings with no click history?

**Candidate:** Three approaches, in order of preference:

1. **Geo-neighbor averaging:** Find the 3 nearest existing listings within 10 miles that share the same listing type (category, price bucket) and average their embeddings. This is simple and effective — Airbnb uses this in production.

2. **Content-based projection network:** Train a small MLP that maps listing features (price, location, amenities, type, photo embeddings) → listing embedding space. This is a "warm-start" model trained on listings that have established embeddings.

   $$\hat{\mathbf{v}}_l = \text{MLP}(\text{concat}[\mathbf{x}_{\text{price}}, \mathbf{x}_{\text{geo}}, \mathbf{x}_{\text{amenities}}, \mathbf{x}_{\text{photo}}])$$
   
   Loss: $\|\hat{\mathbf{v}}_l - \mathbf{v}_l^*\|^2$ where $\mathbf{v}_l^*$ is the learned embedding from click sessions.

3. **Meta-learning approach:** For truly new listings, use the average embedding from their listing-type bucket as a prior, then update as interaction data accumulates.

### ANN Index for Retrieval

Given 7M listings with 32-dim embeddings, we need an efficient nearest neighbor search.

**Choice: HNSW (Hierarchical Navigable Small World) via FAISS or ScaNN.**

| Approach | Recall@100 | QPS | Memory |
|----------|-----------|-----|--------|
| Brute force | 100% | ~500 | Low |
| IVF-PQ | ~92% | ~10K | Low |
| HNSW | ~98% | ~5K | High |
| ScaNN | ~97% | ~15K | Medium |

I'd use **HNSW** for the primary index (best recall-latency tradeoff) and **IVF-PQ (Product Quantization)** as a fallback for memory-constrained environments.

**Key implementation detail:** We partition the ANN index by **market (city/region)**. When retrieving similar listings for a listing in Paris, we primarily search the Paris index (+ a small fraction from "nearby" markets like Versailles, Fontainebleau).

This has two benefits:
1. **Smaller index per shard** → faster search
2. **More relevant candidates** → users viewing Paris listings rarely want something in Berlin

We also maintain a smaller **global index** for users whose behavior suggests cross-market interest (e.g., they've been searching in multiple cities).

---

## Phase 5: Feature Engineering & Ranking Model (~10 min)

**Interviewer:** Now let's talk about the scoring/ranking stage. What model and features would you use?

**Candidate:** The ranking model takes ~200 candidate listings from retrieval and scores each with a predicted probability of engagement (click) or conversion (booking).

### Feature Categories

**Listing Features (of candidate listing B):**

| Feature | Type | Notes |
|---------|------|-------|
| Price per night | Continuous | Log-transformed |
| Number of reviews | Continuous | |
| Average review score | Continuous | |
| Superhost status | Binary | |
| Instant book enabled | Binary | |
| Listing type (entire/private/shared) | Categorical | One-hot |
| Number of bedrooms/bathrooms | Continuous | |
| Amenity set | Multi-hot vector | Top 50 amenities |
| Photo quality score | Continuous | Pre-computed CNN score |
| Days since last booking | Continuous | Freshness signal |
| Cancellation policy | Categorical | Flexible/moderate/strict |
| Response rate | Continuous | |

**Seed Listing Features (of listing A being viewed):**

Same feature set as above, representing the listing the user is currently viewing.

**Cross Features (relationship between A and B):**

These are the most powerful features for "similarity."

| Feature | Formula / Description |
|---------|----------------------|
| Embedding cosine similarity | $\cos(\mathbf{v}_A, \mathbf{v}_B)$ |
| Price ratio | $\text{price}_B / \text{price}_A$ |
| Geographic distance | Haversine distance (km) |
| Same neighborhood | Binary |
| Same listing type | Binary |
| Review score difference | $\text{review}_B - \text{review}_A$ |
| Amenity overlap (Jaccard) | $\|A_{\text{amen}} \cap B_{\text{amen}}\| / \|A_{\text{amen}} \cup B_{\text{amen}}\|$ |
| Photo style similarity | Cosine sim of CNN feature vectors |
| Co-click rate | # users who clicked both A & B / # who clicked A |

**User Features (of the person viewing):**

| Feature | Notes |
|---------|-------|
| User-type embedding | Long-term preference |
| Search filter preferences | Price range, dates, guests |
| Device type | Mobile vs. desktop (affects UI) |
| User history embedding | Average embedding of recently clicked listings |
| Days until trip | Urgency signal |
| Past booking count | Experience level |

**Context Features:**

| Feature | Notes |
|---------|-------|
| Time of day | Browsing patterns differ |
| Day of week | Weekend browsers behave differently |
| Position in session | First listing viewed vs. 10th |

### Model Choice

**Interviewer:** What model architecture?

**Candidate:** I'd use a **GBDT (Gradient Boosted Decision Trees)** — specifically **LambdaMART** — as the starting point, then consider a DNN ranker for V2.

**Why GBDT first:**

1. **Handles heterogeneous features** (categorical, continuous, embeddings) without extensive preprocessing
2. **Interpretable feature importances** — critical for debugging in production
3. **Fast training and inference** — scoring 200 candidates in < 80ms is straightforward
4. **Robust to feature scale differences** — no normalization needed
5. **Industry standard** at Airbnb, Booking.com, and most marketplace rankers for v1

**Why LambdaMART specifically:**

This is a listwise learning-to-rank approach. Rather than treating ranking as pointwise classification (predict P(click) independently), LambdaMART optimizes NDCG directly by computing "lambda gradients" that measure how much swapping two items in the ranked list would improve NDCG.

The lambda gradient for a pair of documents $(i, j)$ where $i$ is ranked higher than $j$:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta \text{NDCG}_{ij}|$$

Where:
- $s_i, s_j$ are the model scores
- $\Delta \text{NDCG}_{ij}$ is the change in NDCG if $i$ and $j$ were swapped
- $\sigma$ is a scaling parameter

This means the model focuses gradient on pairs where a swap would significantly improve the ranking — rather than treating all pairs equally.

**Label generation for training:**

We need relevance labels. I'd use a multi-grade scheme:

| Label | Signal | Weight |
|-------|--------|--------|
| 0 | Impression only (no click) | — |
| 1 | Click on listing B from carousel | Low |
| 2 | Click → viewed listing B for > 30s | Medium |
| 3 | Click → added to wishlist or contacted host | High |
| 4 | Click → booked listing B | Highest |

These graded labels let NDCG weight bookings much higher than casual clicks.

### V2: Two-Tower DNN for Deeper Personalization

For a more advanced iteration, I'd move to a **two-tower architecture** for the retrieval stage and a **cross-attention DNN** for ranking.

```
         Seed Listing A              Candidate Listing B
              │                             │
    ┌─────────┴─────────┐        ┌─────────┴─────────┐
    │   Listing Tower   │        │   Listing Tower    │
    │                   │        │   (shared weights)  │
    │ [features → MLP]  │        │ [features → MLP]   │
    └─────────┬─────────┘        └─────────┬──────────┘
              │                             │
         embedding_A                   embedding_B
              │                             │
              └──────────┬──────────────────┘
                         │
                    dot product / cosine
                         │
                    similarity score
```

The two-tower model has the advantage that we can **pre-compute all listing tower outputs** offline and serve retrieval via ANN. We only need to compute the seed tower online.

But for the ranking stage, I'd use a **cross-network** where seed and candidate features interact:

$$\text{score} = \text{MLP}([\mathbf{h}_A; \mathbf{h}_B; \mathbf{h}_A \odot \mathbf{h}_B; \mathbf{h}_{\text{user}}; \mathbf{x}_{\text{cross}}])$$

Where $\odot$ is element-wise product (captures multiplicative interactions) and $\mathbf{x}_{\text{cross}}$ are hand-crafted cross features.

---

## Phase 6: Re-Ranking & Business Logic (~5 min)

**Interviewer:** Tell me about the re-ranking stage.

**Candidate:** The re-ranker applies constraints that are hard to encode in a differentiable loss function:

**1. Diversity (MMR — Maximal Marginal Relevance):**

We don't want 12 listings from the same building. MMR re-ranks by balancing relevance and diversity:

$$\text{MMR}(l_i) = \lambda \cdot \text{score}(l_i) - (1 - \lambda) \cdot \max_{l_j \in S_{\text{selected}}} \text{sim}(l_i, l_j)$$

Where $S_{\text{selected}}$ is the set of already-selected listings and $\lambda \approx 0.7$ trades off relevance vs. diversity. We greedily select listings one by one.

Diversity dimensions to penalize:
- Same host (hard filter — max 2 per host)
- Same building/complex
- Same exact price point
- Too similar photo thumbnails

**2. Price Spread:**

Ensure the carousel includes options at slightly higher and lower price points than listing A. Users viewing a $200/night listing should see some $150 and some $250 options — this helps anchor value perception and captures users with flexible budgets.

**3. Availability Boost:**

Listings with open availability for the user's searched dates (if known) get a score boost. If the user hasn't specified dates, we use the next popular booking window for the market.

**4. Explore/Exploit:**

Reserve 1–2 of the K slots for exploration — show a random or low-confidence listing to collect signal. This helps:
- New listings get exposure (cold-start)
- Overcome popularity bias
- Continuously improve the model

We use an **epsilon-greedy** strategy with $\epsilon \approx 0.1$, or a Thompson sampling approach where we sample from the posterior of our score prediction.

---

## Phase 7: Training Pipeline & Serving (~8 min)

**Interviewer:** Walk me through the training pipeline and serving infrastructure.

**Candidate:**

### Training Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING PIPELINE                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────────┐     │
│  │  Search  │───▶│  Session     │───▶│  Embedding        │     │
│  │  Logs    │    │  Constructor │    │  Training (W2V)   │     │
│  └─────────┘    └──────────────┘    └────────┬──────────┘     │
│                                              │                 │
│  ┌─────────┐    ┌──────────────┐             ▼                │
│  │ Booking  │───▶│  Label       │    ┌───────────────────┐    │
│  │ Logs     │    │  Generator   │    │  ANN Index Build  │    │
│  └─────────┘    └──────┬───────┘    │  (HNSW / FAISS)   │    │
│                        │            └───────────────────┘     │
│  ┌─────────┐           ▼                                      │
│  │ Feature  │    ┌──────────────┐                             │
│  │ Store    │───▶│  Ranker      │                             │
│  └─────────┘    │  Training    │                             │
│                  │  (LambdaMART)│                             │
│                  └──────────────┘                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Embedding retraining cadence:** Daily. Click sessions accumulate fast, and we want to capture seasonal shifts (e.g., ski season vs. beach season changes what's "similar").

**Ranker retraining cadence:** Weekly with daily feature refresh. The ranking model is more stable, but features like "days since last booking" and "current review score" need daily updates in the feature store.

**Data splitting:** Temporal split only — train on weeks 1–8, validate on week 9, test on week 10. Never random split, because of temporal leakage (user behavior patterns leak across time).

### Serving Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     ONLINE SERVING PATH                          │
│                                                                  │
│  User views Listing A                                            │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐  seed embedding   ┌─────────────────────┐     │
│  │  Embedding    │─────────────────▶│   ANN Service        │     │
│  │  Lookup       │                  │   (FAISS/ScaNN)      │     │
│  │  (Redis/      │                  │                      │     │
│  │   Memcached)  │                  │   Returns top-1000   │     │
│  └──────────────┘                   │   listing IDs        │     │
│                                     └──────────┬────────────┘    │
│                                                │                 │
│  ┌──────────────┐                              ▼                 │
│  │  Feature      │  ◀───── feature request ── Filter             │
│  │  Store        │  ─────▶ features ────────▶ (~200)             │
│  │  (Redis +     │                              │                │
│  │   DynamoDB)   │                              ▼                │
│  └──────────────┘                        ┌──────────────┐        │
│                                          │  Ranking      │        │
│                                          │  Service      │        │
│  ┌──────────────┐                        │  (LambdaMART) │        │
│  │  User Profile │ ──── user features ──▶│              │        │
│  │  Service      │                        └──────┬───────┘        │
│  └──────────────┘                               │                │
│                                                  ▼                │
│                                          ┌──────────────┐        │
│                                          │  Re-Ranker    │        │
│                                          │  + MMR        │        │
│                                          └──────┬───────┘        │
│                                                 │                │
│                                                 ▼                │
│                                          Top K Listings          │
│                                          to Frontend             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key infrastructure decisions:**

1. **Embedding storage:** All 7M listing embeddings (32-dim float32) = ~900MB. Fits entirely in Redis. Lookup is O(1) — sub-millisecond.

2. **ANN service:** Deployed as a stateful microservice with HNSW index loaded in memory. Sharded by market. We use **index replicas** for redundancy and horizontal scaling during peak traffic.

3. **Feature store:** Two tiers:
   - **Hot path (Redis):** Pre-computed features for the top 500K most-viewed listings. < 1ms latency.
   - **Warm path (DynamoDB):** Full feature store for all listings. < 10ms latency.
   
4. **Ranking service:** Stateless, horizontally scalable. Model loaded as a serialized LightGBM or XGBoost model. Scoring 200 candidates takes ~10–30ms on a single CPU core.

5. **Caching:** The similar listings result for Listing A doesn't change minute-to-minute. We cache results with a **TTL of 1 hour** (invalidated on listing updates). This dramatically reduces compute during peak hours — a popular listing might be viewed thousands of times per hour.

**Interviewer:** What about the latency breakdown?

**Candidate:**

| Stage | Target (p99) | Notes |
|-------|-------------|-------|
| Embedding lookup | < 1ms | Redis GET |
| ANN search | 20–40ms | HNSW with ef_search=100 |
| Availability filter | 5–10ms | Parallel lookups |
| Feature assembly | 10–20ms | Batch Redis MGET |
| Ranking model | 10–30ms | LightGBM predict_proba |
| Re-ranking (MMR) | 2–5ms | Greedy selection |
| Serialization + network | 10–20ms | Protobuf |
| **Total** | **~100–130ms** | Well within 200ms budget |

With caching hit rates around 60–70% for popular listings, the effective average latency is much lower.

---

## Phase 8: Handling Edge Cases & Production Concerns (~5 min)

**Interviewer:** What are the main edge cases and production challenges?

**Candidate:**

### Position Bias

The carousel has a strong position bias — users click the first few items disproportionately. If we naïvely train on click data, the model learns "things shown in position 1 are good" — a self-reinforcing loop.

**Mitigation:**
- **Inverse Propensity Weighting (IPW):** Weight each training example by $1/P(\text{position}_k)$, where $P$ is the estimated CTR purely due to position. This debiases the gradient.
- **Position as a feature during training, removed during inference:** Train the ranker with position as an input feature, then set position = 0 (or mean) during serving. This lets the model "factor out" position effects.

### Feedback Loops (Popularity Bias)

Popular listings get shown more → get more clicks → rank higher → get shown even more. This starves new and niche listings.

**Mitigation:**
- Explore/exploit slots (discussed above)
- **Popularity penalty:** Add a feature $\log(\text{impression\_count})$ with a learned negative weight
- **Counterfactual evaluation:** Use logged bandit data to estimate what *would have happened* if we'd shown different listings (IPS estimator)

### Seasonality

"Similar" shifts with seasons. A beachfront listing's similar listings in summer should include other beach properties; in winter, maybe cozy mountain lodges attract the same users.

**Mitigation:** Daily embedding retraining naturally captures this, since recent sessions dominate the training data. We can also add seasonal features to the ranker (month, holiday_flag, local_event_flag).

### A/B Testing Considerations

- **Randomization unit:** User-level (not session-level) to avoid within-user contamination.
- **Novelty effect:** Run the test for at least 2 weeks — initial CTR lift often decays.
- **Network effects:** If we improve one listing's carousel, it might cannibalize traffic from another listing. We need to monitor **platform-level booking rate**, not just carousel-level conversion.

---

## Phase 9: Iterative Improvements & Advanced Topics (~5 min)

**Interviewer:** If you had more time and resources, what would you add next?

**Candidate:** Here's my V2/V3 roadmap, ordered by expected impact:

**1. Multi-Modal Embeddings (High Impact)**

Incorporate listing photos into the embedding. Train a **visual similarity model** (ResNet/EfficientNet backbone → 128-dim visual embedding). A rustic cabin's photos encode "vibe" that text features can't capture.

Combine: $\mathbf{v}_{\text{final}} = \text{MLP}(\text{concat}[\mathbf{v}_{\text{behavioral}}, \mathbf{v}_{\text{visual}}, \mathbf{v}_{\text{text}}])$

Where $\mathbf{v}_{\text{text}}$ comes from an encoding of the listing description (BERT or sentence-transformers).

**2. Graph Neural Networks (Medium Impact)**

Build a listing graph where edges represent co-click or co-booking relationships. Run GraphSAGE to learn embeddings that capture multi-hop neighborhood structure — listing A is similar to listing C because users who liked A also liked B, and users who liked B also liked C.

**3. Real-Time Session Personalization (High Impact)**

Maintain a running embedding of the user's current session:

$$\mathbf{u}_{\text{session}} = \frac{1}{|H_c|} \sum_{l \in H_c} \mathbf{v}_l - \frac{1}{|H_s|} \sum_{l \in H_s} \mathbf{v}_l$$

Where $H_c$ is the set of clicked listings and $H_s$ is the set of skipped listings. This session vector captures real-time preference shifts — if the user started looking at apartments but switched to houses, the similar listings should adapt immediately.

This is streamed via Kafka and updated per-click.

**4. Contextual Bandits for Exploration**

Replace epsilon-greedy with a LinUCB or neural contextual bandit for the exploration slots. This makes exploration more targeted — showing listings that are uncertain but plausibly good, rather than random.

**5. Cross-Market Discovery**

If a user is viewing a listing in Barcelona, occasionally surface similar listings in comparable cities (Lisbon, Nice, Athens). This is valuable for users still in the "dreaming" phase. We'd use the user-type/listing-type embeddings for this, since they generalize across markets.

---

## Closing Summary

**Interviewer:** Great. Let's wrap up. Can you summarize the key design decisions?

**Candidate:** Absolutely.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding approach | Modified Skip-Gram on click sessions | Proven at Airbnb scale, captures behavioral similarity |
| Embedding dim | 32 | Sufficient for marketplace similarity; storage-efficient |
| Cold start | Geo-neighbor averaging + content projection MLP | Immediate coverage without sacrificing quality |
| ANN index | HNSW (FAISS), sharded by market | Best recall-latency tradeoff for 7M listings |
| Ranking model | LambdaMART (GBDT) | Interpretable, fast, handles heterogeneous features |
| Ranking objective | Graded relevance (0–4) optimizing NDCG | Weights bookings over clicks appropriately |
| Diversity | MMR with λ=0.7 | Balances relevance and variety in carousel |
| Latency | < 200ms e2e (< 130ms typical) | Caching at 1hr TTL for popular listings |
| Retraining | Embeddings daily, ranker weekly | Captures seasonality and drift |
| Primary metric | Booking conversion rate from carousel | Aligned with business objective |

The design follows a principled progression from simple (content-based features + cosine similarity) to sophisticated (behavioral embeddings + learned ranker + real-time personalization), with clear offline and online metrics at each stage.

---

## Mental Model: When to Apply This Pattern

This "similar items" design is a reusable template for any marketplace or content platform. The core mental model:

```
Content similarity alone → limited (misses "vibe" and user intent)
Behavioral similarity alone → cold start problem, popularity bias
Combine both via embeddings + learned ranker → best of both worlds
```

**The key structural insight is the funnel shape:** Cheap retrieval (ANN on pre-computed embeddings) filters millions to thousands, then an expensive ranker (GBDT/DNN with rich features) orders hundreds. This pattern — retrieval → filtering → ranking → re-ranking — shows up in search, ads, recommendations, and feed ranking across every major tech company. Once you internalize this four-stage funnel, you can adapt it to any "find similar X" problem.
