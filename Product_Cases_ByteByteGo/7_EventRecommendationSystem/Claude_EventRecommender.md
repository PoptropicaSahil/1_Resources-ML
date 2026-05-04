# Design an Event Recommender System

**ML System Design Interview - Senior Data Scientist (L5/L6)**
45–60 min • Full math & intuition • Multi-stage funnel architecture

`I` = Interviewer | `C` = Candidate (10 YoE)

---

## Interview Timeline

| Phase | Topic | Time |
|-------|-------|------|
| 01 | Requirements Clarification | 5–8 min |
| 02 | Metrics Definition | 5 min |
| 03 | High-Level Architecture | 8 min |
| 04 | Feature Engineering | 8–10 min |
| 05 | Model Design & Training | 12–15 min |
| 06 | Serving & Infrastructure | 5–8 min |
| 07 | Monitoring, Iteration & Edge Cases | 5 min |

---

## Phase 01 - Requirements Clarification (5–8 min)

**`I`:** Let's say you're at a company like Eventbrite or Meetup. You need to design an ML system that recommends events to users. Walk me through how you'd approach this.

**`C`:** Before diving in, I'd like to clarify a few things to scope the problem properly.

- **Product surface:** Are we recommending events on a homepage feed, in email digests, or as search results? I'll assume a **homepage personalized feed** - the highest-traffic surface.
- **Event types:** Are these in-person events (concerts, meetups, conferences), virtual, or both? I'll assume **primarily in-person with some virtual**.
- **Scale:** What order of magnitude are we talking?

**`I`:** Let's say 50M monthly active users, 500K active events at any time, concentrated in 200 metro areas globally. Primarily a homepage feed.

**`C`:** Great. Let me also clarify a few domain-specific constraints that make events fundamentally different from, say, movie or product recommendations:

**1. Temporal perishability:** Events expire. A concert on March 5th is useless on March 6th. Our system must handle inventory that's constantly churning - roughly 10-20K new events/day, 10-20K expiring.

**2. Geographic constraint:** Unlike Netflix or Spotify, physical proximity is a hard filter. A user in San Francisco won't attend a meetup in Tokyo (usually).

**3. Severe cold-start:** Most events are new and have zero interaction history. Unlike products that live for months, the average event might only exist for 2-4 weeks before occurring. We can't rely heavily on collaborative filtering alone.

**4. Capacity constraints:** Events have limited seats. Once sold out, recommending them is harmful.

**5. Implicit vs explicit signals:** Users rarely "rate" events. Our primary signals are clicks, RSVPs, ticket purchases, and attendance.

**`I`:** Those are exactly the right constraints to surface. What assumptions will you make?

**`C`:** **Assumptions:**

- User is logged in (we have user history)
- We have location data (GPS or user-set city)
- Events have metadata: title, description, category, venue, time, price, organizer
- Latency budget: <200ms p99 for the full recommendation pipeline
- We need to generate ~50 recommendations per page load
- We serve ~1000 requests/sec at peak

> 🧠 **Mental Model - Events vs. Static Items**
>
> The key distinction for event recommendation vs. general RecSys is the **item lifecycle**. In product recommendations, item embeddings can be precomputed and cached for weeks. In event recommendation, the item corpus is a *sliding window* - you're constantly indexing new items and expiring old ones. This fundamentally affects your ANN index refresh strategy and cold-start approach.

---

## Phase 02 - Metrics Definition (5 min)

**`I`:** How would you measure success for this system?

**`C`:** I'd structure metrics at three levels: business, online, and offline.

**Business Metrics** - what the CEO cares about:

- *Ticket revenue / RSVP volume* per user per month
- *Event discovery rate* - % of events getting ≥N RSVPs from recommendations (supply-side health)
- *User retention* - DAU/MAU ratio

**Online Metrics** - what we A/B test on:

- *Click-through rate (CTR)* on recommended events
- *RSVP rate* - stronger signal than clicks
- *Conversion rate* - click → RSVP → actual attendance
- *Recommendation diversity* - entropy across event categories in top-10

**Offline Metrics** - for model iteration:

| Metric | Formula / Intuition | Used For |
|--------|---------------------|----------|
| Recall@K | Of all relevant events, what fraction did we retrieve in top K? | Retrieval stage |
| NDCG@K | Are the most relevant events ranked highest? Penalizes relevant items ranked low. | Ranking stage |
| MAP@K | Average precision across all users - rewards correct ordering | End-to-end |
| AUC-ROC | Probability that a positive (RSVP'd) event is scored higher than a negative | Pointwise ranker |
| Log Loss | −[y·log(p) + (1−y)·log(1−p)] - calibration of probability estimates | CTR prediction model |

**`C`:** One critical nuance: **CTR alone is a trap** for events. A clickbait-y event title gets clicks but not RSVPs. So I'd use a **composite objective**:

```
score(user, event) = w₁·P(click) + w₂·P(RSVP|click) + w₃·P(attend|RSVP)
```

Where w₁ < w₂ < w₃ to weight deeper funnel actions more heavily. Concretely, in practice I might use w₁=0.1, w₂=0.3, w₃=0.6. We'd tune these weights via online A/B tests watching the business metrics.

> 🧠 **Mental Model - Metric Hierarchy**
>
> Think of metrics as a pyramid: *offline metrics* give fast iteration signal (minutes), *online metrics* validate via A/B tests (days), and *business metrics* confirm long-term impact (weeks). If your offline metrics improve but online metrics don't, your offline evaluation setup is broken (e.g., data leakage, selection bias). If online metrics improve but business metrics don't, your proxy objective is misaligned with true user value.

**`I`:** How do you handle the position bias problem in offline evaluation?

**`C`:** Great callout. Position bias is the fact that users click items at position 1 more than position 5 regardless of relevance. For offline evaluation, I'd use **Inverse Propensity Scoring (IPS)**:

```
IPS-weighted reward = Σᵢ (rᵢ / P(examine position i))
where P(examine position i) is estimated from position-click curves in logs
```

We can estimate examination probabilities by running a randomization experiment - swap items at positions i and j and measure click differences. The ratio gives you the position bias curve.

---

## Phase 03 - High-Level Architecture (8 min)

**`I`:** Walk me through the system architecture end to end.

**`C`:** I'll use the standard **multi-stage funnel** pattern - the same approach Instagram Explore, YouTube, and Pinterest use. The key insight is that each stage trades off between recall and precision, with computational cost increasing as the funnel narrows.

### Multi-Stage Recommendation Funnel

```
┌─────────────────────────────────────────────────────────┐
│                    EVENT CORPUS                          │
│                   500K active events                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              STAGE 0: GEO + BUSINESS FILTER             │
│  Hard filters: location radius, sold-out, date range    │
│  500K → ~20K candidates         Latency: <5ms           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         STAGE 1: CANDIDATE RETRIEVAL (Recall)           │
│  Two-Tower model + ANN search (HNSW/ScaNN)              │
│  Multiple retrieval channels merged                     │
│  20K → ~500 candidates          Latency: <20ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│        STAGE 2: PRE-RANKING (Light Scoring)             │
│  Lightweight model (distilled from ranker)              │
│  500 → ~100 candidates          Latency: <15ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 3: RANKING (Precision)                   │
│  Deep model: DCN-v2 / MTML with cross-features         │
│  Multi-task: P(click), P(RSVP), P(attend)               │
│  100 → ~50 scored items         Latency: <50ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 4: RE-RANKING (Business Logic)           │
│  Diversity injection, freshness boost, fairness         │
│  Organizer exposure guarantees, dedup                   │
│  50 → 50 re-ordered             Latency: <10ms          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                   [ Final 50 events ]
```

**`C`:** **Why this architecture?**

Scoring all 500K events with the heavy ranking model would take ~500K × 0.5ms = 250 seconds per request. That's obviously impossible with a 200ms budget. The funnel lets us spend our compute budget wisely: cheap models see many items, expensive models see few.

**Latency budget decomposition:**

| Stage | Latency | Items | Compute/Item |
|-------|---------|-------|--------------|
| Geo + Business Filter | ~5ms | 500K → 20K | O(1) lookups |
| Retrieval (Two-Tower + ANN) | ~20ms | 20K → 500 | ANN query |
| Pre-Ranking | ~15ms | 500 → 100 | ~0.03ms/item |
| Ranking (Deep Model) | ~50ms | 100 → 50 | ~0.5ms/item |
| Re-Ranking | ~10ms | 50 → 50 | Rule-based |
| Network/Orchestration | ~30ms | - | - |
| **TOTAL** | **~130ms p50** | - | **< 200ms p99** |

> ⚖️ **Tradeoff - Pre-Ranking: Is It Worth It?**
>
> Pre-ranking adds latency and complexity. You can skip it if your retrieval is precise enough (say, returns ~100 candidates). But at our scale (500 candidates from retrieval), scoring 500 items with a heavy DCN model at 0.5ms/item = 250ms - that blows our budget. The pre-ranker acts as *knowledge distillation at serving time*: it's a lightweight student model trained to approximate the full ranker's output, cutting 5x candidates at 10x speed.

### Offline/Online System Split

```
┌───── OFFLINE (Batch/Streaming) ─────┐    ┌───── ONLINE (Real-time) ──────────┐
│                                      │    │                                   │
│  ┌──────────────┐  ┌──────────────┐  │    │  Request: (user_id, location,     │
│  │ Training Data │  │ Feature      │  │    │           timestamp)              │
│  │ Generation    │  │ Engineering  │  │    │         │                         │
│  │ (click/RSVP   │  │ Pipeline     │  │    │         ▼                         │
│  │  logs)        │  │              │  │    │  ┌─────────────┐                  │
│  └──────┬───────┘  └──────┬───────┘  │    │  │ Feature     │ ← Feature Store  │
│         │                 │          │    │  │ Fetching    │   (Redis/Feast)   │
│         ▼                 ▼          │    │  └──────┬──────┘                   │
│  ┌──────────────┐  ┌──────────────┐  │    │         │                         │
│  │ Model        │  │ Feature      │  │    │         ▼                         │
│  │ Training     │  │ Store        │──│────│  ┌─────────────┐                  │
│  │ (GPU cluster)│  │ (Offline)    │  │    │  │ Multi-Stage │                  │
│  └──────┬───────┘  └──────────────┘  │    │  │ Funnel      │                  │
│         │                            │    │  └──────┬──────┘                   │
│         ▼                            │    │         │                         │
│  ┌──────────────┐  ┌──────────────┐  │    │         ▼                         │
│  │ Event Index  │  │ Model        │  │    │  Response: ranked event list      │
│  │ (ANN rebuild │  │ Registry     │──│────│                                   │
│  │  every ~1hr) │  │              │  │    │                                   │
│  └──────────────┘  └──────────────┘  │    │                                   │
└──────────────────────────────────────┘    └───────────────────────────────────┘
```

**`C`:** One important detail for events specifically: the ANN index needs to be **refreshed frequently** - at least hourly - because events are constantly being created and expiring. Compare this to a product catalog where you might rebuild the index daily. I'd use an **incremental ANN index** (like Milvus or Vespa) that supports real-time inserts and deletes, rather than a batch-rebuilt FAISS index.

---

## Phase 04 - Feature Engineering (8–10 min)

**`I`:** What features would you use, and how do you handle the cold-start problem?

**`C`:** I'll organize features by entity and signal freshness. This matters because different features have different update cadences and serving costs.

| Category | Features | Update Freq | Storage |
|----------|----------|-------------|---------|
| User - Static | age_bucket, gender, city, account_age, preferred_categories (from profile) | Daily | Feature Store |
| User - Behavioral | past_RSVPs_by_category (sparse vector), avg_event_price, avg_distance_traveled, time_of_day_preference, recency_weighted_interaction_embedding | Hourly | Feature Store |
| User - Real-time | current_location, current_session_clicks, time_since_last_visit | Per-request | Computed online |
| Event - Content | title_embedding (BERT), category, sub_category, price_bucket, is_free, is_virtual, duration_hours, venue_embedding | At creation | Feature Store |
| Event - Popularity | total_RSVPs, RSVP_velocity (RSVPs/hour in last 24h), page_views, organizer_avg_rating, seats_remaining_pct | Every 15min | Feature Store |
| Context - Temporal | day_of_week, hour_of_day, is_weekend, days_until_event, is_holiday_week | Per-request | Computed online |
| Cross - User×Event | user_organizer_affinity, user_category_affinity, user_venue_distance_km, user_price_preference_match, social_signal (friends attending) | Per-request (ranking only) | Computed online |

**`I`:** How do you generate the text embeddings for event titles and descriptions?

**`C`:** For the **retrieval stage**, I'd use a distilled sentence transformer (e.g., all-MiniLM-L6-v2 or E5-small) to generate 384-dimensional embeddings. These are precomputed offline when an event is created.

For the **ranking stage**, I don't pass raw BERT embeddings. Instead, I'd extract a few semantic features: the top-3 predicted categories from a text classifier, sentiment score, and keyword features. The ranking model learns its own feature interactions - giving it raw 384-dim embeddings would be wasteful.

Now, for **cold-start** - this is the core challenge for events:

### Cold-Start Strategy by User/Event Matrix

```
                        Known Event          New Event (Cold)
                    ┌──────────────────┬──────────────────────┐
                    │                  │                      │
   Known User       │  Collaborative   │  Content-based       │
                    │  filtering +     │  (text embeddings +  │
                    │  behavioral      │  category match +    │
                    │  features        │  organizer affinity) │
                    │                  │                      │
                    ├──────────────────┼──────────────────────┤
                    │                  │                      │
   New User (Cold)  │  Popularity +    │  Global popularity + │
                    │  geo-contextual  │  trending events +   │
                    │  (trending in    │  onboarding quiz     │
                    │   your city)     │  preferences         │
                    │                  │                      │
                    └──────────────────┴──────────────────────┘
```

**`C`:** The two-tower model is particularly good for cold-start because the event tower can produce an embedding from **content features alone** - title, category, price, venue location - without needing any interaction history. The moment an event is created, it gets an embedding and enters the ANN index. This is a major advantage over pure collaborative filtering approaches like matrix factorization, where a new item with zero interactions has no representation.

For **new users**, I'd use a two-phase approach: (1) show popular/trending events in their geo for the first session, (2) after they interact with 3-5 events, switch to the personalized model. We can also use an **onboarding preference quiz** (like Spotify Wrapped categories) to bootstrap the user embedding.

> 🧠 **Mental Model - Feature Serving Tiers**
>
> Think of features in three serving tiers by latency cost:
>
> **Tier 1 (precomputed, <1ms):** Looked up from feature store by key. User and event static features.
>
> **Tier 2 (near-real-time, <5ms):** Aggregated from streaming pipeline (Kafka → Flink). Event popularity, RSVP velocity.
>
> **Tier 3 (computed at request time, <10ms):** Cross-features like distance, social overlap. These can only exist in the ranking stage where you have both user and event context.
>
> The two-tower retrieval model can *only* use Tier 1 features (because user and event towers must be independent for caching). The ranking model uses all three tiers - that's why it's more powerful.

---

## Phase 05 - Model Design & Training (12–15 min)

**`I`:** Let's go deep on the model architecture. Walk me through retrieval and ranking mathematically.

### Stage 1: Two-Tower Retrieval Model

**`C`:** The core idea: learn separate embedding functions for users and events such that their dot product approximates relevance.

```
  User Features                              Event Features
  ┌─────────────┐                            ┌──────────────┐
  │ user_id     │                            │ event_id     │
  │ city        │                            │ category     │
  │ past_RSVPs  │                            │ title_emb    │
  │ age_bucket  │                            │ price_bucket │
  │ pref_cats   │                            │ venue_geo    │
  └──────┬──────┘                            └──────┬───────┘
         │                                          │
         ▼                                          ▼
  ┌──────────────┐                           ┌──────────────┐
  │  Embedding   │                           │  Embedding   │
  │  Layers      │                           │  Layers      │
  └──────┬───────┘                           └──────┬───────┘
         │                                          │
         ▼                                          ▼
  ┌──────────────┐                           ┌──────────────┐
  │  MLP Layers  │                           │  MLP Layers  │
  │  512→256→128 │                           │  512→256→128 │
  │  + BatchNorm │                           │  + BatchNorm │
  │  + ReLU      │                           │  + ReLU      │
  └──────┬───────┘                           └──────┬───────┘
         │                                          │
         ▼                                          ▼
     uₑ ∈ ℝ¹²⁸                               eₑ ∈ ℝ¹²⁸
     (user emb)                              (event emb)
         │                                          │
         └──────────────┬───────────────────────────┘
                        │
                        ▼
               sim(u, e) = uᵀe / τ
               (cosine similarity / temperature)
```

**Training objective:** We use **in-batch sampled softmax** with temperature scaling. Given a batch of N (user, event) positive pairs, for each user uᵢ, the loss is:

```
L(uᵢ) = −log( exp(uᵢᵀeᵢ / τ) / Σⱼ₌₁ᴺ exp(uᵢᵀeⱼ / τ) )
```

Where τ (temperature) is typically 0.05–0.1. The denominator sums over all events in the batch - the other N−1 events serve as **implicit negatives**.

**Key issue: popularity bias correction.** In-batch negatives are sampled proportional to their frequency in training data, so popular events appear disproportionately as negatives. This causes the model to under-recommend popular items (the logQ correction from the YouTube paper):

```
corrected_logit(uᵢ, eⱼ) = uᵢᵀeⱼ / τ − log(pⱼ)
where pⱼ = frequency of event j in training data / total events
```

Without this correction, the model learns to penalize popular events because they appear as negatives too often. The YouTube DNN paper showed this can drop Recall@K by 5-10%.

### Stage 3: Ranking Model - Deep Cross Network v2 (DCN-v2) with Multi-Task

**`C`:** The ranking model is fundamentally different from retrieval: it sees user-event *pairs* and can compute cross-features. I'd use a DCN-v2 architecture with multi-task heads.

```
    ┌─────────────────────────────────────────────────┐
    │           Input Feature Layer                    │
    │  [user_features ⊕ event_features ⊕ cross_feats] │
    │  concat → x₀ ∈ ℝᵈ                               │
    └─────────────────────┬───────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │   Cross Network  │    │   Deep Network   │
    │                  │    │                  │
    │  xₗ₊₁ = x₀ ⊙   │    │  MLP: 1024→512   │
    │   (Wₗ·xₗ + bₗ)  │    │  →256→128        │
    │   + xₗ           │    │  ReLU + Dropout  │
    │                  │    │  (0.1)           │
    │  (3 cross layers)│    │                  │
    └────────┬─────────┘    └────────┬─────────┘
             │                       │
             └───────────┬───────────┘
                         │ concat
                         ▼
              ┌──────────────────────┐
              │    Shared Layer      │
              │    256 → 128         │
              └──────────┬───────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
    ┌────────────┐ ┌──────────┐ ┌───────────┐
    │ P(click)   │ │ P(RSVP)  │ │ P(attend) │
    │ σ(wc·h+bc) │ │ σ(wr·h+br│ │ σ(wa·h+ba)│
    │ (sigmoid)  │ │ (sigmoid)│ │ (sigmoid) │
    └────────────┘ └──────────┘ └───────────┘
```

**Why DCN-v2?**

The cross network explicitly models feature interactions like "user_preferred_price × event_price × day_of_week" without manual feature crossing. Each cross layer computes:

```
xₗ₊₁ = x₀ ⊙ (Wₗ · xₗ + bₗ) + xₗ
```

Where ⊙ is element-wise multiplication. This is bounded-degree polynomial feature interaction - layer l captures up to (l+1)-order interactions. With 3 cross layers, we get up to 4th-order interactions at linear cost. Compare this to a pure DNN which approximates these interactions less efficiently.

**Multi-Task Training Loss:**

```
L = λ₁·BCE(ŷ_click, y_click) + λ₂·BCE(ŷ_rsvp, y_rsvp) + λ₃·BCE(ŷ_attend, y_attend)

where BCE(ŷ, y) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]

Typical weights: λ₁ = 0.2, λ₂ = 0.5, λ₃ = 0.3
```

The multi-task setup has two big advantages:

**1. Shared representation:** The click task has abundant data (millions/day) and helps learn good lower-layer features that transfer to the RSVP task (sparser, thousands/day) and attend task (sparsest, hundreds/day). This is essentially a form of **auxiliary task regularization**.

**2. Calibrated multi-objective scoring:** At serving time, the final score combines all three predictions:

```
final_score = w₁·P(click) + w₂·P(RSVP|click)·P(click) + w₃·P(attend|RSVP)·P(RSVP|click)·P(click)

≈ w₁·P(click) + w₂·P(RSVP) + w₃·P(attend)
```

**`I`:** Good. What about the training data? How do you construct positive and negative labels?

**`C`:**

**For retrieval (two-tower):**

- *Positives:* (user, event) pairs where user RSVP'd or purchased a ticket
- *Negatives:* In-batch negatives (all other events in the batch) + hard negatives sampled from events the user saw but didn't click

**For ranking (DCN-v2):**

- Training data comes from *logged impressions* - events that were actually shown to users
- Label is click=1/0, RSVP=1/0, attend=1/0
- Critical: only train on events the user actually *saw*, not all events. This avoids **selection bias**.

**Hard negative mining** is crucial for retrieval quality. I'd use a mix of:

1. In-batch negatives (easy, free)
2. Events retrieved by the model but not clicked (semi-hard)
3. Events in same category/geo but not interacted with (hard)

> ⚖️ **Tradeoff - DCN-v2 vs. DeepFM vs. Transformer**
>
> **DCN-v2:** Explicit bounded-degree feature crosses. Efficient, interpretable crosses. Best when feature interactions are important but you want controlled complexity. This is my pick.
>
> **DeepFM:** Factorization machine + DNN. Good for sparse features. Slightly weaker on high-order interactions.
>
> **Transformer-based (BST, DIN):** Great for modeling user behavior sequences ("attended jazz → clicked blues → ?"). Higher latency (~2-5x DCN). Worth it if sequential behavior is a dominant signal. I'd consider adding a DIN-style attention layer on top of DCN-v2 for the user's recent interaction sequence.

**`I`:** How do you handle the class imbalance? RSVP rate might be ~2% and attend rate ~0.5%.

**`C`:** Three complementary approaches:

**1. Negative downsampling:** Randomly sample negatives at rate α (say 0.1), then correct the prediction at serving time:

```
P_corrected = P_model / (P_model + (1 - P_model) / α)
```

This reduces training data size by ~10x without losing signal, which is what Google's Wide & Deep paper recommends.

**2. Focal loss** for the attend task (very sparse):

```
FL(pₜ) = −αₜ (1 − pₜ)ᵞ · log(pₜ)    where γ = 2
```

The (1−pₜ)ᵞ term down-weights easy negatives that the model is already confident about, focusing learning on the hard cases.

**3. Task weighting** via uncertainty-based multi-task loss (Kendall et al.):

```
L = Σₜ (1/2σₜ²) · Lₜ + log(σₜ)
where σₜ is a learned per-task uncertainty parameter
```

This automatically balances the loss across tasks - the attend task (high uncertainty) gets a smaller effective weight initially, preventing it from destabilizing training.

---

## Phase 06 - Serving & Infrastructure (5–8 min)

**`I`:** How do you serve this in production at 1000 QPS?

### Online Serving Architecture

```
User Request (user_id, lat/lng, timestamp)
         │
         ▼
┌──────────────────────────────────────┐
│         API Gateway / LB             │
│         (rate limiting, auth)        │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      Recommendation Orchestrator     │
│  (async parallel execution)          │
│                                      │
│  ┌─────────┐ ┌──────────┐ ┌───────┐ │
│  │ Feature  │ │ Geo      │ │ User  │ │
│  │ Store    │ │ Filter   │ │ Emb.  │ │
│  │ Lookup   │ │ Service  │ │ Compute│ │
│  │ (Redis)  │ │ (PostGIS)│ │       │ │
│  └────┬─────┘ └────┬─────┘ └───┬───┘ │
│       └─────────────┼───────────┘     │
│                     ▼                 │
│  ┌─────────────────────────────────┐  │
│  │   ANN Index (ScaNN / Milvus)   │  │
│  │   Returns top-500 event IDs    │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Pre-Ranker (TF Serving lite) │  │
│  │   Scores 500 → keeps 100       │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Ranker (TF Serving / GPU)    │  │
│  │   Scores 100 items, 3 heads    │  │
│  └──────────────┬──────────────────┘  │
│                 ▼                     │
│  ┌─────────────────────────────────┐  │
│  │   Re-Ranker (rule engine)      │  │
│  │   Diversity, dedup, freshness  │  │
│  └──────────────┬──────────────────┘  │
│                 │                     │
└─────────────────┼─────────────────────┘
                  ▼
         JSON: [event_1, event_2, ...]
```

**`C`:** Key infrastructure decisions:

**Feature Store (Feast + Redis):** User features and event features are precomputed offline and stored in Redis for <1ms lookups. Cross-features (like user-event distance) are computed online during the ranking stage.

**ANN Index:** I'd use **ScaNN** (Google) or **HNSW** (via Milvus/Vespa). For 20K geo-filtered events with 128-dim embeddings, query time is ~2ms. The index is rebuilt incrementally every 30-60 minutes as new events are created.

**Model Serving:** Ranking model on TensorFlow Serving with batching enabled (batch size 32, max wait 5ms). This amortizes GPU compute. Pre-ranker on CPU (it's lightweight enough).

**Caching:** Two-layer cache:

- L1: User-level result cache (TTL=5min) - if the same user refreshes within 5 min, serve cached results
- L2: User embedding cache (TTL=1hr) - avoid recomputing the user tower for every request

**`I`:** What about the ANN index - how do you handle the fact that events expire?

**`C`:** This is the trickiest infra challenge specific to events. Two approaches:

**Option A: Filtered ANN search.** Store all events in the index with metadata (expiry_time, remaining_seats, geo_hash). At query time, apply a pre-filter on metadata before ANN search. Milvus and Vespa support this natively. Downside: filtering reduces effective index size and can hurt recall.

**Option B: Time-partitioned indexes.** Maintain separate ANN indexes per time window (events this week, next week, next month). Query all relevant indexes in parallel and merge results. This avoids filtering overhead but adds operational complexity.

I'd go with **Option A** for simplicity. With 20K geo-filtered candidates (after the geo pre-filter step), even with metadata filtering, the ANN search completes in <5ms.

> 🧠 **Mental Model - ANN Index Refresh Strategy**
>
> Think of ANN indexes on a spectrum:
>
> **Batch (FAISS flat rebuild):** Rebuild entirely from scratch periodically. Simple but stale. OK for product catalogs.
>
> **Incremental (HNSW, Milvus):** Insert/delete individual vectors. Near-real-time freshness. More complex to maintain but essential for fast-changing inventories like events.
>
> **Streaming (Vespa, custom):** Events indexed within seconds of creation. Maximum freshness at maximum complexity.
>
> For events, you need at least "Incremental." The refresh cadence should match your business SLA - if an organizer expects their event to appear in recommendations within 1 hour, your pipeline must index it within 1 hour.

---

## Phase 07 - Monitoring, Iteration & Edge Cases (5 min)

**`I`:** How do you monitor this system and iterate on it?

**`C`:**

**Online Monitoring:**

- *Real-time dashboards:* CTR, RSVP rate, p50/p99 latency per stage, ANN index freshness lag
- *Alerting on:* Latency spikes (>300ms p99), CTR drops >10% from baseline, model serving errors, feature store stale data
- *Data quality monitors:* Feature distribution drift (KL-divergence between training and serving distributions), null rate monitoring

**Model Monitoring:**

- *Prediction calibration:* Is P(RSVP)=0.05 actually resulting in 5% RSVP rate? Plot calibration curves daily.
- *Feature importance drift:* If the model suddenly relies heavily on a single feature, investigate data issues.
- *A/B testing framework:* I'd use a holdout-based system where 5% of traffic sees the existing model and 5% sees the new model, testing on RSVP rate as the primary metric with 95% confidence.

**Retraining Cadence:**

- Retrieval model: retrain weekly (embeddings are relatively stable)
- Ranking model: retrain daily with the last 30 days of data (it needs fresh behavioral signals)
- Use **warm-starting**: initialize from last checkpoint and fine-tune on new data, rather than training from scratch

**`I`:** What are some failure modes or edge cases?

**`C`:**

**1. Filter bubble:** User only attends tech meetups → system only recommends tech meetups → user never discovers cooking classes they'd love. Solution: inject an **exploration component** in re-ranking - 10-15% of slots filled with "contextually adjacent" categories (e.g., if tech → design meetups, entrepreneurship events).

**2. Popularity bias amplification:** Popular events get more clicks → more training signal → ranked higher → more clicks. Solution: add a **popularity penalty** in re-ranking: `score_final = score_model × (1 / log(1 + total_RSVPs)^β)`, where β controls the penalty strength.

**3. Temporal exploitation:** Events happening tomorrow get urgency clicks, not genuine interest. Solution: include "days_until_event" as a feature and be careful not to overweight imminent events.

**4. Organizer fairness:** New organizers with no history get buried. Solution: organizer-level exposure guarantees - ensure every organizer gets a minimum number of impressions proportional to their event count (a form of multi-sided fairness).

**5. Social signal leakage:** "3 friends attending" is a powerful feature but creates a rich-get-richer dynamic for socially connected events. We should A/B test the marginal value of the social signal vs. its concentration effects.

> 🧠 **Mental Model - The Exploration-Exploitation Dial**
>
> Every recommender system sits on a spectrum between *exploitation* (showing what you know the user likes) and *exploration* (showing novel items to learn more). For events, exploration is *more important* than for products because: (a) events are ephemeral - if you don't explore now, the event expires, and (b) user preferences for events are less stable than for products (someone might want jazz one week and hiking the next). A good rule of thumb: 15-20% exploration for events vs. 5-10% for products.

---

## Summary & Key Takeaways

### Complete System - One-Slide Summary

```
┌─────────────────── OFFLINE ──────────────────┐
│                                               │
│  Click/RSVP/Attend Logs                       │
│       │                                       │
│       ▼                                       │
│  Feature Engineering ──→ Feature Store (Feast) │
│       │                                       │
│       ▼                                       │
│  Two-Tower Training ──→ Event ANN Index        │
│  (weekly, sampled softmax + logQ correction)  │
│       │                                       │
│       ▼                                       │
│  DCN-v2 Multi-Task Training (daily)           │
│  (P(click), P(RSVP), P(attend))               │
│  focal loss + uncertainty-weighted MTL         │
│       │                                       │
│       ▼                                       │
│  Model Registry → Canary Deploy → Full Rollout│
│                                               │
└───────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────── ONLINE ───────────────────┐
│                                               │
│  User Request → Geo Filter (500K→20K)         │
│       │                                       │
│       ▼                                       │
│  Two-Tower Retrieval + ANN (20K→500) [<20ms]  │
│       │                                       │
│       ▼                                       │
│  Pre-Ranker: distilled model (500→100) [<15ms]│
│       │                                       │
│       ▼                                       │
│  DCN-v2 Ranker: 3-head scoring (100→50) [<50ms│
│       │                                       │
│       ▼                                       │
│  Re-Rank: diversity, freshness, fairness      │
│       │                                       │
│       ▼                                       │
│  Response: Top 50 events [< 200ms p99 total]  │
│                                               │
└───────────────────────────────────────────────┘
```

### Key Design Decisions

Two-tower for retrieval (cold-start friendly) → DCN-v2 for ranking (explicit feature crosses) → Multi-task heads (P(click), P(RSVP), P(attend)) → Re-ranking for business constraints

### What Makes Events Unique

- Temporal perishability requires incremental ANN indexes
- Geo constraints enable a cheap pre-filter stage
- Severe cold-start demands content-based embeddings over collaborative signals
- High exploration rate (15-20%) because preferences are less stable

### Interviewer Scoring Rubric

- ✓ Clarified domain-specific constraints (not generic RecSys)
- ✓ Multi-level metrics with composite objective
- ✓ Multi-stage funnel with latency budget
- ✓ Mathematical depth (loss functions, corrections)
- ✓ Cold-start strategy leveraging two-tower architecture
- ✓ Production considerations (ANN refresh, caching, monitoring)
- ✓ Edge cases and fairness awareness
