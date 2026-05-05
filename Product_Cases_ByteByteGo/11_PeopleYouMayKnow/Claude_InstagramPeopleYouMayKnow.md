# ML System Design Interview: "People You May Know" (PYMK)

**Role:** Senior Data Scientist, FAANG  
**Duration:** 45–60 minutes  
**Format:** Interviewer ↔ Candidate Q&A

---

## Phase 1 - Problem Clarification & Scoping (5–7 min)

**Interviewer:** Design the "People You May Know" feature for a social media platform like Instagram. Walk me through how you'd approach this.

**Candidate:** Before jumping in, I'd like to clarify a few things to scope the problem properly.

First - **what is the platform's social graph model?** Instagram uses a directed graph (follow model), unlike Facebook's undirected graph (friend model). This fundamentally changes how we define "connection strength" - on Instagram, A following B doesn't mean B follows A. So we're recommending *accounts to follow*, not mutual friendships.

**Interviewer:** Good catch. Let's say directed graph - Instagram-like. Follow model.

**Candidate:** Great. A few more questions:

1. **Scale:** What's the user base? I'll assume ~2B monthly active users, ~500M daily active users.
2. **Where does this surface?** Explore tab, home feed suggestions, profile page "Suggested for You"? Each surface has different latency budgets and intent signals.
3. **What's the primary business objective?** I'll assume it's **increasing meaningful engagement** - measured through follow-through rate and subsequent interactions (likes, comments, DMs) - not just raw follow counts, which can lead to follow/unfollow churn.
4. **Are we designing from scratch or improving an existing system?** I'll assume greenfield but with existing infrastructure (feature stores, embedding services, etc.).

**Interviewer:** Those are solid assumptions. Let's go with all of that. Surface it in the "Suggested for You" card on the home feed and the Explore tab.

**Candidate:** Perfect. Let me also state a key **constraint assumption**: latency budget is **~200ms p99** for the home feed surface (it's in the critical path), and we can afford **~500ms** for the Explore tab since users expect some loading there.

---

## Phase 2 - Metrics Definition (5–7 min)

**Interviewer:** How would you measure success for this system?

**Candidate:** I'd think about metrics at three levels: **offline**, **online**, and **business/guardrail**.

### Offline Metrics (Model Quality)

These are what we optimize during model development:

`why is auc roc here?`

| Metric | Why |
|---|---|
| **AUC-ROC** | Measures ranking quality - can the model separate "will follow" from "won't follow"? Standard for binary classification with heavy class imbalance. |
| **AUC-PR (Precision-Recall)** | More informative than ROC when positive rate is very low (~0.1–1% follow rate). Tells us how well we find actual follows among our recommendations. |
| **NDCG@K** (K=10, 25) | Since we're showing a ranked list, we care about *where* the good recommendations appear. NDCG penalizes relevant items appearing lower in the list. |
| **Recall@K** | Of all the people a user *would* follow, what fraction appears in our top-K? Important for coverage. |

Mathematically, NDCG@K is:

```
NDCG@K = DCG@K / IDCG@K

where DCG@K = Σ(i=1 to K) [rel_i / log₂(i + 1)]
```

Here `rel_i` is the relevance of the item at position `i` (binary: 1 if followed, 0 if not), and IDCG is the ideal DCG (perfect ranking).

### Online Metrics (A/B Test)

| Metric | Definition |
|---|---|
| **Follow-through rate (FTR)** | `# follows from PYMK / # impressions` - primary metric |
| **Engagement rate post-follow** | Within 7/14/28 days after following, does the user like/comment/DM/share content from the recommended account? This is the **north star** because it measures *meaningful* connections, not vanity follows. |
| **Impression-to-profile-visit rate** | Even if they don't follow, did they click through? Signals curiosity/relevance. |

### Guardrail Metrics (Don't Regress These)

| Metric | Purpose |
|---|---|
| **Unfollow rate within 7 days** | High unfollow = bad recommendations, hollow engagement |
| **Report/block rate from PYMK** | Safety signal - are we recommending harassers, spam, inappropriate accounts? |
| **Diversity of recommendations** | Avoid filter bubbles. Measure entropy across categories (interests, demographics, geographies) |
| **Latency p50/p95/p99** | Must stay within budget |

**Interviewer:** Good. Why do you prefer engagement-after-follow over raw follow-through rate as the north star?

**Candidate:** Because **Goodhart's Law** - once you optimize for follows, the model learns to recommend accounts that are easy to follow (celebrities, viral accounts) but don't create meaningful engagement. You get inflated follow counts but hollow connections. The 7-day engagement rate is a **delayed reward** signal that captures actual relationship quality. The tradeoff is feedback delay - we can't train on this in real-time. So in practice we use FTR as the **proxy metric** for online optimization and engagement-after-follow for **periodic model evaluation and A/B test decisions**.

---

## Phase 3 - System Architecture Overview (8–10 min)

**Interviewer:** Walk me through the high-level architecture.

**Candidate:** The classic pattern for recommendation systems at scale is the **multi-stage funnel**: Candidate Generation → Ranking → Re-ranking / Business Logic. This is essentially the architecture described in YouTube's deep recommendation system paper `and what ByteByteGo outlines for recommendation systems at scale.`

```
┌─────────────────────────────────────────────────────────────────┐
│                     PYMK SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

  User Request
       │
       ▼
┌──────────────┐
│  API Gateway  │  (latency budget starts: 200ms)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│              CANDIDATE GENERATION  (~50ms budget)             │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Graph    │  │  Embedding   │  │  Heuristic / Contact   │ │
│  │  Traversal│  │  ANN Search  │  │  Upload Matching       │ │
│  │  (FoF)   │  │  (Two-Tower) │  │                        │ │
│  └────┬─────┘  └──────┬───────┘  └───────────┬────────────┘ │
│       │               │                      │               │
│       └───────────┬───┘──────────────────────┘               │
│                   ▼                                          │
│          Merge & Deduplicate                                 │
│          ~10,000 → ~1,000 candidates                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              RANKING MODEL  (~100ms budget)                   │
│                                                              │
│         Deep Neural Network (Multi-task Learning)            │
│         Input: <user, candidate, context> features           │
│         Output: P(follow), P(engage), P(hide)                │
│                                                              │
│         Score = weighted combination of task heads            │
│         ~1,000 → Top 50                                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              RE-RANKING / BUSINESS LOGIC  (~30ms)             │
│                                                              │
│  • Diversity injection (MMR)                                 │
│  • Freshness boost for new users                             │
│  • Safety filtering (blocked users, policy violations)       │
│  • Position bias correction                                  │
│  • Top 50 → Final 10-25 shown                               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                    Response to Client
```

Let me walk through each stage.

---

## Phase 4 - Candidate Generation (10–12 min)

**Interviewer:** Let's go deep on candidate generation. What sources would you use?

**Candidate:** Candidate generation needs to be **high recall, reasonable precision** - we cast a wide net cheaply. I'd use three parallel sources:

### Source 1: Graph-Based (Friends-of-Friends / FoF)

This is the **strongest signal** for PYMK. The core idea: if user A follows user B, and user B follows user C, then C is a candidate for A.

But raw FoF gives too many candidates. We need to **score** them. The classic approach is **common neighbors**:

```
Score(A, C) = |N(A) ∩ N(C)| / f(|N(A)|, |N(C)|)
```

Where `N(x)` = set of accounts x follows. The denominator normalizes for popularity. Common choices:

| Method | Formula | Property |
|---|---|---|
| **Jaccard** | \|N(A) ∩ N(C)\| / \|N(A) ∪ N(C)\| | Penalizes high-degree nodes equally |
| **Adamic-Adar** | Σ_{z ∈ N(A)∩N(C)} 1/log(\|N(z)\|) | Weights *rare* mutual connections higher - if the mutual friend follows very few people, that's a stronger signal |
| **Resource Allocation** | Σ_{z ∈ N(A)∩N(C)} 1/\|N(z)\| | Similar but sharper penalty on popular nodes |

I'd use **Adamic-Adar** as the primary graph score because it captures the intuition that a mutual connection through a niche account (your college friend who follows 200 people) is far more meaningful than a mutual through a celebrity (who has 50M followers).

**Implementation detail:** At Instagram scale, you can't do full graph traversal in real-time. So we **pre-compute FoF candidates** in a batch pipeline (Spark/MapReduce on the social graph), store the top-500 per user in a key-value store (like RocksDB or Redis), and refresh every few hours.

**Interviewer:** What about the cold-start problem with graph traversal?

**Candidate:** Great point. New users with few or no follows have sparse or empty FoF sets. That's where Source 2 and 3 come in.

### Source 2: Embedding-Based (Two-Tower Model + ANN)

This is the **workhorse for scaling and cold-start**. The idea comes from YouTube's deep recommendation paper and the DSSM (Deep Structured Semantic Models) family.

```
┌─────────────────┐         ┌──────────────────┐
│   USER TOWER     │         │  CANDIDATE TOWER  │
│                 │         │                  │
│  User features: │         │  Candidate feats: │
│  • demographics │         │  • profile info   │
│  • follow history│        │  • content stats  │
│  • interaction  │         │  • follower growth│
│    history      │         │  • content embeds │
│  • device/locale│         │  • account age    │
│                 │         │                  │
│  ┌───────────┐  │         │  ┌───────────┐   │
│  │ MLP layers│  │         │  │ MLP layers│   │
│  └─────┬─────┘  │         │  └─────┬─────┘   │
│        ▼        │         │        ▼         │
│   u ∈ ℝ^128    │         │   v ∈ ℝ^128     │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         └──────────┬────────────────┘
                    ▼
            score = u · v  (dot product)
            or cosine similarity
```

**Training objective:** Given a user `u` and candidate `v`, predict the probability of follow:

```
P(follow | u, v) = σ(u^T · v)
```

We train with **in-batch negatives** + **hard negatives** (candidates the user saw but didn't follow) using a **sampled softmax** loss:

```
L = -log[ exp(u^T · v⁺) / (exp(u^T · v⁺) + Σ_j exp(u^T · v⁻_j)) ]
```

Where `v⁺` is the positive (followed account) and `v⁻_j` are negatives.

**Key design decisions:**

1. **Why dot product over cosine similarity?** Dot product allows the model to learn that some users are generally "harder to please" (shorter embeddings) and some candidates are generally "more followable" (longer embeddings). Cosine kills this information by normalizing. However, for ANN retrieval we need normalized vectors. So we **train with dot product but normalize at serving time** - the ranking model downstream corrects for this.

2. **Embedding dimension = 128.** Standard tradeoff: higher dimensions capture more nuance but increase storage (2B users × 128 × 4 bytes = ~1TB) and ANN search latency. 128 is the industry sweet spot.

3. **ANN Index:** At serving time, we compute the user embedding online, then do **approximate nearest neighbor** search against a pre-built index of all candidate embeddings. I'd use **HNSW** (Hierarchical Navigable Small World) via FAISS or ScaNN. HNSW gives us ~95%+ recall@100 with sub-10ms latency.

**Interviewer:** How do you handle the cold-start for a brand new user who has no interaction history?

**Candidate:** Several strategies layered together:

1. **Contact graph matching** (Source 3) - if the user uploads contacts, we match phone numbers/emails to existing accounts. This is the **highest-precision cold-start signal** and why apps aggressively ask for contact permissions.

2. **Registration signals** in the user tower: device language, country from IP, sign-up source (came from a link shared by user X → recommend X's network), connected Facebook/Google account.

3. **Popularity-based fallback**: For truly zero-signal users, recommend globally popular accounts in their locale, seeded by category diversity (music, sports, fashion, news).

4. **Explore-exploit**: For new users, we should be more **exploratory** - show diverse candidates and use the early follow/skip signals to rapidly update their embedding. This is essentially an **epsilon-greedy** or **Thompson sampling** approach applied to the slate.

### Source 3: Heuristic / Contact Upload

As mentioned - phone contacts, email contacts, Facebook friends (if linked). These have very high precision but limited coverage. Implementation is a simple lookup table.

### Merging Candidates

We take union of all three sources, deduplicate, and pass ~1,000 candidates to the ranker. Each candidate carries a **source tag** (graph, embedding, contact) which becomes a feature for the ranking model.

---

## Phase 5 - Ranking Model (10–12 min)

**Interviewer:** Now let's talk about the ranking model. What architecture would you use?

**Candidate:** The ranker is where we invest the most model complexity because it only scores ~1,000 candidates per request - we can afford richer features and a bigger model here.

### Architecture: Multi-Task Deep Neural Network

I'd use a **shared-bottom multi-task architecture** (or MMoE - Multi-gate Mixture of Experts if we want more sophistication):

```
┌────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                       │
│                                                        │
│  USER           CANDIDATE         CROSS / CONTEXT      │
│  ─────          ─────────         ───────────────      │
│  • user embed   • cand embed      • # mutual follows   │
│  • follow count • follower count  • Adamic-Adar score  │
│  • avg daily    • engagement rate • same city/country?  │
│    session time • content type    • candidate source    │
│  • account age    distribution    • time of day         │
│  • interest     • growth velocity • device type         │
│    categories   • verification    • request surface     │
│                   status          • # times shown before│
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Feature Crossing  │  (DCN-v2 or DeepFM style)
          │   & Embedding Layer │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │   Shared MLP Layers │  (3-4 layers, 512→256→128)
          │   (ReLU + BatchNorm │
          │    + Dropout 0.1)   │
          └──────────┬──────────┘
                     │
            ┌────────┼────────┐
            ▼        ▼        ▼
        ┌───────┐┌───────┐┌───────┐
        │P(follow)│P(engage)│P(hide)│  ← Task-specific heads
        │ head  ││ head  ││ head  │     (each: 64→1, sigmoid)
        └───┬───┘└───┬───┘└───┬───┘
            │        │        │
            ▼        ▼        ▼
         p_follow  p_engage  p_hide
```

### Why Multi-Task?

Single-task (predict follow only) suffers from **label sparsity** - follow events are rare (~0.5-1% of impressions). By jointly training on:

- **P(follow)** - primary task
- **P(engage | follow)** - will they interact after following? Uses delayed labels.
- **P(hide/not_interested)** - negative signal, explicit user feedback

...the shared layers learn richer representations. The engage task acts as a **regularizer** that steers the model toward quality connections, not just easy follows.

### Loss Function

Multi-task weighted binary cross-entropy:

```
L = w₁ · BCE(p_follow, y_follow) + w₂ · BCE(p_engage, y_engage) + w₃ · BCE(p_hide, y_hide)
```

Where `w₁ = 1.0`, `w₂ = 0.3`, `w₃ = 0.5`. The weights are tuned via hyperparameter search. We weight `p_hide` relatively high because **avoiding bad recommendations is as important as surfacing good ones**.

### Final Score

At inference, the final ranking score is a **weighted combination**:

```
score = α · p_follow + β · p_engage - γ · p_hide
```

Where α, β, γ are tuned per surface. For home feed, we weight `p_engage` higher (we want quality). For Explore tab, we might weight `p_follow` higher (users are in discovery mode).

### Feature Engineering Deep-Dive

Let me highlight the most important features and the **intuition** behind them:

**Graph features (highest signal):**

| Feature | Intuition |
|---|---|
| `mutual_follow_count` | Raw count of shared connections |
| `adamic_adar_score` | Quality-weighted mutual connections (rare mutuals > common mutuals) |
| `jaccard_similarity` | Normalized overlap - controls for users who follow thousands of accounts |
| `shortest_path_length` | 2 = FoF, 3 = FoFoF. 2 is much stronger. |
| `bidirectional_mutual_ratio` | Of the mutual connections, how many are *mutual follows* (A↔B, not just A→B)? Higher ratio = tighter social circle |

**Interaction features (behavioral):**

| Feature | Intuition |
|---|---|
| `liked_candidate_content_count` | User already engages with this person's content (via Explore, hashtags) but hasn't followed - strong signal |
| `profile_visits_to_candidate` | Visited profile but didn't follow - could mean interest or deliberate non-follow (feature + the count both matter) |
| `shared_hashtag_interaction_overlap` | Cosine similarity of hashtag engagement vectors |
| `time_since_last_interaction` | Recency matters - interacted yesterday vs. 6 months ago |

**Candidate quality features:**

| Feature | Intuition |
|---|---|
| `follower_to_following_ratio` | High ratio = influential account. But extreme values (1M:10) might mean celebrity, which has different follow dynamics |
| `avg_engagement_rate` | Likes+comments / followers. Healthy accounts have 1-5%. <0.1% might be a dead/bot account |
| `content_posting_frequency` | Active accounts are better recommendations - no one wants to follow someone who hasn't posted in months |
| `account_age_days` | Combined with follower count, detects organic vs. suspicious growth |

**Cross features (these are crucial):**

| Feature | Intuition |
|---|---|
| `candidate_source` | One-hot: {graph, embedding, contact, hybrid}. The ranker learns that contact-sourced candidates have 3-5× higher follow-through |
| `num_times_previously_shown` | Fatigue signal. If we showed this suggestion 5 times and user ignored it, stop showing it |
| `same_city`, `same_country` | Locality matters enormously for social connections |
| `interest_overlap_score` | Cosine similarity of user and candidate interest embeddings (derived from content they engage with) |

**Interviewer:** Why DCN-v2 (Deep & Cross Network) for feature crossing instead of just letting the MLP learn interactions?

**Candidate:** Great question. MLPs *theoretically* can learn any feature interaction, but they do so **inefficiently** for multiplicative interactions. Consider the feature cross `mutual_follow_count × same_city` - this is a powerful signal (many mutual follows AND same city = very likely to know each other). An MLP needs many layers and neurons to approximate this multiplicative relationship via compositions of ReLU activations.

DCN-v2 explicitly models bounded-degree feature crosses:

```
x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
```

This `x_0 ⊙ (...)` term creates **explicit multiplicative interactions** between the input `x_0` and transformed intermediate layers. It's more parameter-efficient for learning these crosses, and the MLP layers on top then learn residual non-linear patterns. In practice, DCN-v2 gives us ~0.5-1% AUC improvement over pure MLP, which at our scale translates to millions of additional meaningful connections.

---

## Phase 6 - Re-Ranking & Business Logic (5 min)

**Interviewer:** What happens after the ranking model scores candidates?

**Candidate:** The raw ranked list needs several post-processing steps:

### 1. Diversity Injection via MMR (Maximal Marginal Relevance)

Without diversity controls, the model might recommend 10 people from the same friend group. We use MMR:

```
MMR(c) = λ · score(c) - (1-λ) · max_{c' ∈ S} sim(c, c')
```

Where `S` is the already-selected set, `sim` is embedding cosine similarity, and `λ ∈ [0.6, 0.8]` trades off relevance vs. diversity. We iteratively select candidates using this criterion.

**Why this matters:** Diverse recommendations give us more **information gain** per impression. If we show 10 candidates from 10 different contexts (work, school, hobby, neighborhood), we learn much more about the user's preferences than 10 candidates from one cluster.

### 2. Position Bias Correction

Users are more likely to follow candidates shown in position 1 vs. position 5, regardless of relevance. This creates a **feedback loop**: items shown higher get more clicks → more positive labels → ranked higher → shown higher...

We correct this by training a **position bias model** separately and dividing out the position effect at serving time. Specifically, during training we include `position` as a feature with a **shallow tower** (separate from the main model), and at inference we set position = 0 (or a default value) so the main model's score isn't contaminated.

### 3. Safety & Policy Filters

- Remove blocked/muted users
- Remove accounts flagged by Trust & Safety
- For minors: only recommend accounts of similar age range, no adult content creators
- Respect "don't suggest my account" privacy settings

### 4. Freshness Controls

- Boost new candidates the user hasn't seen before
- Demote candidates shown >3 times without interaction (fatigue cap)
- For new users, increase exploration (higher temperature in score → softmax sampling)

---

## Phase 7 - Training Pipeline & Infrastructure (5 min)

**Interviewer:** How would you set up the training pipeline?

**Candidate:**

### Training Data

**Positive labels:** User followed the recommended candidate within the session or within 24 hours of impression.

**Negative labels:** Two types:
1. **Impression negatives** - shown but not followed. These are "hard negatives" and very valuable.
2. **Random negatives** - randomly sampled accounts. Needed because impression negatives are biased by the previous model's distribution.

**Ratio:** ~1:10 positive:negative for the follow task. We use **negative downsampling** and correct with a **log-odds correction** at inference:

```
p_corrected = p_model / (p_model + (1 - p_model) / sampling_rate)
```

### Training Schedule

```
┌─────────────────────────────────────────────┐
│             TRAINING PIPELINE               │
│                                             │
│  Daily batch:                               │
│  • Full model retrain on 14-day window      │
│  • ~500M–1B training examples               │
│  • Takes ~4-6 hours on GPU cluster          │
│  • Validated against holdout + shadow mode   │
│                                             │
│  Real-time feature updates:                 │
│  • User embeddings updated hourly via        │
│    streaming pipeline (Flink/Spark Streaming)│
│  • Graph features updated every 2-4 hours   │
│  • Candidate embeddings recomputed daily     │
│                                             │
│  ANN index rebuild:                         │
│  • Full index rebuild every 6-12 hours      │
│  • Incremental updates for new accounts     │
└─────────────────────────────────────────────┘
```

### Why not real-time model updates?

For PYMK, the **label delay** (follow happens hours/days after impression) makes real-time model training impractical. Unlike ads CTR where clicks happen in seconds, follow decisions are slow. Daily retraining captures distribution shifts while maintaining stability.

However, we DO update **features** in near-real-time - if a user follows 5 new accounts, their graph features and embedding should update within hours, not days.

---

## Phase 8 - Evaluation & Iteration (5 min)

**Interviewer:** How do you evaluate the model before launching to production?

**Candidate:** A rigorous evaluation framework with multiple gates:

### Gate 1: Offline Evaluation
- Compare AUC-PR, NDCG@25 against the current production model on a held-out test set (last day's data, not seen during training)
- **Threshold:** Must improve AUC-PR by ≥0.1% and not regress NDCG@25

### Gate 2: Shadow Mode / Interleaving
- Deploy the new model alongside production. For each request, generate recommendations from both models. 
- **Interleaved test:** Mix recommendations from both models in a single list, attribute follows to the originating model. This is more sample-efficient than A/B testing for ranking changes.
- Run for 3-5 days.

### Gate 3: A/B Test
- 5% holdout for the new model
- Run for 2-4 weeks (need time for engagement-after-follow metrics to mature)
- Primary decision metric: **7-day engagement rate post-follow**
- Guardrails: unfollow rate, block rate, latency, diversity metrics
- Statistical significance: p < 0.05 with Bonferroni correction for multiple comparisons

### Gate 4: Gradual Rollout
- 5% → 25% → 50% → 100% over 2 weeks
- Monitor guardrails at each stage
- Automatic rollback if latency p99 exceeds budget or block rate increases >10%

---

## Phase 9 - Advanced Topics & Tradeoffs (5–7 min)

**Interviewer:** Let's discuss some tradeoffs and advanced considerations.

**Candidate:** Sure, let me hit a few important ones:

### Tradeoff 1: Graph Neural Networks vs. Handcrafted Graph Features

**GNN approach:** Use GraphSAGE or GAT to learn node embeddings directly from the social graph. The node embedding captures multi-hop neighborhood structure automatically.

```
h_v^(k) = σ(W^(k) · AGGREGATE({h_u^(k-1) : u ∈ N(v)}) + B^(k) · h_v^(k-1))
```

**Pros:** Captures complex graph topology, no manual feature engineering, can incorporate node and edge attributes.

**Cons:** Computationally expensive at inference for 2B nodes, hard to update incrementally (graph changes constantly), GNN embeddings can be less interpretable.

**My recommendation:** Use GNNs to **generate graph embeddings** in a batch pipeline (updated every 12-24 hours), then use these embeddings as *features* in the ranking model alongside handcrafted graph features. This gives us the best of both worlds - GNN captures complex patterns, handcrafted features provide interpretability and are easier to debug.

### Tradeoff 2: Serving Freshness vs. Cost

| Approach | Freshness | Cost | When to Use |
|---|---|---|---|
| Fully pre-computed (batch) | Low (hours old) | Lowest | Stable users with rich history |
| Pre-computed candidates, real-time ranking | Medium | Medium | Default for most users |
| Fully real-time (compute on request) | High | Highest | New users, users who just followed many accounts |

I'd implement a **tiered approach**: most users get pre-computed candidates with real-time ranking. Users whose graph changed significantly (>5 new follows in last hour) get fresh candidate generation triggered asynchronously.

### Tradeoff 3: Privacy-Preserving Recommendations

This is increasingly important. Key considerations:

- **Differential privacy** on graph features: Add calibrated noise to mutual-connection counts so individual connections can't be inferred from recommendations.
- **On-device candidate generation**: For sensitive contexts, run a lightweight candidate generation model on-device using federated learning principles. Only the final candidate IDs are sent to the server for ranking.
- **Transparency**: Allow users to see *why* someone was recommended ("12 mutual connections", "from your contacts") - this is both good UX and a privacy/trust requirement.

### Tradeoff 4: Global vs. Regional Models

User behavior varies by region (follow patterns in Japan differ from Brazil). Options:

1. **Single global model** with region features → simplest, works well with enough data
2. **Regional fine-tuned models** → better for regions with distinct behavior
3. **Multi-lingual embeddings with region-aware attention** → best quality, highest complexity

I'd start with (1) and move to (2) only if offline metrics show >2% improvement for specific regions.

**Interviewer:** This has been a thorough walkthrough. Any final thoughts?

**Candidate:** One thing I want to emphasize: **the most impactful improvements to PYMK usually aren't model architecture changes** - they're:

1. **Better training data** - getting clean follow/skip labels, handling bots and spam in training data
2. **Feature quality** - a single new high-quality feature (like `liked_candidate_content_count`) can outperform an architecture change
3. **Better negative sampling** - moving from random negatives to a mix of random + hard (shown-but-not-followed) negatives typically gives 2-5% AUC lift
4. **Feedback loops** - ensuring the model doesn't just reinforce its own biases. Periodic exploration (showing random candidates to a small % of traffic) provides unbiased data.

The system I've described handles ~2B users, serves recommendations within 200ms, and optimizes for meaningful social connections rather than vanity follows. The multi-stage architecture makes each component independently improvable, and the multi-task ranking model aligns the optimization with the true business objective.

---

## Mental Models to Take Away

Here are the reusable frameworks from this design that apply to many ML system design problems:

**1. The Multi-Stage Funnel Pattern**
Any recommendation problem at scale follows Candidate Generation (cheap, high recall) → Ranking (expensive, high precision) → Re-ranking (business logic). The key insight: each stage has a different **cost-per-item** budget, so you progressively narrow the candidate set while increasing model complexity.

**2. Proxy vs. True Metrics**
The metric you optimize in real-time (follow-through rate) is rarely your actual business metric (engagement quality). Always identify the true north star, understand why you can't optimize it directly (label delay, sparsity), and design your system so the proxy stays aligned with the true metric.

**3. Goodhart's Law as a Design Principle**
"When a measure becomes a target, it ceases to be a good measure." Build multi-task objectives and guardrail metrics specifically to prevent your model from gaming a single metric.

**4. Cold Start → Warm Start Transition**
Every recommendation system needs a graceful degradation path: rich model → simpler model → heuristics → popularity. Design your candidate generation with multiple sources so you're never stuck with zero recommendations.

**5. Feature Importance Hierarchy for Social Recommendations**
Graph features > Behavioral interaction features > Content similarity > Demographics. This ordering holds across most social recommendation problems and should guide where you invest engineering effort first.
