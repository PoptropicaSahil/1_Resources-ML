# ML System Design Interview: Ad Click Prediction on Social Platforms

> **Format:** Interviewer (I) ↔ Candidate (C) | **Level:** Senior Data Scientist (L6+, ~10 YoE) | **Duration:** ~55 min

---

## Phase 1: Problem Scoping & Requirements (8 min)

**I:** Let's say you're building the ad click prediction system for a social media platform — think Facebook or Instagram scale. Walk me through how you'd approach this.

**C:** Before jumping into modeling, I want to clarify the problem scope and business context. Let me ask a few questions.

**First — what's the prediction target?** Are we predicting click-through rate (CTR) specifically, or a broader engagement objective like click + conversion? Because many platforms have moved to multi-objective optimization — predicting P(click), P(conversion), P(install) etc. and combining them.

**I:** Good question. Let's focus on **P(click | user, ad, context)** as the primary objective, but keep in mind we'll eventually combine it with a conversion signal for ranking.

**C:** Got it. Second — **what's the serving context?** Is this for the ad ranking stage specifically, or also the candidate retrieval stage? Because at scale, we typically have a funnel:

```
┌─────────────────────────────────────────────────┐
│           Ad Selection Funnel                    │
│                                                  │
│   Ad Corpus (millions)                           │
│       │                                          │
│       ▼                                          │
│   Candidate Retrieval ← lightweight model/rules  │
│   (~1000 ads)                                    │
│       │                                          │
│       ▼                                          │
│   Ranking (our model) ← heavy model              │
│   (~100 ads)                                     │
│       │                                          │
│       ▼                                          │
│   Auction & Pricing                              │
│   (top k shown)                                  │
│       │                                          │
│       ▼                                          │
│   Ads Shown to User (1-5)                        │
└─────────────────────────────────────────────────┘
```

I'll assume we're designing the **ranking model** — the core CTR predictor that scores ~100–1000 candidates per request.

**I:** Correct. Let's go with that.

**C:** Third — **scale assumptions.** I'll assume:

- **~1 billion DAU** (Facebook-scale)
- **~100B ad impressions/day**, meaning ~1M+ QPS at peak
- **Latency budget:** the ranking model needs to score candidates in **<50ms** (p99) since it's in the critical path
- **Training data:** billions of labeled (impression, click/no-click) pairs per day
- **Label is binary:** click = 1, no-click = 0, with heavy class imbalance (~2-5% CTR overall)

**I:** Those are reasonable. One thing to flag — why does the latency constraint matter so much here?

**C:** Because ad serving is revenue-critical. Every millisecond of added latency in ad ranking translates to lost impressions and revenue. At Facebook's scale, a 1% drop in ad relevance can mean hundreds of millions in annual revenue. So we need a model that's both accurate *and* fast at inference. That tension shapes every architectural decision we'll make.

---

## Phase 2: Metrics (7 min)

**I:** How would you evaluate this system? Walk me through both offline and online metrics.

**C:** I'd split this into three tiers:

### Offline Metrics

**1. Normalized Cross-Entropy (NE)** — This is the *primary* offline metric at most ad systems. It's the ratio of the model's log-loss to the log-loss of a baseline model that simply predicts the background CTR.

$$
NE = \frac{-\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]}{-\left[p \cdot \log(p) + (1-p) \cdot \log(1-p)\right]}
$$

where $p$ is the average background CTR, and $p_i$ is the model's predicted probability.

**Why NE over raw log-loss?** Because background CTR varies by ad slot, platform, and time. NE normalizes this out — an NE of 1.0 means "no better than predicting the average," and lower is better. It's invariant to the base rate, which makes it comparable across segments.

**I:** That's a nuanced point. What's the intuition for why that matters?

**C:** Consider two ad placements — Instagram Stories (CTR ~1%) and Facebook Feed (CTR ~3%). Raw log-loss on the Stories segment will look "better" just because the base rate is lower. NE corrects for this, so you can compare model quality apples-to-apples.

**2. AUC-ROC** — Measures ranking quality. "Can the model distinguish clicks from non-clicks?" But AUC doesn't care about calibration. A model that predicts 0.9 for all clicks and 0.1 for all non-clicks has perfect AUC but terrible calibration.

**3. Calibration** — The ratio of predicted CTR to observed CTR across decile buckets. This matters enormously for ad pricing. In a generalized second-price auction, the bid is:

$$
\text{eCPM} = \text{pCTR} \times \text{bid\_per\_click} \times 1000
$$

If pCTR is systematically 2x too high, advertisers overpay and churn. If it's too low, the platform leaves revenue on the table. So calibration directly impacts revenue and advertiser trust.

### Online Metrics

**1. Revenue per 1000 impressions (RPM)** — The north star business metric.

**2. CTR** — Are users actually clicking more? But this can be gamed (clickbait-y ads), so we track it alongside...

**3. Conversion rate / downstream value** — To ensure click quality.

**4. User engagement metrics** — Session time, hide-ad rate, negative feedback rate. A model that boosts CTR but degrades user experience is a failure.

**I:** How do you connect offline improvements to online wins?

**C:** This is one of the hardest parts of ad ranking. A **0.1% improvement in NE** typically correlates with a meaningful revenue lift, but the mapping isn't linear and depends on the calibration layer. We'd validate via **A/B testing** with a proper experimentation framework — randomizing at the user level, running for 1-2 weeks, and measuring RPM as the primary guardrail.

One important nuance: you need **adequate power** for the test. Ad revenue has high variance, so even billion-user platforms sometimes need 1-2 weeks to detect small effects. We'd use variance reduction techniques like CUPED (Controlled-experiment Using Pre-Experiment Data) to speed this up.

---

## Phase 3: Feature Engineering (10 min)

**I:** Walk me through the features you'd use.

**C:** Feature engineering is where most of the alpha is in CTR prediction. I'll organize by category:

```
┌──────────────────────────────────────────────────────────────┐
│                    Feature Taxonomy                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  User        │  Ad          │  Context     │  Cross         │
│  Features    │  Features    │  Features    │  Features      │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ Demographics │ Ad creative  │ Time of day  │ User×Ad_cat    │
│ Interests    │ Category     │ Device       │ User×Advertiser│
│ Engagement   │ Advertiser   │ Placement    │ Historical CTR │
│  history     │  history     │ Geo-location │  per segment   │
│ Social graph │ Landing page │ Feed position│ User embedding │
│  features    │  quality     │ Session depth│  × Ad embedding│
│ Device/OS    │ Text/image   │ Day of week  │                │
│ Past ad      │  embeddings  │ Network type │                │
│  interactions│ Campaign age │              │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

Let me go deeper on the most impactful ones:

### 3.1 User Historical Features

These are the highest-signal features. Specifically:

- **User's historical CTR on ads** — overall, per ad category, per advertiser. This is the single most predictive feature in most systems.
- **Smoothed CTR with Bayesian priors**: Raw CTR is noisy for low-impression users. We use empirical Bayes:

$$
\text{smoothed\_CTR} = \frac{n \cdot \text{raw\_CTR} + k \cdot \text{global\_CTR}}{n + k}
$$

where $n$ is the user's impression count and $k$ is a smoothing constant (~100–1000). For new users, this defaults to the global CTR (cold start handling).

- **Recency-weighted engagement**: Clicks from 1 hour ago are more predictive than clicks from 30 days ago. We use exponential decay or windowed aggregates (1h, 6h, 1d, 7d, 30d).

### 3.2 Ad Features

- **Creative embeddings**: Pass the ad image through a pre-trained vision model (ResNet/ViT) and ad text through a language model to get dense representations.
- **Campaign-level statistics**: Historical CTR of this campaign, age of campaign (fresh campaigns often have higher CTR due to novelty).
- **Advertiser quality score**: An aggregate of the advertiser's landing page quality, historical ad performance, and trust signals.

### 3.3 Cross Features

This is where the magic happens. Individual features capture marginal effects, but **interactions** capture "this specific user type responds to this specific ad type."

- **User interest × Ad category**: A user interested in fitness clicking on a fitness ad is very different from a baseline.
- **Historical user-advertiser affinity**: Has this user engaged with this advertiser before?

In classical models (logistic regression), you'd manually define these crosses. In deep models, the network learns them — but explicit crosses still help as we'll discuss in the architecture.

### 3.4 Real-time Features vs. Batch Features

```
┌───────────────────────────────────────────────────────┐
│              Feature Freshness Pipeline                 │
│                                                         │
│  Real-time (seconds)          Batch (hours/daily)       │
│  ┌─────────────────┐         ┌─────────────────────┐   │
│  │ Session clicks   │         │ 30-day engagement   │   │
│  │ Current scroll   │         │ User demographics   │   │
│  │  depth           │         │ Interest profiles   │   │
│  │ Time since last  │         │ Advertiser quality  │   │
│  │  ad impression   │         │ Campaign stats      │   │
│  │ Recent queries   │         │ Social graph feats  │   │
│  └────────┬────────┘         └──────────┬──────────┘   │
│           │                              │              │
│           ▼                              ▼              │
│        ┌─────────────────────────────────────┐          │
│        │        Feature Store / Serving       │          │
│        │     (combines both at inference)     │          │
│        └─────────────────────────────────────┘          │
└───────────────────────────────────────────────────────┘
```

Real-time features are crucial. Facebook's research showed that features computed from the **last few minutes** of user behavior significantly outperform static features. This requires a streaming pipeline (Kafka → Flink → feature store).

**I:** How do you handle feature leakage in this context?

**C:** Great question. The main leakage risk is **position bias**. If we train on "ad shown at position 1" and encode position as a feature, the model learns that position 1 → higher CTR, which is confounded — position 1 gets more clicks because it's more visible, not because the ad is better.

Solutions:
1. **Drop position at serving** but include it during training as a bias correction term
2. **Inverse propensity weighting**: weight samples by 1/P(position) to debias
3. **Train with a position tower** that is detached at inference time (as in YouTube's position-bias work)

---

## Phase 4: Model Architecture (12 min)

**I:** Now let's talk architecture. What model would you use?

**C:** Let me walk through the evolution, because it motivates the current state-of-the-art:

### 4.1 Historical Evolution

```
Logistic Regression (2010s)
    │
    ▼
GBDT + LR (Facebook 2014)
    │
    ▼
Wide & Deep (Google 2016)
    │
    ▼
DeepFM / DCN (2017-2018)
    │
    ▼
DLRM (Facebook 2019)
    │
    ▼
Multi-task / Multi-tower models (current)
```

### 4.2 My Recommendation: DLRM-style Architecture

For a FAANG-scale system, I'd propose a **Deep Learning Recommendation Model (DLRM)** variant with multi-task heads. Here's the architecture:

```
                        ┌──────────────┐
                        │   P(click)   │   ← sigmoid output
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │   MLP Head   │   ← 3-4 FC layers
                        │  (top MLP)   │     [512→256→128→1]
                        └──────┬───────┘
                               │
                    ┌──────────┴──────────┐
                    │                      │
              ┌─────┴──────┐        ┌─────┴──────┐
              │  Feature    │        │   Dense     │
              │ Interaction │        │  Features   │
              │   Layer     │        │  (bottom    │
              │ (dot product│        │   MLP)      │
              │  or DCN)    │        │             │
              └─────┬──────┘        └─────┬──────┘
                    │                      │
         ┌──────────┴──────────┐          │
         │                      │          │
    ┌────┴────┐  ┌────┴────┐   │    ┌─────┴─────┐
    │Embed    │  │Embed    │  ...   │ Dense     │
    │Table 1  │  │Table 2  │        │ Features  │
    │(user_id)│  │(ad_id)  │        │ (age,time │
    │         │  │         │        │  CTR,...)  │
    └────┬────┘  └────┬────┘        └─────┬─────┘
         │            │                    │
    ┌────┴────┐  ┌────┴────┐        ┌─────┴─────┐
    │Sparse   │  │Sparse   │        │Continuous │
    │Features │  │Features │        │Features   │
    └─────────┘  └─────────┘        └───────────┘
```

### 4.3 Key Components — Deep Dive

**Embedding Tables** are the memory-intensive part. For user_id alone at Facebook scale, you'd have ~2 billion entries × 64 dimensions = ~500 GB. This is why DLRM is a **memory-bound** model, not compute-bound. The embedding lookups dominate inference cost.

**Feature Interaction Layer** — This is where we model explicit feature crosses. Two main approaches:

**Option A: Dot-product interactions (original DLRM)**

Given $n$ embedding vectors $\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n$, compute all pairwise dot products:

$$
\text{interactions} = \{\mathbf{e}_i^T \mathbf{e}_j \mid i < j\}
$$

This produces $\binom{n}{2}$ interaction terms. It's efficient and captures second-order feature crosses.

**Option B: Deep & Cross Network (DCN-v2)**

Cross layers explicitly learn feature crosses of arbitrary order:

$$
\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l
$$

where $\mathbf{x}_0$ is the input and $\odot$ is element-wise multiplication. Each cross layer adds one order of feature interaction. 2-3 cross layers capture up to 3rd-4th order crosses efficiently.

**I'd go with DCN-v2** because it captures higher-order interactions without the quadratic blowup, and Google's research shows consistent gains over basic dot-product interactions.

**Bottom MLP** processes dense (continuous) features through a small network [256→128→64] to project them into the same space as the embeddings.

**Top MLP** takes the concatenation of interaction outputs and dense feature representations, and produces the final logit through [512→256→128→1].

### 4.4 Multi-Task Learning

In practice, we don't just predict click. We jointly predict multiple objectives:

```
                   Shared Layers
                        │
            ┌───────────┼───────────┐
            │           │           │
        ┌───┴───┐  ┌───┴───┐  ┌───┴───┐
        │ Click │  │Convert│  │ Hide  │
        │ Tower │  │ Tower │  │ Tower │
        │ (MLP) │  │ (MLP) │  │ (MLP) │
        └───┬───┘  └───┬───┘  └───┬───┘
            │          │          │
          P(click)   P(conv)   P(hide)
```

The final ad score combines these:

$$
\text{score} = w_1 \cdot P(\text{click}) \times \text{bid} + w_2 \cdot P(\text{conv}) \times \text{conv\_value} - w_3 \cdot P(\text{hide})
$$

**Why multi-task?** Because conversion events are much sparser than clicks (~10-100x fewer). By sharing bottom layers, the conversion head benefits from the click signal through **transfer learning**. The shared representation captures general user-ad affinity, and each head specializes.

**I:** What loss function would you use?

**C:** Binary cross-entropy (log loss) per task, with task-specific weights:

$$
\mathcal{L} = \sum_{t \in \text{tasks}} \lambda_t \cdot \left[-\frac{1}{N}\sum_{i=1}^{N} y_i^{(t)} \log p_i^{(t)} + (1-y_i^{(t)}) \log(1-p_i^{(t)})\right]
$$

For the click task, given the heavy class imbalance (2-5% positive rate), I'd use **negative downsampling** — keep all positives but sample negatives at rate $q$ (e.g., 0.1). This reduces training data by ~10x without much quality loss, but you need to **recalibrate** afterwards:

$$
p_{\text{calibrated}} = \frac{p_{\text{model}}}{p_{\text{model}} + \frac{1-p_{\text{model}}}{q}}
$$

This corrects for the sampling bias and restores true probability estimates.

**I:** Why not use focal loss for the imbalance?

**C:** Focal loss $\mathcal{L}_{FL} = -\alpha_t(1-p_t)^\gamma \log(p_t)$ down-weights easy negatives by a factor $(1-p_t)^\gamma$. It works well in detection tasks where the imbalance is extreme (1:1000+). For CTR with ~2-5% positive rate, negative downsampling is simpler, more controllable, and in practice gives similar results. But focal loss is a valid alternative — the key insight is the same: reduce the gradient contribution from easy negatives.

---

## Phase 5: Training Pipeline (8 min)

**I:** How would you train this at scale?

**C:** Several key decisions here:

### 5.1 Training Data & Freshness

```
┌─────────────────────────────────────────────────────┐
│              Training Pipeline                       │
│                                                      │
│  Kafka Stream (impressions + clicks)                 │
│       │                                              │
│       ▼                                              │
│  Join & Label (click within 30min window)            │
│       │                                              │
│       ▼                                              │
│  Feature Backfill (attach feature snapshots)         │
│       │                                              │
│       ▼                                              │
│  Training Data Store (hourly partitioned)            │
│       │                                              │
│       ├──► Batch Retraining (daily, full model)      │
│       │                                              │
│       └──► Online/Incremental Update (hourly,        │
│            embedding tables + top MLP only)           │
└─────────────────────────────────────────────────────┘
```

**Model freshness is critical in ads.** User behavior shifts (weekday vs. weekend, trending topics), new ads launch daily, and seasonal patterns are strong. Stale models leave money on the table.

My approach: **daily full retraining** on a sliding window of the last 7-30 days, plus **near-real-time incremental updates** to the embedding tables using the last few hours of data. This is similar to what Facebook described in their 2014 paper and what most FAANG ad systems use today.

### 5.2 Training Infrastructure

At this scale (~100B examples/day), we need **distributed training**:

- **Data parallelism** for the MLP components — replicate the model across GPUs, shard the data
- **Model parallelism** for the embedding tables — they don't fit on a single GPU (~100s of GB). Shard embedding tables across multiple machines.

This is the DLRM-specific challenge: embedding lookups are **all-to-all communication patterns**, which create network bottlenecks. Facebook built custom infrastructure (ZionEX) for this.

### 5.3 Label Definition — Subtle but Important

**What counts as a click?** This needs careful definition:

- **Attribution window:** Click must happen within T minutes of impression (typically 30 min)
- **Deduplication:** Count only distinct clicks (users may double-click)
- **Invalid traffic filtering:** Remove bot clicks, accidental clicks (clicked and bounced in <1 sec), and adversarial clicks (click fraud)

The click-through time matters too — a click after 0.5 seconds might be accidental, while a click after scanning the ad for 3 seconds is intentional. Some systems use **dwell time post-click** as a quality signal.

### 5.4 Avoiding Training-Serving Skew

This is a silent killer. The features used at training time must exactly match what's available at serving time. Common pitfalls:

1. **Using future information**: e.g., aggregating features over a time window that extends past the impression timestamp
2. **Feature computation differences**: training uses batch Spark jobs, serving uses streaming Flink — subtle numerical differences accumulate
3. **Feature staleness**: a feature was fresh during training but 6 hours stale at serving time due to pipeline lag

**Solution:** Log features at serving time (feature logging), and train on the **logged features** rather than recomputing them. This guarantees consistency. The downside is storage cost (~TB/day), but it's worth it.

---

## Phase 6: Serving & Inference (7 min)

**I:** How do you serve this model at 1M+ QPS with <50ms latency?

**C:** This is where system design meets ML. Let me outline the serving architecture:

```
┌──────────────────────────────────────────────────────────┐
│                 Serving Architecture                      │
│                                                          │
│  User Request                                            │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Feature  │───►│   Feature    │───►│   Model       │  │
│  │ Request  │    │   Store      │    │   Server      │  │
│  │ Builder  │    │  (Redis /    │    │  (TorchServe/ │  │
│  │          │    │   RocksDB)   │    │   TF Serving) │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │           │
│       ┌──────────────────────────────────────┘           │
│       ▼                                                  │
│  ┌──────────┐    ┌──────────────┐                       │
│  │  Score   │───►│   Auction    │───► Ad Response       │
│  │  Cache   │    │   Module     │                       │
│  └──────────┘    └──────────────┘                       │
└──────────────────────────────────────────────────────────┘
```

### Key Optimizations

**1. Embedding lookup optimization**
Embedding tables are too large for GPU memory. Options:
- **CPU-based lookups with GPU-based MLP**: embeddings stay in CPU memory or SSDs; only the dense computation runs on GPU
- **Embedding compression**: use hashing tricks, mixed-dimension embeddings (important features get larger dims), or quantization (float32 → int8)

**2. Batched inference**
Score all ~100-1000 candidate ads for a user in a single batched forward pass. The MLP portion parallelizes well on GPU.

**3. Two-phase ranking**
- **Phase 1 (coarse):** lightweight model (logistic regression or small NN) scores all 1000 candidates → top 100
- **Phase 2 (fine):** full DLRM scores 100 candidates → final ranking

This reduces the expensive model's load by 10x.

**4. Score caching**
For ads that are shown repeatedly to many users, cache partial computations. The user-independent ad features and embeddings can be precomputed.

**5. Model distillation**
Train a large "teacher" model offline, then distill it into a smaller "student" model for serving. The student mimics the teacher's soft probability outputs:

$$
\mathcal{L}_{\text{distill}} = \alpha \cdot \text{CE}(y, p_{\text{student}}) + (1-\alpha) \cdot \text{KL}(p_{\text{teacher}} \| p_{\text{student}})
$$

This can cut model size by 3-5x with minimal quality loss.

**I:** What about the cold start problem for new ads?

**C:** New ads have no historical engagement data, so their embeddings are uninformative. Several mitigation strategies:

1. **Content-based features as fallback**: Use the ad's creative (image/text embeddings), category, and advertiser-level statistics. These are available from day zero.
2. **Exploration budget**: Reserve a fraction (~5-10%) of impressions for exploration — show new ads to gather data, even if predicted CTR is uncertain. This can be formalized as a **Thompson Sampling** or **ε-greedy** approach:
   - Sample from the posterior of the CTR: $\hat{p} \sim \text{Beta}(\alpha + \text{clicks}, \beta + \text{no-clicks})$
   - New ads with wide posteriors naturally get explored
3. **Warm-starting embeddings**: Initialize a new ad's embedding as the average embedding of similar ads in the same category/advertiser.

---

## Phase 7: Calibration (5 min)

**I:** You mentioned calibration earlier. How do you ensure it in practice?

**C:** Calibration is the most underappreciated aspect of ad CTR models. Even a well-trained model drifts out of calibration over time.

### Calibration Pipeline

**1. Isotonic Regression (Platt Scaling variant)**
After the base model produces raw scores, fit a monotonic mapping from predicted → actual CTR using held-out data.

**2. Field-aware calibration**
Calibrate separately per segment — by ad format (video, image, carousel), by platform (mobile, desktop), by country. A model can be well-calibrated overall but badly miscalibrated on specific slices.

**3. Continuous monitoring**
Plot expected vs. observed CTR in real-time, bucketed by predicted score decile:

```
Expected vs. Observed CTR (Calibration Plot)

Observed │
   CTR   │              ╱ ideal (y=x)
         │           ╱
  0.05   │        ╱ ●
         │      ╱●
  0.03   │    ●╱
         │  ●╱
  0.01   │●╱
         └──────────────────
         0.01  0.03  0.05
              Predicted CTR
```

Points should lie on the diagonal. Systematic deviations indicate the model is over- or under-predicting in that range.

---

## Phase 8: Monitoring & Failure Modes (5 min)

**I:** What can go wrong in production?

**C:** Lots. Here's what I'd monitor:

### Real-time Alerts
- **Predicted CTR distribution shift**: if the mean predicted CTR suddenly changes by >10%, something is broken (bad feature, stale data, model corruption)
- **Revenue anomalies**: sudden RPM drops, which could indicate a model serving issue
- **Feature coverage**: if a critical feature (e.g., user historical CTR) suddenly has 50% missing values, the model degrades silently

### Known Failure Modes

**1. Feedback loops**: The model predicts high CTR for an ad → it gets shown more → it gets more clicks → reinforcing the prediction, even if the ad isn't actually good. This creates a "rich get richer" dynamic.
*Mitigation:* Randomized exploration + measuring causal effect via holdout experiments.

**2. Data pipeline delays**: If the streaming feature pipeline goes down, the model falls back to stale features. A user who just bought running shoes might still see running shoe ads.
*Mitigation:* Graceful degradation — detect stale features and fall back to batch features with a bias correction.

**3. Adversarial attacks**: Click farms inflate CTR for specific ads.
*Mitigation:* Anomaly detection on click patterns (IP clustering, timing patterns, user behavior fingerprinting).

**4. Concept drift**: User behavior shifts due to external events (holidays, pandemics, elections). Models trained on "normal" data underperform.
*Mitigation:* Short training windows + continuous retraining + monitoring NE over time.

---

## Summary & Mental Model

**I:** Great walkthrough. Can you summarize the key tradeoffs?

**C:** Here's how I think about the overall system:

```
┌────────────────────────────────────────────────────────┐
│            Key Tradeoffs in Ad CTR Systems              │
│                                                         │
│  Model Complexity ◄──────────────► Serving Latency      │
│  (deeper = better AUC)            (deeper = slower)     │
│                                                         │
│  Feature Freshness ◄─────────────► System Complexity    │
│  (real-time = higher CTR)         (streaming infra cost)│
│                                                         │
│  Exploration ◄───────────────────► Exploitation         │
│  (learn about new ads)            (maximize revenue now) │
│                                                         │
│  Calibration ◄───────────────────► Discrimination       │
│  (accurate probabilities)         (ranking quality)      │
│                                                         │
│  Privacy ◄───────────────────────► Personalization      │
│  (user data protection)           (better targeting)     │
│                                                         │
└────────────────────────────────────────────────────────┘
```

The best ad CTR systems aren't the ones with the fanciest models — they're the ones that nail the **feature engineering pipeline**, maintain **tight training-serving consistency**, keep the model **fresh and well-calibrated**, and do all of this within a **strict latency budget**.

> The 80/20 of CTR prediction: **features > model architecture > training tricks > serving optimizations.**
