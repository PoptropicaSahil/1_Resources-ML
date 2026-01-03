# ML System Design Interview: Ad Click Predictor for Meta
## Senior Data Scientist Role

---

## PART 1: CLARIFICATION & REQUIREMENTS GATHERING

### Interviewer
"Great, let's talk about building an ad click prediction system for Meta. Before we dive into architecture, I want to make sure we're aligned on what problem we're solving. Let me start with some clarifying questions."

### Candidate
"Perfect, I appreciate that. Let me share my initial understanding and I'd like your feedback. We're building a system to predict whether a user will click on an ad. This probability would be used to rank ads in real-time for a user. Is that correct?"

### Interviewer
"Yes, exactly. But let me probe a bit deeper. When you say 'real-time', what latency constraints are we working with? And are there other metrics we care about beyond just CTR prediction?"

### Candidate
"Great questions. Let me lay out some assumptions:

**Latency Requirements:**
- Inference latency: 50-100ms total for serving ads to users
- This includes candidate retrieval, ranking, and any post-processing

**Business Metrics & Trade-offs:**
- Primary: Maximize CTR (Click-Through Rate) = Clicks / Impressions
- Secondary: Consider user satisfaction (don't show irrelevant ads)
- Tertiary: Advertiser ROI (help advertisers' campaigns succeed)
- Revenue impact: Even 0.1% improvement in model accuracy = significant revenue gain at Meta's scale

**Data Scale:**
- Daily active users: ~100 million
- Daily ad impressions: ~10 billion (estimated)
- CTR baseline: ~1-2%
- Training data: ~1 billion clicks/day with ~50 billion impressions

**Model Update Frequency:**
- Can we retrain daily or do we need more frequent updates? I'm thinking at least daily, but online learning capabilities would be ideal for fresher models."

### Interviewer
"Excellent context. Those numbers are pretty accurate. Now, let me add some complexity. You mentioned considering user satisfaction. How do we measure that? And what about the cold-start problem when we have new users, new ads, or ads from new advertisers?"

### Candidate
"Those are excellent points. Let me think through this:

**User Satisfaction:**
- Could use implicit signals: view duration, hover time, skip/dismiss rates
- Could combine CTR with conversion rate (CV) to look at downstream metrics
- Need to be careful about feedback loops—predicting clicks but users hate the ads

**Cold-Start Problem:**
- New users: Use content-based features (demographics, device, location)
- New ads: Use campaign features, advertiser historical performance, ad creative features
- New advertisers: Apply transfer learning or use embeddings from similar advertisers
- This is where feature engineering becomes critical

**Feedback Delay:**
- For social platforms, clicks are immediate, but conversions can take days/weeks
- Should we use only clicks as labels or incorporate conversion signals?
- I suspect Meta uses multi-task learning to predict both CTR and CVR

Are these the right directions? Should we focus primarily on CTR or design for multi-task predictions?"

### Interviewer
"Good thinking. For this session, let's focus primarily on CTR as our main objective, but keep the architecture flexible for future multi-task additions. Now, one more critical piece: what are the **training data requirements** you're thinking? How old should training data be? Should it be balanced or real-world distribution?"

### Candidate
"Excellent question on data quality. Here's my thinking:

**Data Freshness:**
- Training data recency matters significantly
- Day-1 data might work for batch models retraining daily
- But for online learning, we'd want data from recent hours
- I'd suggest: retrain daily on last 30 days of data to capture trends while having enough volume

**Class Imbalance:**
- With 1-2% CTR, we have heavily imbalanced data (98-99% negatives)
- Raw imbalanced data could bias model toward predicting "no click"
- Options:
  1. Use downsampling (keep all positives, sample negatives)
  2. Use weighted loss functions (give more weight to positive class)
  3. Use calibration post-training to correct class imbalance effects
- I'd lean toward **weighted loss + calibration** to preserve learned signals

**Train/Val/Test Split:**
- Temporal split is critical (not random split) since ads evolve over time
- Maybe: 30 days train, 7 days validation, 1 day test
- Or: keep validation as holdout from training period to check overfitting

Does this align with Meta's thinking?"

### Interviewer
"Perfect. You're thinking like a production systems engineer. One more crucial thing before we jump to architecture: **what features do you think we'll need**? Give me your first-pass list of feature categories."

### Candidate
"Great, let me bucket features by source:

**User Features:**
- Demographics: age, gender, location, device_type, OS
- Behavioral: past_clicks_history, past_impressions, click_through_rate, favorite_categories
- Interest: inferred interests from past activity, follower count, engagement pattern
- Device: device_type, OS version, screen_size, connection_type
- Temporal: time_of_day, day_of_week, timezone

**Ad Features:**
- Ad identity: ad_id, advertiser_id, campaign_id, ad_type (image/video/carousel)
- Content: category, subcategory, brand, creative_text, image_embeddings
- Creative quality: historically_high_performing_ads, advertiser_quality_score
- Bid info: bid_amount, campaign_budget_remaining
- Historical performance: historical_ctr, historical_cvr, impressions_count

**Context Features:**
- Contextual: feed_position, impression_timestamp, user_session_duration
- Page context: page_type (feed/search/marketplace), competing_ads_quality
- Seasonal/Trend: time_since_ad_launch, trend_signal, viral_score

**Cross Features (interaction):**
- user_id × ad_id_embedding_similarity
- user_interest_category × ad_category (many matching)
- user_past_ctr_in_category × ad_category
- advertiser_id × user_past_engagement_with_advertiser

Questions I have:
1. Should we include embeddings (like ad creative embeddings) or keep it to tabular features initially?
2. How do we handle real-time features vs. batch features?
3. Are there feature latency constraints—some features need to be computed instantly vs. can be pre-computed?"

### Interviewer
"Excellent first-pass. You're on the right track. For now, assume we can mix embeddings and tabular features—modern architectures handle both. Let's dig into one more thing before we design: **how do you think about model serving**? What's the inference pipeline look like?"

### Candidate
"Good setup for the high-level design. Let me sketch this:

**Inference Pipeline:**
```
1. User enters feed
   ↓
2. Fetch candidate ads (billions → thousands of relevant ads via retrieval)
   ↓
3. Retrieve features for user + each candidate ad
   ↓
4. Score each ad with CTR model (latency critical)
   ↓
5. Apply business rules (diversity, budget pacing, policy filters)
   ↓
6. Rank top N ads and display
```

**Latency Breakdown (targeting 100ms total):**
- Candidate generation: 10-20ms
- Feature retrieval: 20-30ms
- Model inference: 20-30ms
- Reranking/filtering: 10-20ms
- Network overhead: 10-20ms

This means:
- We need **pre-computed embeddings** for ads and users (stored in cache/index)
- We can't afford to compute complex features on-the-fly
- Model needs to be lightweight or use batch inference tricks

Should we design for a two-stage ranking system—one lightweight pre-ranker and one heavy ranker? Or single-stage?"

### Interviewer
"Smart thinking on latency. Let's use two-stage ranking—it's what Meta actually does. Pre-ranker filters massively, main ranker does deep predictions. Now I'm satisfied with the requirements. Let's move to architecture."

---

## PART 2: HIGH-LEVEL ARCHITECTURE

### Interviewer
"Alright, before we dive into components, give me the 10,000-foot view. What are the major pieces of your system?"

### Candidate
"Let me draw this out. Here's my high-level architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINES                                 │
├─────────────────────────────────────────────────────────────────┤
│  Raw Logs (Kafka)  →  ETL (Spark/Flink)  →  Feature Store       │
│                                                                   │
│  Batch Features                     Online Features              │
│  (Pre-computed daily)              (Near real-time)              │
│   - Historical stats                - Recent clicks              │
│   - User interests                  - Session context            │
│   - Ad performance                  - Current trends             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE                                    │
├─────────────────────────────────────────────────────────────────┤
│  Training Data Store  →  Data Prep  →  Model Training           │
│                                      (GPU cluster)               │
│                                                                   │
│  Models: Pre-ranker, Main Ranker                                │
│  Training Frequency: Daily batch + hourly online updates        │
│  Model Store: S3/distributed storage                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              SERVING PIPELINE (Online)                            │
├─────────────────────────────────────────────────────────────────┤
│  User Request                                                     │
│      ↓                                                            │
│  Candidate Generation (lightweight retrieval) - 50k-100k ads    │
│      ↓                                                            │
│  Pre-ranker Model (shallow model) - filters to top 1k           │
│      ↓                                                            │
│  Feature Enrichment (fetch from feature store + cache)          │
│      ↓                                                            │
│  Main Ranker Model (deep CTR model) - scores top 1k             │
│      ↓                                                            │
│  Reranking Service (diversity, budget pacing, policies)         │
│      ↓                                                            │
│  Return Top-N Ads                                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              MONITORING & FEEDBACK                                │
├─────────────────────────────────────────────────────────────────┤
│  Track predictions vs. actuals                                   │
│  Detect model drift, data drift                                 │
│  Log interactions as new training signals                       │
│  A/B test new models before full rollout                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components:**
1. **Data Ingestion & Feature Store** - handles both batch and streaming
2. **Model Training Infrastructure** - distributed GPU training
3. **Model Serving** - low-latency serving for inference
4. **Monitoring** - detects drift, evaluates performance
5. **Feedback Loop** - generates training labels

Does this architecture align with what you'd expect?"

### Interviewer
"Very good structure. I like that you separated pre-ranker from main ranker. Now, let's dig deeper into each piece. Let's start with the data pipeline—specifically the **feature store** and how you handle feature freshness."

---

## PART 3: DEEP DIVE - DATA PIPELINE & FEATURE ENGINEERING

### Candidate
"Great. Feature freshness is one of the trickiest parts. Let me break down the feature store architecture:

### Feature Store Design

**Batch Features (precomputed daily):**
```
User Features:
  - user_historical_ctr (float)
  - user_impression_count_30d (int)
  - user_click_count_30d (int)
  - user_avg_engagement_time (float)
  - user_interests_embedding (vector)
  - user_device_type (category)
  - user_location (geography)

Ad Features:
  - ad_historical_ctr (float)
  - ad_advertiser_quality_score (float)
  - ad_category (category)
  - ad_creative_embedding (vector)
  - ad_campaign_budget_spent_pct (float)
```

**Online Features (near real-time, sub-second latency):**
```
Session-level:
  - session_impression_count (int)
  - session_clicks_count (int)
  - session_duration (float)
  
Recent User Behavior (last 24 hours):
  - user_recent_ctr_24h (float)
  - user_recent_interactions (list of ad_ids)
  
Real-time Context:
  - current_timestamp (datetime)
  - is_peak_hour (bool)
  - current_user_online_status (bool)
```

**How do we manage freshness?**

Option 1: **Lazy Computation** (more common in practice)
```
During training:
  1. Read logs from HDFS/S3 (raw events)
  2. For each user-ad pair, compute features on-the-fly
  3. Feature extraction is the bottleneck (can be 50-70% of training time)
  4. This is expensive but ensures consistency between training & serving
```

Option 2: **Pre-computed Features** (what we implement for speed)
```
Batch pipeline (runs daily):
  1. Spark/Flink reads event logs from last 30 days
  2. Aggregates user features: click_count, impression_count, engagement_stats
  3. Aggregates ad features: historical_ctr, quality_score
  4. Stores in Feature Store (columnar format like Parquet)
  5. Also stores in Redis/in-memory cache for serving (1000 most active users)

Online pipeline (runs continuously):
  1. Streaming engine (Kafka → Flink) processes events
  2. Computes recent features: session_context, last_24h_interactions
  3. Writes to Redis (TTL: 24 hours for session features)
  4. Updates embeddings lazily when user/ad state changes significantly
```

**Feature Joining at Serving Time:**
```
When user requests ads:
  1. Get user_id, context from request
  2. Fetch batch features for user from Feature Store → Redis cache (cached)
  3. For each candidate ad (1000 ads):
       a. Fetch batch features for ad from Feature Store → Redis
       b. Fetch session features from Redis
       c. Compute cross-features if needed (category match, etc.)
  4. Concatenate into feature vector
  5. Feed to model
  
Latency: batch features ~1-5ms (cached), online features ~2-5ms
Total feature joining: ~10-15ms per batch of 1000 ads
```

**Handling Feature Inconsistency (Training vs. Serving):**

This is critical. In production, features can drift:
```
Problem: Feature computed differently in training vs. serving
Example: user_historical_ctr
  - During training: computed from full 30-day history available at training time
  - During serving: computed from history UP TO the inference moment
  - After 1 day, the value changes slightly

Solution: Nearline Feature Attribution (like LinkedIn does)
  1. During logging, capture the features used in the serving request
  2. Store these exact feature values with the label (click/no-click)
  3. Use these saved features for training
  4. Guarantees feature consistency
  
Implementation:
  - Serving system logs: [user_id, ad_id, features_used, timestamp]
  - Logging system: [user_id, ad_id, features_used, label] (label comes later)
  - Training reads pre-joined data with consistent features
```

**Feature Dimensionality:**

Meta's production CTR models have:
- ~10^12 dimensions in sparse feature space
- ~10^3 dimensions after embedding layer

How?
```
Categorical features with high cardinality:
  - user_id: 100M unique users
  - ad_id: 1B+ unique ads
  - query/text features: billions of unique tokens
  
One-hot encoding multiplier:
  100M × 1B = 10^17 possible user-ad pairs (sparse)
  
Approximate dimensions: 10^12 is reasonable for production scale
```"

### Interviewer
"Excellent detail on feature stores. This shows you understand the engineering. One question: **how do you handle the feature extraction bottleneck** you mentioned? 50-70% of training time is a lot."

### Candidate
"Great catch. This is exactly where systems like **FeatureBox** (Baidu's solution) come in. Let me explain the problem and solution:

**The Problem:**
```
Traditional Training Pipeline:
  Raw Logs (50 TB)
  ↓
  Feature Extraction (MapReduce, ~18 hours, generates 200 TB intermediate I/O)
  ↓
  Feature Store (15 TB materialized)
  ↓
  Model Training (GPU, ~4 hours)
  
  Total: 22 hours just for feature extraction!
```

**Why is extraction slow?**
- Joins across multiple data sources (user logs, ad logs, context)
- String processing, embeddings lookups
- Writing intermediate data to HDFS repeatedly
- I/O bandwidth becomes bottleneck

**FeatureBox Solution: Pipeline Feature Extraction with Training**
```
GPU Pipeline:
  Raw Logs → 
  Read Views → 
  Clean Views (null handling) → 
  Join Views → 
  Extract Features → 
  Merge Features → 
  (IMMEDIATELY FEED TO TRAINING without materializing)
  Train Model
  
  Result: 5-10X speedup (18 hours → 3.5 hours for 1B instances)
  Saves: 50-100 TB intermediate I/O
  
Key insight: Features computed and consumed in GPU memory, 
             not written back to distributed file system
```

**How to implement:**
```
Architecture:
  Heterogeneous Operator Scheduling
  - Computation-intensive ops → GPU kernels
  - Memory-intensive ops (joins, lookups) → CPU
  - Network I/O intensive → CPU (read from HDFS)
  
  Layer-wise DAG scheduling:
  1. Create dependency graph of feature extraction operations
  2. Topologically sort: layer 1 → layer 2 → ... → layer N
  3. In each layer, operations have no dependencies (can run parallel)
  4. Sync at layer boundaries
  5. Use meta-kernels to avoid CUDA launch overhead

  GPU Memory Management:
  - Pre-allocate memory pool for dynamic allocations
  - Use block-level parallel prefix sum for thread allocation
  - Reset pool between layers (constant time)
```

**Implementation Trade-offs:**
- Plus: 5-10X faster feature extraction, massive I/O savings
- Minus: Complex system, requires GPU expertise, engineering effort
- For Meta's scale: Absolutely worth it

For an interview, you could say: 'Ideally, we'd pipeline feature extraction with training to avoid intermediate I/O. In practice, we'd likely use a framework similar to FeatureBox to achieve this.'"

### Interviewer
"Perfect. That shows you know the state-of-the-art optimizations. Now let's move to **model training**—what models would you use, and how would you handle the data scale?"

---

## PART 4: DEEP DIVE - MODEL ARCHITECTURE

### Candidate
"Excellent question. Let me think through this step-by-step.

### Model Selection

**What are the requirements for our CTR model?**
- Handle high-dimensional sparse inputs (~10^12)
- Learn feature interactions (not just linear relationships)
- Be trainable at scale (billions of examples)
- Be servable with low latency (<50ms per 1000 ads)
- Handle both tabular and embedding features
- Be interpretable (at least somewhat, for ads)

### Option 1: Traditional Logistic Regression + Decision Trees

From Facebook's famous paper on CTR prediction:
```
Model: Logistic Regression on carefully engineered features
  + Decision Trees (as additional features)
  
Why it works:
  - Historical information (past CTR, engagement) dominates
  - Simple feature-cross engineering beats complex models
  - Fast inference
  
Performance: Outperforms either model alone by ~3%

Trade-offs:
  - Requires heavy manual feature engineering
  - Limited ability to learn implicit patterns
  - Doesn't scale to modern feature complexity
```

### Option 2: Shallow Models - Factorization Machines

```
Factorization Machines (FM):
  - For each pair of features (i, j), learn latent interaction vector
  - Prediction: w_0 + Σ w_i*x_i + Σ<v_i, v_j>*x_i*x_j
  
Where:
  - w_0: global bias
  - w_i: first-order weights
  - v_i, v_j: latent interaction vectors
  
Pros:
  - Captures pairwise feature interactions explicitly
  - Works with sparse data
  - Relatively fast to train
  
Cons:
  - Only captures pairwise interactions (misses higher-order)
  - Doesn't automatically learn embeddings
```

### Option 3: Deep Learning Models (Modern Approach)

Here's where we get into the sophisticated architectures:

#### **Model A: Wide & Deep Learning (Google, 2016)**

```
        ┌─────────────────────────┐
        │  Final Output (CTR)      │
        └────────────┬─────────────┘
                     │
            ┌────────┴────────┐
            ↓                 ↓
        ┌────────┐      ┌──────────┐
        │  Wide  │      │  Deep    │
        │ Layer  │      │ Network  │
        └────────┘      └──────────┘
            ↑                 ↑
        
        Manual        Embeddings
        Features      + DNN

Architecture Details:
  Wide Part:
    - Linear model on carefully engineered cross-product features
    - Memorizes specific user-item interactions
    - Good for recent patterns
    
  Deep Part:
    - Embedding layer: sparse features → dense vectors
    - Hidden layers: [256, 128, 64] neurons with ReLU
    - Learns general patterns and generalization
    
  Joint Optimization:
    - Loss = weighted sum of wide & deep losses
    - Example: 0.5 * L_wide + 0.5 * L_deep
```

#### **Model B: DeepFM (2017) - Recommended for this problem**

```
        ┌────────────────────────┐
        │  Final Prediction (CTR) │
        └────────────┬────────────┘
                     │
            ┌────────┴───────┐
            ↓                ↓
        ┌────────┐      ┌─────────┐
        │   FM   │      │   DNN   │
        │ Part   │      │ Part    │
        └────────┘      └─────────┘
            ↑                ↑
            └────────┬───────┘
                     │
            ┌────────┴────────┐
            ↓                 ↓
         Embedding Layer   Embedding Layer
           (shared)          (shared)

Advantages over Wide & Deep:
  - FM part explicitly models second-order interactions
  - No need for manual feature crosses
  - Scales better than manual cross-features
  - Both parts learn from same embeddings
  
Mathematical Formulation:
  CTR = σ(y_FM + y_DNN)
  
  y_FM = w_0 + Σ w_i*x_i + Σ<v_i, v_j>*x_i*x_j
  y_DNN = MLP(Embedding(x))

Why DeepFM works well:
  1. FM captures pairwise interactions (low-order)
  2. DNN captures high-order interactions (implicit)
  3. Shared embedding layer prevents over-parameterization
  4. End-to-end differentiable, easy to train
```

#### **Model C: Deep Cross Network (DCN)**

```
        ┌───────────────────────┐
        │  Final Output (CTR)    │
        └────────────┬───────────┘
                     │
            ┌────────┴─────────┐
            ↓                  ↓
        ┌────────┐         ┌────────┐
        │ Cross  │         │  Deep  │
        │Network │         │Network │
        └────────┘         └────────┘
            ↑                  ↑
            └────────┬─────────┘
                     │
            Embedding Layer

Cross Network Design:
  x_(l+1) = x_0 ⊙ (w_l^T x_l + b_l) + x_l
  
  Where:
    - ⊙: element-wise multiplication
    - x_0: input embeddings
    - x_l: output from previous layer
    - w_l, b_l: learnable parameters
    
  Effect: 
    - Each layer computes explicit feature crosses
    - Scales linearly in terms of depth (efficient)
    - Parametric control over interaction complexity

Pros:
  - More expressive than FM for feature crosses
  - Computationally efficient
  - Works well in practice
  
Cons:
  - Still limited to explicit crosses (may miss patterns)
  - Harder to interpret what crosses are being learned
```

#### **Model D: Deep Interest Network (DIN, 2017 from Alibaba)**

```
        ┌───────────────────────┐
        │ Final CTR Prediction   │
        └────────────┬───────────┘
                     │
                ┌────┴─────┐
                ↓          ↓
            ┌────────┐  ┌──────┐
            │  Final │  │Final │
            │Pooling│  │  MLP │
            └────────┘  └──────┘
                ↑
         ┌──────┴────────┐
         ↓               ↓
    [Target Item]   [User Interest Sequence]
         +         with Attention Mechanism
         
    Attention Score = softmax(DNN([user_interest_i, target_ad]))
    User Representation = Σ attention_i * embedding_i

Key Innovation:
  - Captures which historical ads are relevant to current ad
  - Solves: "Why does user suddenly have different interests?"
  - Attention mechanism: soft selection over historical interests

When to use DIN:
  - If you have user sequence data (good for this)
  - When user interests change dynamically within session
  - For platforms with rich interaction history (Meta has this!)

Complexity: More parameters, slightly higher latency
Performance: ~2-3% improvement over DeepFM on sequence-dependent tasks
```

### **My Recommendation for Meta CTR Predictor: DeepFM + Optional DIN**

```
Stage 1: Pre-ranker (lightweight)
  - Logistic Regression with engineered features
  - OR Simple 2-layer DNN
  - Purpose: Filter from 100k candidates to 1000
  - Latency: <5ms
  
Stage 2: Main Ranker
  - DeepFM as baseline
  - Architecture:
    
    Input Features:
      Sparse: [user_id, ad_id, category, ...]
      Dense: [user_ctr, ad_performance, ...]
    
    Embedding Layer:
      user_id → [64]d
      ad_id → [64]d
      category → [16]d
      (total: ~10,000 dimensions of features → 1000-2000 dimensional embeddings)
    
    FM Part:
      Learns pairwise interactions: user_id × ad_category, etc.
      Output: scalar
    
    DNN Part:
      [embedding_concat] → [512] ReLU → Batch Norm → Dropout
                       → [256] ReLU → Batch Norm → Dropout
                       → [128] ReLU → Batch Norm → Dropout
                       → [1] Sigmoid
      Output: probability (0-1)
    
    Combine: CTR = sigmoid(FM_output + DNN_output)
    
    Latency: ~20-30ms per batch of 1000 ads (parallelized)

Stage 2b (Future): Consider adding DIN
  - If A/B testing shows user sequence matters
  - Attention over historical interactions
  - Could improve by 1-2% but adds latency
```

### **Training Specifics**

```
Loss Function:
  Binary Crossentropy (log loss):
    L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    
  Where:
    y = 1 if ad was clicked, 0 otherwise
    ŷ = model prediction (CTR probability)

Optimization:
  Optimizer: Adam or SGD with momentum
  Learning Rate: 0.001-0.01 (adaptive based on dataset)
  Batch Size: 2048-4096 (largest that fits in GPU memory)
  
  Learning rate schedule:
    - Warm-up: 0 → 0.01 over first 5% of training
    - Plateau: 0.01 for 60% of training
    - Decay: exponential decay in final 35%

Regularization:
  - L2 regularization: λ = 0.0001
  - Dropout: p = 0.3 on hidden layers
  - Batch Normalization: reduces internal covariate shift
  - Early stopping: monitor validation loss, stop if no improvement for 3 epochs

Class Imbalance Handling:
  Since CTR is 1-2%, use:
  - Class weights: {0: 1.0, 1: 50.0} (upweight positives)
  OR
  - Downsampling: keep all positives, sample 5-10% of negatives
  - Then multiply loss by 10-20 to correct for downsampling
```

### **Training Infrastructure**

```
Data Volume:
  - Training set: 30 days × 10B impressions = 300B examples
  - But with downsampling (1:10 ratio): 30B examples
  - On disk: ~100-150 TB (TFRecord format)

GPU Training:
  - Use GPU clusters (e.g., NVIDIA A100s)
  - Distributed training: Data parallelism
  - Framework: TensorFlow or PyTorch with Horovod
  
  Example setup:
    - 8 GPU servers (8 GPUs each = 64 GPUs total)
    - Each GPU handles batch of 2048 samples
    - Total batch: 64 × 2048 = 131k samples per step
    
  Training time:
    - ~2-4 hours for 300B downsampled examples (30B)
    - With 1000 steps/hour = 1000 × 3 hours = 3000 steps
    - Each step: 131k samples, so covers 391B samples total (good!)

Checkpointing:
  - Save model every 10% of training (3 checkpoints)
  - Keep last 3 checkpoints in case of issues
  - Version in S3/distributed storage

Retraining Schedule:
  - Daily full retrain: captures new patterns, accounts for ad/user churn
  - Hourly online training (optional): for time-critical patterns
    - Lighter version using recent data only
    - Smaller batch size, fewer epochs
```"

### Interviewer
"Excellent deep-dive on models. Your recommendation of DeepFM is solid. But I want to probe one more thing: **how would you actually serve this model at scale to meet the 50-100ms latency requirement**? You can't possibly run deep inference on 100k candidates."

### Candidate
"Exactly! This is where the two-tower architecture + approximate nearest neighbor search comes in. Let me explain:"

---

## PART 5: MODEL SERVING & INFERENCE OPTIMIZATION

### Candidate
"Great question. The key insight is that we CAN'T score all 100k candidates with the heavy DeepFM model in 100ms. Here's the solution:

### Two-Stage Ranking Pipeline

```
┌────────────────────────────────────────────────────────┐
│  User Request                                           │
│  [user_id, session_context, request_metadata]          │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ↓ (0ms)
┌────────────────────────────────────────────────────────┐
│ STAGE 1: CANDIDATE GENERATION (10-15ms)                │
│  Lightweight Retrieval                                  │
├────────────────────────────────────────────────────────┤
│ Methods:                                                 │
│  1. Rule-based filtering: remove ineligible ads         │
│     - Budget exhausted ads                             │
│     - Advertiser-blocked ads                           │
│     - Category filter (user preferences)               │
│  2. Heuristic scoring:                                 │
│     - Historical CTR of ad (pre-computed)              │
│     - Advertiser quality score                         │
│     - Simple user-ad similarity                        │
│  3. ANN index retrieval:                               │
│     - Compute user embedding (2-3ms)                   │
│     - Query ANN index (Approximate Nearest Neighbor)   │
│     - Get top 1000 ads by embedding similarity         │
│                                                         │
│ Output: ~1000-5000 candidate ads                       │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ↓ (15-20ms)
┌────────────────────────────────────────────────────────┐
│ STAGE 2: RANKING (20-40ms)                             │
│  Deep CTR Model (DeepFM)                               │
├────────────────────────────────────────────────────────┤
│ For each of 1000 candidate ads:                         │
│  1. Fetch user & ad features (batched, cached)         │
│  2. Score with DeepFM model                            │
│  3. Get CTR probability                                │
│                                                         │
│ Optimization:                                           │
│  - Batch inference: score all 1000 at once             │
│  - GPU parallelization: ~50-100 ads/ms throughput      │
│  - Model quantization: use float16 instead of float32  │
│                                                         │
│ Output: Ranked ads with CTR scores                     │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ↓ (40-50ms)
┌────────────────────────────────────────────────────────┐
│ STAGE 3: RERANKING & FILTERING (10-20ms)              │
│  Business Logic & Constraints                          │
├────────────────────────────────────────────────────────┤
│  - Apply diversity constraints (don't show same        │
│    advertiser multiple times)                          │
│  - Budget pacing (ensure advertiser's daily budget     │
│    is spent smoothly, not all at once)                 │
│  - Policy enforcement (filter unsafe ads)              │
│  - Freshness boost (boost new campaigns)               │
│                                                         │
│ Output: Final top-N ads to display                     │
└─────────────────┬──────────────────────────────────────┘
                  │
                  ↓ (50-100ms total)
         Return ads to user
```

### Implementation Details: Two-Tower Embedding Architecture

```
For efficient ANN retrieval, use Two-Tower model:

TRAINING:
  ┌────────────────────┐
  │  Training Data     │
  │  (user, ad, label) │
  └─────────┬──────────┘
            │
       ┌────┴────┐
       ↓         ↓
  ┌────────┐ ┌──────────┐
  │ User   │ │   Ad     │
  │ Tower  │ │  Tower   │
  └───┬────┘ └────┬─────┘
      │           │
      └─────┬─────┘
            │
       [Similarity]
            ↓
        [Loss: Log Loss]
        
Training Objective:
  - Maximize similarity of user & clicked ad embeddings
  - Minimize similarity of user & non-clicked ad embeddings
  
Mathematical Form:
  Loss = -log(sigmoid(u · v_pos)) - log(1 - sigmoid(u · v_neg))
  
  Where:
    u: user embedding
    v_pos: positive (clicked) ad embedding
    v_neg: negative (unclicked) ad embeddings

SERVING:
  Offline (happens once daily/weekly):
    1. For all ads, compute ad embedding using Ad Tower
    2. Store embeddings in ANN index (e.g., ScaNN, Faiss, Milvus)
    3. These embeddings are static (can be cached/indexed efficiently)
    
  Online (for each user request):
    1. Compute user embedding: user_features → User Tower → embedding
       (only need current user's embedding, which is lightweight)
    2. Query ANN index: "Find top-1000 ads similar to this user"
       (Approximate Nearest Neighbor search, <5ms)
    3. Pass these 1000 ads to ranking stage
```

### Architecture Diagrams

**Two-Tower Model Architecture:**

```
USER TOWER:
  
  Raw User Features:
  ├─ demographics (age, gender, location)
  ├─ historical behavior (past clicks, impressions)
  ├─ interests (inferred from history)
  └─ session context (current activity)
  
  ↓ Embedding Layer
  
  ├─ Category embeddings (32-128 dims each)
  └─ Continuous features (batch normalized)
  
  ↓ Concatenate: [e_1, e_2, ..., e_n] → (1000-2000 dims)
  
  ↓ Hidden Layers
  
  Dense [512] ReLU
     ↓
  Dense [256] ReLU
     ↓
  Dense [128] ReLU
     ↓
  Dense [64] ReLU
     ↓
  [User Embedding] (64 dims)

AD TOWER:
  
  Raw Ad Features:
  ├─ ad metadata (id, category, advertiser)
  ├─ creative (text, image embeddings)
  ├─ campaign info (budget, historical perf)
  └─ contextual (recency, trend)
  
  ↓ Embedding Layer
  
  ├─ Categorical embeddings
  └─ Continuous features
  
  ↓ Concatenate
  
  ↓ Hidden Layers
  
  Dense [512] ReLU
     ↓
  Dense [256] ReLU
     ↓
  Dense [128] ReLU
     ↓
  Dense [64] ReLU
     ↓
  [Ad Embedding] (64 dims)

SIMILARITY SCORING:
  
  Score = User_Embedding · Ad_Embedding  (dot product)
  
  Or with normalization:
  Score = cos_sim(User_Embedding, Ad_Embedding)  (cosine similarity)
```

### Optimization Techniques for Inference

```
1. Model Quantization:
   - Use float16 instead of float32: 2X memory savings, ~10% latency gain
   - Or int8 quantization: 4X savings, ~20% latency gain (with tiny accuracy loss)
   
2. Batch Inference:
   - Score 1000 ads in parallel (batch size 1000)
   - GPU parallelization: 100-500 ads/ms throughput
   - Total time: 1000 ads ÷ 100 ads/ms = 10ms
   
3. Model Serving Infrastructure:
   - Use TensorFlow Serving, TorchServe, or Seldon
   - Implement model caching: keep weights in GPU memory
   - Connection pooling: reuse connections to inference service
   
4. Feature Caching:
   - Pre-fetch and cache user features in local memory
   - Use Redis for frequently accessed user features (~1-5ms latency)
   - Cache hit rate: 70-80% for active users
   
5. Approximate Nearest Neighbor:
   - Use HNSW (Hierarchical Navigable Small World) or ScaNN
   - Trade-off: 99% recall @ 50X faster than brute force
   - 100k ads: brute force ~ 50ms, ANN ~ 1-2ms
```

### Handling Model Versioning & A/B Testing

```
Challenge: How do we A/B test new models without serving both?

Solution: Multi-Armed Bandit approach + offline evaluation

Model Registry:
  - Model V1 (current prod): 50% traffic
  - Model V2 (new candidate): 25% traffic
  - Model V3 (another variant): 25% traffic
  
Metrics tracked per model:
  - CTR (clicks / impressions)
  - CVR (conversions / clicks)
  - User satisfaction (engagement metrics)
  - Model inference latency
  
Online evaluation:
  - Run for 1-7 days
  - Statistically test: V2 CTR vs V1 CTR
  - If V2 wins by >0.1%, promote to prod gradually
  - If no improvement or regression, discard V2

Gradual rollout:
  Day 1-2: 5% traffic
  Day 3-4: 20% traffic
  Day 5-7: 50% traffic
  Day 8+: 100% traffic (if metrics hold)
```"

### Interviewer
"Excellent explanation of the serving pipeline. I really like how you mapped out the latency constraints. Now let's talk about **model evaluation**—both offline and online. How do you measure if a new model is actually better?"

---

## PART 6: MODEL EVALUATION & MONITORING

### Candidate
"Perfect. This is often underestimated but equally important to model training. Let me walk through offline and online evaluation:

### Offline Evaluation Metrics

```
Primary Metric: AUC (Area Under ROC Curve)

What it measures:
  - Probability that model ranks a random click higher than random non-click
  - Range: [0.5, 1.0] where 0.5 = random, 1.0 = perfect
  - Robust to class imbalance
  
Mathematical formula:
  ROC Curve: plot(FPR, TPR) for various thresholds
    where FPR = false positive rate
          TPR = true positive rate
  
  AUC = area under this curve
  
Typical baseline:
  - Random model: AUC = 0.50
  - Logistic regression: AUC = 0.65-0.70
  - DeepFM: AUC = 0.78-0.82
  - Meta's production model: likely 0.83-0.85
  
Improvement benchmark:
  - An improvement of 0.001-0.002 in AUC = significant at Meta's scale
  - Translates to millions of dollars in revenue

Why use AUC over accuracy?
  - Accuracy is misleading with imbalanced data
  - Example: if you predict 'no click' always, accuracy = 99%!
  - AUC captures ranking quality, which is what we care about
```

```
Secondary Metrics: Log Loss (Cross Entropy)

Formula:
  LogLoss = -(1/N) * Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
  
  Where:
    y_i = true label (0 or 1)
    ŷ_i = predicted probability
    
Interpretation:
  - Lower is better (0 is perfect)
  - Typical values: 0.40-0.50 for CTR models
  - Penalizes confident wrong predictions heavily
  
Use case:
  - More sensitive to probability calibration than AUC
  - Important for auction systems where exact probabilities matter

Normalized Cross Entropy (NCE):
  - Log Loss relative to baseline model
  - NCE = (LogLoss_model - LogLoss_baseline) / LogLoss_baseline * 100%
  - Easier to compare improvements across time
```

```
Calibration Metrics: O/E Ratio (Observed-to-Expected)

Problem: Model might predict 0.7 CTR, but actual is 0.5
Solution: Calibration - adjust predicted probabilities

Measurement:
  - Bucket predictions into deciles: [0-0.1], [0.1-0.2], ..., [0.9-1.0]
  - For each bucket, compute:
    - Expected CTR = average predicted probability
    - Observed CTR = actual click rate in bucket
    - O/E = Observed / Expected
    
Interpretation:
  - O/E = 1.0: perfectly calibrated
  - O/E < 1.0: model overestimates CTR
  - O/E > 1.0: model underestimates CTR
  
Example:
  Predicted: 0.70, Actual: 0.50 → O/E = 0.50/0.70 = 0.71 (underestimating)
  
LinkedIn's improvement:
  'Nearline Feature Attribution improved O/E from 0.65 to 1.0'
```

### Online Evaluation Metrics

```
User-Facing Metrics:

1. Click-Through Rate (CTR)
   CTR = Clicks / Impressions
   
   Primary metric, business critical
   
   Example:
   - Day 1 (old model): 1.5% CTR (150k clicks, 10M impressions)
   - Day 2 (new model): 1.53% CTR (153k clicks, 10M impressions)
   - Improvement: +0.03 percentage points = +2% relative lift
   
2. Conversion Rate (CVR)
   CVR = Conversions / Clicks
   
   Often more important than CTR for revenue
   - CTR means user interested in ad
   - CVR means user actually bought/completed action
   
3. Revenue Per Impression (RPI)
   RPI = Total Revenue / Total Impressions
   
   Directly tied to business outcome
   
4. User Engagement (secondary guardrail)
   - Avg session duration
   - Return rate (users come back next day)
   - Dismissal rate (users hide/close ads)
   
   Why track: if CTR goes up but dismissal goes up too, 
              users are annoyed by lower-quality ads
```

```
Advertiser-Facing Metrics:

1. ROAS (Return on Ad Spend)
   ROAS = Revenue / Ad Spend
   
   Advertiser perspective: are their ads working?
   
2. Campaign Completion Rate
   - Did advertisers' campaigns finish or end early?
   - Early termination signals dissatisfaction
   
3. Advertiser Retention
   - Do advertisers keep using Meta for advertising?
   - Proxy: percentage of advertisers with ads in the system
```

### Evaluation Workflow

```
OFFLINE EVALUATION (batch):

1. Holdout Test Set
   - Use past 1 day of data
   - Model trained on prior 30 days
   - Temporal split (not random) to avoid leakage
   - Size: ~100M examples for statistical significance
   
2. Compute Metrics
   - AUC on test set
   - Log Loss
   - O/E ratio in deciles
   - Per-segment analysis:
     * By user cohort: new vs. existing users
     * By ad type: image vs. video vs. carousel
     * By time of day: peak vs. off-peak
     * By geography: US vs. international
   
3. Compare to Baseline
   - If AUC improves by >0.001, consider for online test
   - If AUC same or worse, discard
   - Compute statistical significance: need 100M+ examples
     for small differences to be significant
   
4. Segment Analysis
   - Does model help all users equally?
   - Or does it hurt new users while helping established?
   - Example red flag: great overall AUC, but 5% drop for new users
     → Model likely over-optimized for data bias

5. Error Analysis
   - Look at examples where model fails
   - High predicted CTR but didn't click: why?
     * Maybe content is misleading
     * Maybe ad placement is bad
   - Low predicted CTR but did click: why?
     * Maybe model undervalues certain categories
     * Maybe user interest changed suddenly
```

```
ONLINE EVALUATION (A/B test):

Setup:
  - Control group (50%): old model
  - Treatment group (50%): new model
  - Run for 7 days minimum (captures weekly pattern)
  - Need 1M+  impressions per group for significance
  
Hypothesis:
  H0: new model CTR = old model CTR
  H1: new model CTR > old model CTR
  
Statistical Test:
  - Chi-square test for independence
  - Or Bayesian approach: compute posterior probability
    that new model is better
  
Results:
  - Point estimate: (treatment CTR - control CTR) / control CTR
  - Confidence interval: 95% CI around estimate
  - P-value: probability result is by chance
  
Decision:
  - If p-value < 0.05 AND point estimate > +0.1%: promote
  - If p-value > 0.05 OR estimate is negative: discard
  - If estimate is 0.01-0.1%: run longer (potential future improvement)

Risk Mitigation:
  - Monitor key guardrail metrics:
    * User engagement (session duration, return rate)
    * Advertiser satisfaction (campaign completion)
    * Revenue per impression
  - If any guardrail drops >0.5%, stop experiment immediately
  - Gradual rollout instead of binary switch
```

### Monitoring in Production

```
Metrics Dashboard (real-time):

1. Model Predictions
   - Avg predicted CTR (should stay ~constant)
   - Distribution of predictions (check for strange patterns)
   - Calibration: O/E ratio (should stay ~1.0)
   
2. Actual Performance
   - Observed CTR (hourly)
   - Alert if deviates >10% from 7-day average
   
3. Data Drift Detection
   - Feature distributions: compare current vs. baseline
   - Use KL-divergence or Kolmogorov-Smirnov test
   - Alert if feature distribution changes significantly
   
4. Model Drift Detection
   - Prediction distribution over time
   - Check for "model collapse": predictions cluster at 0.5
   - Monitor feature importance: are rankings shifting?

5. Latency Monitoring
   - P50, P95, P99 latency for inference
   - Alert if P99 > 100ms (violates SLA)

Alerting Strategy:
  - Threshold-based:
    * If CTR drops >5%, page on-call engineer
    * If avg prediction probability > 0.7, investigate model
  - Trend-based:
    * If CTR declining for 3 consecutive hours, alert
  - Anomaly detection:
    * Use isolation forests to detect unusual patterns
```

### Debugging Model Issues

```
Problem: Model performance drops in production
Diagnostic Process:

1. Verify Data Integrity
   Q: Are features being computed correctly?
   A: Check feature store: compare serving features vs. training features
   Q: Are new labels coming in correctly?
   A: Spot check: for 100 random clicks, verify label = 1
   
2. Identify Data Distribution Shift
   Q: Has user/ad distribution changed?
   A: Compare feature distribution: current week vs. baseline
   Q: Did a major event happen? (outage, viral trend)
   A: Cross-check with business metrics: CTR across all ads
   
3. Verify Model Deployment
   Q: Is the new model actually being served?
   A: Log model version per prediction, verify in logs
   Q: Did quantization introduce errors?
   A: A/B test: quantized vs. non-quantized on small traffic
   
4. Revert or Retrain
   Q: Should we revert to old model?
   A: If issue is production bug: YES, revert immediately
      If issue is data shift: NO, retrain instead
   Q: How quickly can we retrain?
   A: Ideally within 2-4 hours using incremental training
   
5. Post-Mortem
   - Document: what happened, why, how we fixed it
   - Implement safeguards: if O/E drops >10%, pause model
   - Update monitoring: add alerts for early warning signs
```"

### Interviewer
"Great comprehensive evaluation framework. Now let me ask a slightly different angle: **what are the key technical challenges you'd face in production, and how would you solve them?**"

### Candidate
"Excellent question. Let me think through the major production challenges:

### Production Challenges & Solutions

```
Challenge 1: FEATURE CONSISTENCY (Training vs Serving)

Problem:
  - In training: compute user_ctr from historical data at training time
  - In serving: compute user_ctr from history up to inference time
  - Result: Feature values differ between training & serving
  - Model sees different feature values at test time
  - This is called "feature leakage" or "training-serving skew"

Real Example (LinkedIn):
  - Training O/E ratio: 0.85
  - Serving O/E ratio: 0.65
  - Difference caused by features computed differently

Solution: Nearline Feature Attribution
  1. During serving inference, log the exact features used
  2. Later, when label arrives (click/no-click), join with features
  3. Use these pre-joined data for training
  4. Guarantees feature consistency
  
Implementation:
  Serving System:
    └─> [user_features, ad_features] → model → prediction
    └─> LOG: {user_id, ad_id, [features_used], timestamp}
  
  Labeling System (next day):
    └─> [serving logs] JOIN [click logs] → [features, label]
    └─> Upload to training data store
  
  Training System:
    └─> Read pre-joined data (features already computed consistently)
    └─> Train model
  
  Benefit: eliminate training-serving skew
  Cost: 1-day delay in training data availability
```

```
Challenge 2: DELAYED FEEDBACK (Clicks vs. Conversions)

Problem:
  - Click happens immediately (can train same day)
  - Conversion (purchase) might take days/weeks
  - During initial training, most conversions aren't labeled yet
  - Model trained on incomplete data

Timeline Example:
  Day 1: User clicks on ad (immediate label available)
  Day 2: Training starts (conversion not happened yet, labeled as 0)
  Day 3-7: User finally converts (but model already trained)
  Result: We trained the model to NOT predict this user as converter
  
Impact:
  - Models converge too fast on incomplete labels
  - Might learn spurious patterns from day-1 labels

Solution: Delayed Feedback Loss Function (from Twitter)
  
  Instead of:
    L = -log(ŷ) if clicked, -log(1-ŷ) if not clicked
  
  Use:
    L = -log(ŷ) if clicked OR converted
        -log(1-ŷ) if clicked but NOT converted (yet)
        (ignore until label arrives)
  
  In practice: down-weight unlabeled examples
  
  Mathematical formulation:
    Loss_delayed(y_t, ŷ_t) = -w_t * [y_t * log(ŷ_t) + (1-y_t) * log(1-ŷ_t)]
    
    where w_t = 0 if example still missing labels
               = 1 once all labels arrived
  
  Twitter's results: 2.99% improvement in RCE (Relative Cross Entropy)
```

```
Challenge 3: MODEL STALENESS & ONLINE TRAINING

Problem:
  - Retrain once daily: model is 24 hours old
  - Ads change rapidly, user interests change
  - 24-hour-old model is suboptimal
  
Solution: Online Learning / Incremental Training

Approach:
  - Traditional batch training: train from scratch every 24h on 30 days data
  - Online training: continuous updates on streaming data
  
  Gradient-based update:
    new_weights = old_weights - α * gradient(new_example)
  
  Challenges of online training:
    1. Catastrophic forgetting: never seeing historical patterns
    2. Concept drift: sudden changes in distribution
    3. Non-stationary data: CTR changes by time of day
  
  LinkedIn's solution:
    - Base model: trained daily on 30 days data (strong foundation)
    - Online updates: hourly updates on recent data only
    - Warm-start: use yesterday's weights as initialization
    - Incremental training: only update relevant weights
    
  Benefits:
    - Model stays fresh (updated hourly vs. daily)
    - Faster reaction to trends
    - Meta result: >4% lift in CTR for ads models
    
  Trade-offs:
    - More complex engineering
    - Need infrastructure to handle continuous data streams
    - Risk of instability if not tuned properly
```

```
Challenge 4: MODEL CALIBRATION AT SCALE

Problem:
  - Model predicts probabilities, but probabilities might be biased
  - Predicts 0.7 CTR but actual is 0.5
  - In auction systems, probability must be accurate (money involved)
  - Can't use simple calibration post-hoc (too slow)

Solution: Online Calibration Pipeline (LinkedIn's approach)

Offline calibration (batch):
  1. Take validation set
  2. Compute predictions
  3. Fit isotonic regression: f(pred) = calibrated_pred
  4. Store mapping table: {0.0→0.01, 0.1→0.08, 0.2→0.15, ...}
  5. Apply mapping at serving time
  
  Problem: takes hours to run after training
  
Online calibration:
  1. Collect recent examples with both prediction and label
  2. Continuously update calibration mapping
  3. Parallelized computation: multiple workers update independently
  4. Merge updates: combine partial calibration updates
  
  LinkedIn's results:
    - Before: O/E ratio = 0.65±0.1 (significantly miscalibrated)
    - After: O/E ratio = 1.0±0.1 (well-calibrated)
    
  Practical impact:
    - Better auction bid calculations
    - Advertisers make better budget allocation decisions
    - More efficient budget utilization overall
```

```
Challenge 5: COLD-START PROBLEM

Sub-problem 1: New Users
  - No historical click data
  - Embedding can't be pre-trained
  
  Solution:
    - Use demographic features: age, location, device
    - Use content-based matching: show popular ads
    - Exploration: "bandit" approach to explore user's preferences
    - Fallback to average embeddings weighted by demographics
  
  Implementation:
    - Segment users: [age<18, 18-25, 25-35, 35-50, 50+] × [M/F] × [location]
    - For each segment, compute average user embedding
    - Use as default for new users
    - Update to personalized embedding once they have interactions

Sub-problem 2: New Ads
  - No historical CTR
  - Embedding not stable
  
  Solution: Meta-Learning (from Alibaba's "Warm Up Cold-start Ads" paper)
    - Learn how to learn embeddings from similar ads
    - When new ad arrives: use gradient-based meta-learning
    - "Few-shot learning": adapt embeddings with few examples
    - Algorithm: MAML (Model-Agnostic Meta-Learning)
    
  Practical improvement:
    - Warm-up phase: new ads shown mostly to relevant segments
    - After 1000 impressions: embeddings stabilize
    - Expected lift: 2-5% improvement on new ad CTR

Sub-problem 3: New Advertisers
  - No historical campaign performance
  - Quality score unknown
  
  Solution:
    - Transfer learning: use advertiser's ad creative features
    - Quality score: infer from similar advertisers
    - Conservative bidding: lower initial bids until data accumulates
    - After 100k impressions: model has sufficient data
```

```
Challenge 6: COMPUTATIONAL EFFICIENCY AT SCALE

Problem:
  - 10 billion impressions / day
  - 50-100ms latency per user request
  - Must rank millions of users concurrently
  
Solution: Infrastructure optimizations

1. GPU Inference Servers
   - Deploy model on GPUs (not CPUs)
   - Batch inference: 100-1000 examples at once
   - Throughput: 100-500 ads/ms (vs. 5-10 ads/ms on CPU)
   
2. Model Serving Framework
   - Use TensorFlow Serving / TorchServe
   - Adaptive batching: wait up to 10ms to collect batch
   - Request queue: prioritize by request importance
   
3. Feature Caching
   - Pre-fetch top 1000 users' features in Redis
   - Hit rate: 70-80% (popular users repeat)
   - Misses fetch from feature store (slower but rare)
   
4. Model Compression
   - Quantization: float32 → float16 (2X faster, minimal accuracy loss)
   - Pruning: remove unimportant weights (reduces model size)
   - Distillation: train small model to mimic large model
   
5. Approximate Algorithms
   - ANN instead of brute-force nearest neighbor search
   - LSH (Locality Sensitive Hashing) for fast retrieval
   - Trade-off: 99% recall @ 100X faster
```

```
Challenge 7: PREVENTING ADVERSARIAL BEHAVIOR

Problem:
  - Advertisers could manipulate system:
    * Click farms: artificially inflate CTR with fake clicks
    * Fraud: click own ads to waste competitor budgets
  - Malicious users could exploit:
    * Spam: show ads to wrong demographics
    * Privacy abuse: correlate ad displays with user data

Solution: Fraud Detection & Anti-Abuse System

Detection:
  1. Click fraud detection
     - Pattern analysis: user clicking same advertiser repeatedly
     - Device fingerprinting: detect bot networks
     - IP reputation: block known proxy/VPN IPs
     - Time patterns: clicks at unnatural intervals
     
     Algorithm: Isolation Forest or One-Class SVM
     - Train on known fraud patterns
     - Score new clicks as normal vs. anomalous
  
  2. Privacy safeguards
     - Differential privacy: add noise to sensitive queries
     - Federated learning: train models without centralizing user data
     - Aggregation: report metrics, not individual-level data

Prevention:
  - Enforce minimum click intervals per user
  - Limit impressions per ad per user (frequency capping)
  - Require human review for suspicious advertisers
```"

### Interviewer
"Excellent production-ready thinking. You've covered the major gotchas. Let me end with one final question: **What would you do in the first 3 months, 6 months, and 1 year to maximize impact?**"

### Candidate
"Great closing question. Let me frame this as a roadmap:

```
MONTH 1-3: FOUNDATION & BASELINE

Goals:
  - Establish baseline metrics
  - Build reliable training pipeline
  - Deploy initial model to canary traffic

Week 1-2: Understand Existing System
  - Meet with teams (ML, infra, ads ranking)
  - Study production CTR model: architecture, performance
  - Understand feature store: what features exist
  - Analyze bottlenecks: where is latency coming from?

Week 2-4: Baseline Model & Offline Evaluation
  - Replicate existing CTR model (DeepFM or similar)
  - Implement offline evaluation: AUC, log loss, segment analysis
  - Verify features: check for data quality issues
  - Establish baseline: current AUC = 0.805 (example)

Week 4-8: Infrastructure & Data Pipeline
  - Build nearline feature attribution system
  - Implement feature store queries: latency <10ms
  - Set up monitoring dashboard: CTR, calibration, data drift
  - Create A/B testing framework

Week 8-12: First Improvement & Canary Deployment
  - Hypothesis: user sequence matters (add DIN or attention)
  - Offline improvement: AUC 0.805 → 0.812 (+0.007)
  - Online test: 5% canary traffic
  - If +0.1% CTR lift with no guardrail drops: promote gradually

Expected impact: Foundation in place, first +0.1-0.2% CTR lift

MONTH 3-6: RAPID ITERATION & ONLINE LEARNING

Goals:
  - Ship 3-4 model improvements
  - Implement online learning for freshness
  - Scale to 50-100% traffic

Iteration 1: Deep Interest Network
  - Problem: user interests change within session
  - Solution: attention mechanism over click history
  - Expected: +0.15% CTR lift
  - Timeline: 3 weeks

Iteration 2: Feature Engineering (Auto Feature Cross)
  - Problem: manual feature crosses are suboptimal
  - Solution: model learns which crosses matter most
  - Use DCN (Deep Cross Network) for automated crosses
  - Expected: +0.1-0.2% CTR lift
  - Timeline: 3 weeks

Iteration 3: Multi-Task Learning
  - Problem: only predicting CTR, ignoring conversions
  - Solution: joint CTR + CVR prediction model
  - Share embeddings, separate heads
  - Expected: +0.05% CTR, +0.2% CVR lift
  - Timeline: 4 weeks

Iteration 4: Online Learning
  - Problem: daily model is stale, trends change hourly
  - Solution: incremental training on recent data
  - Retrain every hour, use previous model as init
  - Expected: +0.1-0.3% CTR from freshness
  - Timeline: 4-5 weeks (complex engineering)

Cumulative impact: +0.4-0.65% CTR lift
Revenue impact: at Meta scale, 0.5% lift = $5-10M incremental annual revenue

MONTH 6-12: OPTIMIZATION & ADVANCED TECHNIQUES

Goals:
  - Mature system: stable, well-monitored
  - Implement advanced techniques: lifelong learning, bandits
  - Cross-platform improvements

Iteration 5: Contextual Multi-Armed Bandit
  - Problem: exploration-exploitation trade-off
  - Solution: Thompson sampling for new ads/users
  - Route some impressions to explore (learn fast)
  - Expected: +0.05% CTR, better cold-start performance
  - Timeline: 5-6 weeks

Iteration 6: Graph Neural Networks
  - Problem: user-ad graph has rich signal
  - Solution: GNN to propagate information: user → similar users → similar ads
  - Co-purchase patterns, similar interests
  - Expected: +0.1% CTR
  - Timeline: 6-8 weeks (research + implementation)

Iteration 7: Federated Learning
  - Problem: privacy concerns, centralized data
  - Solution: train model on-device with federated averaging
  - Improves user privacy, learns on-device patterns
  - Expected: +0.05-0.1% CTR, significant privacy gains
  - Timeline: 8-10 weeks (R&D heavy)

Iteration 8: Reinforcement Learning
  - Problem: short-term CTR optimization might hurt long-term user satisfaction
  - Solution: RL to optimize for session-level metrics
  - Maximize: (CTRs, user_engagement, advertiser_roi)
  - Expected: +0.1% CTR, +3% user satisfaction
  - Timeline: 10-12 weeks (experimental)

Cumulative impact: +0.7-1.2% CTR lift over baseline
Year 1 revenue impact: $10-20M incremental

YEAR 1 SUCCESS METRICS:
✓ CTR improvement: +0.8-1.0% (from 1.5% → 1.51-1.515%)
✓ Model quality: AUC 0.805 → 0.820+
✓ System stability: 99.99% uptime, <100ms latency p99
✓ Data-driven team: quarterly model reviews, healthy A/B testing culture
✓ Operational excellence: automated retraining, monitoring, quick iteration
```"

### Interviewer
"Excellent roadmap thinking. You've demonstrated strong understanding of ML systems, from requirements through deployment and iteration. Let me ask one final probe before we wrap: **What would you prioritize if you could only work on ONE thing in year 1?**"

### Candidate
"Great question to force prioritization. I'd pick **Nearline Feature Attribution & Feature Store Optimization**.

Here's why:

**Impact:**
- Fixes training-serving skew (a fundamental issue affecting all models)
- Enables all future improvements (every model benefits)
- Improves calibration: O/E ratio 0.65 → 1.0 (real LinkedIn example)
- Unlocks daily retraining on consistent features

**Effort vs. Reward:**
- Engineering effort: significant but contained (6-8 weeks)
- Data science effort: low (mostly verification)
- ROI: 0.2-0.3% CTR improvement immediately
- Foundation: enables all future model improvements

**Multiplier Effect:**
- Once feature consistency is solved, the next model improvement gives its full expected benefit
- Without this, every model improvement is partially wasted (5-10% of benefit lost to train-serve skew)
- It's like removing friction: same engine produces more output

**Alternative answer:**
If forced to pick a modeling improvement: **Online Learning**
- Reason: freshness is critical in social platforms
- User interests change hourly, ads evolve rapidly
- 24-hour-old model inherently suboptimal
- Expected ROI: 0.2-0.3% CTR lift, enables rapid iteration

**Risk mitigation:**
- Start with 5% canary traffic while debugging
- Monitor O/E ratio closely
- Keep old model as fallback

That said, I'd probably do both in parallel across month 1-3 since they're somewhat independent. Feature infrastructure is foundational, online learning is the next lever for production performance."

### Interviewer
"Perfect answer. You've shown deep technical expertise, product thinking, and engineering pragmatism. I'm impressed. Let me ask if you have any questions for me?"

### Candidate
"Yes, great questions:

1. **What does the current production CTR model architecture look like?** (To understand what's already optimized vs. new opportunities)

2. **What's the main bottleneck today?** Is it model quality, infrastructure latency, or something else? (Helps me focus first efforts)

3. **What were the biggest learnings from past CTR model improvements?** (What worked, what didn't)

4. **What's the scale of the system?** Specifically: daily impressions, users, ads. (Helps me estimate infrastructure needs)

5. **What's the organization structure?** How are ML, infra, and ads ranking teams organized? (Understanding dependencies)

6. **What metrics matter most to the business?** CTR, CVR, ROAS, user engagement? (To align priorities correctly)

I'm ready to start contributing from day 1 and excited about the technical challenges ahead."

---

## SUMMARY: KEY TAKEAWAYS

**Requirements Phase:**
✓ Clarify latency (50-100ms), scale (10B impressions/day, 100M users)
✓ Understand business metrics (CTR, CVR, user satisfaction)
✓ Identify data challenges (class imbalance 1-2% CTR, feedback delay)
✓ Get feature scope (user, ad, contextual features)

**High-Level Architecture:**
✓ Two-stage serving: lightweight pre-ranker + deep ranker
✓ Separate training & serving pipelines
✓ Feature store with batch + online features
✓ Monitoring & feedback loops

**Deep Dive Topics:**
✓ Feature store design: batch (S3) + online (Redis)
✓ Model architecture: DeepFM as baseline, DIN for sequences
✓ Serving optimization: two-tower embeddings + ANN search
✓ Offline & online evaluation: AUC, log loss, O/E calibration

**Production Challenges:**
✓ Feature consistency: use nearline feature attribution
✓ Delayed feedback: adjust loss function
✓ Model staleness: online incremental learning
✓ Calibration: online calibration pipeline
✓ Cold-start: meta-learning, transfer learning

**Roadmap:**
✓ Month 1-3: foundation + baseline + first improvement (+0.1%)
✓ Month 3-6: rapid iteration + online learning (+0.5%)
✓ Month 6-12: advanced techniques + optimization (+1.0%)
✓ Year 1 target: +0.8-1.0% CTR improvement
