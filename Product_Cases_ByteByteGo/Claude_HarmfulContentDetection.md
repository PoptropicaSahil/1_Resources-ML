# ML System Design Interview: Harmful Content Detection for Social Media

---

## Phase 1 — Problem Clarification & Scoping (5–7 min)

**Interviewer:** Design a system to detect harmful content on a social media platform like Instagram. Walk me through how you'd approach this.

**Candidate:** Before I dive in, I'd like to ask a few clarifying questions to scope the problem.

**First — what modalities are we covering?** Instagram has text (captions, comments, DMs), images, videos (Reels, Stories), and audio. Are we designing for all of them, or starting with a subset?

**Interviewer:** Good question. Let's say we need a system that handles text, images, and short-form video. Audio can be secondary.

**Candidate:** Got it. **Second — what categories of harm are in scope?** Harmful content is broad. I'm thinking of a taxonomy like:

| Category | Examples |
|---|---|
| **Violence & Gore** | Graphic imagery, threats of violence |
| **Hate Speech** | Slurs, dehumanization based on protected attributes |
| **Nudity / Sexual Content** | CSAM (highest severity), adult nudity |
| **Self-Harm / Suicide** | Promotion of self-injury, suicide methods |
| **Bullying & Harassment** | Targeted personal attacks, doxxing |
| **Misinformation** | Health disinfo, election manipulation |
| **Spam / Scam** | Phishing links, engagement bait |

Are all of these in scope, or should I prioritize?

**Interviewer:** Let's say all are in scope but CSAM and violence are the highest priority — zero tolerance.

**Candidate:** Understood. That priority ordering will directly impact my threshold tuning later. **Third — what's the scale?**

I'll assume Instagram-like numbers:

- **~2B monthly active users**
- **~100M+ pieces of content posted per day** (images, reels, stories, text)
- **~2B+ comments per day**
- Peak traffic is roughly **~3–5× average** (events, holidays)

**Interviewer:** Those are reasonable. What about latency?

**Candidate:** That's my next question. I see two regimes:

- **Pre-publication (upload-time):** Content is screened before it goes live. Latency budget: **~200–500ms** is acceptable because the user is already waiting for an upload. This is where we catch the worst stuff.
- **Post-publication (async):** A more thorough sweep on content already live. Can tolerate **seconds to minutes**. Catches things the fast model missed.

I'll also assume we need a **human review queue** for borderline cases and appeals. The system doesn't just make binary decisions — it produces a **risk score** and routes to the appropriate action.

**Interviewer:** Good framing. Let's proceed.

---

> ### 🧠 Mental Model: The "Scoping Triangle"
>
> Every ML system design starts by nailing down three axes:
>
> 1. **What** — Taxonomy of outputs (harm categories)
> 2. **Where** — Modalities and surfaces (text, image, video, comments, DMs)
> 3. **How fast** — Latency regime (real-time vs. batch)
>
> Getting these wrong means you'll design the wrong system. Interviewers want to see that you don't just jump into model architecture.

---

## Phase 2 — Metrics Definition (5–7 min)

**Interviewer:** How would you measure success for this system?

**Candidate:** I think about metrics at three levels: **offline model metrics**, **online system metrics**, and **business/trust metrics**.

### 2.1 Offline Metrics

For a content moderation system, **precision and recall have asymmetric costs**, and the direction of asymmetry depends on the harm category.

For **CSAM and violence (zero-tolerance categories)**:

$$\text{We optimize for Recall} = \frac{TP}{TP + FN}$$

A false negative (harmful content staying live) is catastrophic — legal liability, user trauma, regulatory risk. We accept more false positives (over-removal) and route them to human review.

**Target: Recall ≥ 0.99, even if Precision drops to ~0.7**

For **spam and borderline content**:

$$\text{We optimize for Precision} = \frac{TP}{TP + FP}$$

Over-removing benign content creates a censorship perception. False positives here erode user trust.

**Target: Precision ≥ 0.95, Recall ~0.85**

Now, a single threshold on precision/recall is limiting. I prefer to look at **the full picture**:

**AUC-PR (Area Under Precision-Recall Curve)** is my go-to over AUC-ROC here because our dataset is extremely imbalanced. If only 0.1% of content is harmful, AUC-ROC can look great even with a bad model because the true negative rate dominates.

$$\text{AUC-PR} = \int_0^1 P(r) \, dr$$

This gives a threshold-independent view of model quality on the minority (harmful) class.

I'd also track **Recall@FPR** — e.g., what's my recall when my false positive rate is 1%? This directly maps to how many innocent posts I'm over-moderating.

### 2.2 Online / System Metrics

| Metric | Definition | Target |
|---|---|---|
| **Prevalence** | % of views on harmful content that escaped detection | < 0.05% of views |
| **Time-to-Action** | Time from upload to enforcement action | < 1 min (severe), < 1 hr (moderate) |
| **Appeal Overturn Rate** | % of removed content restored on appeal | < 5% |
| **Human Review Bandwidth** | % of total content routed to human reviewers | < 2–3% |
| **P99 Latency** | Inference latency at 99th percentile | < 300ms (pre-pub) |

### 2.3 Business / Trust Metrics

- **User-reported harmful content** (should decrease over time)
- **Creator satisfaction scores** (should not degrade from over-moderation)
- **Regulatory compliance rate** (different in EU/DACH vs US)

**Interviewer:** Why not just use F1?

**Candidate:** F1 is the harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

It treats precision and recall symmetrically, but our problem is fundamentally asymmetric. A missed piece of CSAM is not equivalent to a wrongly removed meme. I'd use **Fβ scores** where β encodes the asymmetry:

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

- For zero-tolerance categories: **F₂** (recall weighted 2× more than precision)
- For spam/borderline: **F₀.₅** (precision weighted 2× more than recall)

---

> ### 🧠 Mental Model: "Asymmetric Cost Matrix"
>
> Whenever you're designing a classification system, ask: **"What's more expensive — a false positive or a false negative?"** Then choose your metric and threshold accordingly:
>
> | | FN is worse → Optimize Recall | FP is worse → Optimize Precision |
> |---|---|---|
> | **Examples** | CSAM, violence, fraud detection | Spam filtering, content recommendations |
> | **Metric** | F₂, Recall@K | F₀.₅, Precision@K |
> | **Threshold** | Lower (more aggressive) | Higher (more conservative) |
>
> This asymmetry should cascade into your loss function, threshold tuning, and human review routing.

---

## Phase 3 — High-Level System Architecture (8–10 min)

**Interviewer:** Walk me through the end-to-end system architecture.

**Candidate:** I'll design a **multi-stage cascaded pipeline**. The core insight is: **not every piece of content needs the same computational budget**. Most content is benign — we want to reject obviously safe content cheaply and spend expensive compute only on ambiguous cases.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTENT INGESTION LAYER                        │
│  (Image/Video/Text arrives via Upload API or Comment API)          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 1: LIGHTWEIGHT FILTER                        │
│  ─ Perceptual hashing (PhotoDNA for CSAM)                         │
│  ─ Blocklist/regex matching                                       │
│  ─ Small distilled classifier (~5M params)                        │
│  ─ Latency: < 20ms | Filters ~85-90% of content as SAFE          │
└──────────┬──────────────────────────┬───────────────────────────────┘
           │ SAFE → Publish           │ FLAGGED / UNCERTAIN
           ▼                          ▼
    ┌──────────┐     ┌─────────────────────────────────────────────┐
    │   LIVE   │     │       STAGE 2: DEEP ANALYSIS                │
    │ CONTENT  │     │  ─ Multi-modal transformer ensemble         │
    └──────────┘     │  ─ Text: fine-tuned RoBERTa/DeBERTa         │
                     │  ─ Image: ViT / EfficientNet                │
                     │  ─ Video: frame sampling + temporal model    │
                     │  ─ Multi-task heads per harm category        │
                     │  ─ Latency: 100–300ms                       │
                     └──────┬───────────────┬───────────────────────┘
                            │               │
                   HIGH CONFIDENCE    LOW CONFIDENCE
                   (score > τ_high    (τ_low < score < τ_high)
                    or < τ_low)
                            │               │
                            ▼               ▼
                   ┌──────────────┐  ┌─────────────────────────┐
                   │  AUTO-ACTION │  │  STAGE 3: HUMAN REVIEW  │
                   │  (Remove /   │  │  ─ Priority queue by    │
                   │   Approve)   │  │    severity score       │
                   └──────────────┘  │  ─ Specialist routing   │
                                     │  ─ Reviewer consensus   │
                                     └─────────────────────────┘
```

### 3.1 Stage 1 — Lightweight Filter (The Bouncer)

This is the first line of defense. Its job is to be **fast and cheap**, not perfect.

**Perceptual Hashing (PhotoDNA / pDNA):**
For known CSAM and terrorist content, we don't need a classifier at all. We compute a perceptual hash of the uploaded image and match it against a database of known harmful images (maintained by NCMEC, GIFCT, etc.).

Perceptual hashes are robust to resizing, compression, and minor edits. The match is near-instantaneous (hash table lookup).

**Text Blocklists + Lightweight Classifier:**
A small distilled model (e.g., a 6-layer DistilBERT with ~5M parameters or even a fastText model) catches obvious hate speech, known slur patterns, and spam signatures.

**Why a cascade?** If we ran our full multi-modal transformer on every single piece of content at 100M+ posts/day, the compute cost would be enormous. The cascade exploits a power law: **~90% of content is obviously safe**, so we only need the heavy model for ~10%.

### 3.2 Stage 2 — Deep Analysis (The Specialist)

This is the core ML system. I'll go deep on this in the next phase.

### 3.3 Stage 3 — Human Review (The Judge)

Content in the "gray zone" (model is uncertain) gets routed to human reviewers.

Key design decisions:
- **Priority queue**: Sorted by `severity_category × (1 - model_confidence)`. CSAM with even low confidence jumps to the top.
- **Specialist routing**: Hate speech reviewers see hate speech; they develop calibration.
- **Inter-annotator agreement**: Each item reviewed by 3+ annotators. Majority vote for action. This also generates **high-quality training data** for model retraining.

### 3.4 Post-Publication Sweep (Batch Pipeline)

A separate batch pipeline re-scans live content periodically. Why?

1. **Model updates**: New model catches things old model missed
2. **Context evolution**: A benign hashtag can get co-opted (e.g., #thinspo)
3. **Viral content**: Something going viral deserves re-examination
4. **User reports**: Reported content gets priority re-scoring

**Interviewer:** How do you handle the cold-start problem for new types of harmful content that don't exist in your training data?

**Candidate:** Great question. This is the **distribution shift** problem. Three strategies:

1. **Few-shot / zero-shot classifiers**: Use a large language model (or CLIP for images) with prompt-based classification. "Does this image depict [new harm category]?" — no retraining needed.
2. **Anomaly detection**: Flag content that is distributionally unusual (high embedding distance from known clusters) for human review.
3. **Trend monitoring**: Track emerging hashtags, phrases, and visual memes. If a new hashtag suddenly co-occurs with known harmful content, flag the cluster for review.

---

> ### 🧠 Mental Model: "The Cascade Principle"
>
> In production ML, **you almost never serve a single model**. You build a cascade:
>
> **Cheap & Fast → Expensive & Accurate → Human**
>
> Each stage has increasing cost and decreasing volume. Design the system so the expensive stages only process a small fraction of total traffic. This is the same principle behind search ranking (retrieval → ranking → re-ranking) and recommendation systems.
>
> The key parameters to tune are the **pass-through rates** between stages. If Stage 1 passes too much, you overwhelm Stage 2. If it's too aggressive, you get false removals.

---

## Phase 4 — Model Architecture Deep Dive (12–15 min)

**Interviewer:** Let's go deep on the Stage 2 model. What architecture would you use?

**Candidate:** The fundamental challenge is that harmful content is **multi-modal and context-dependent**. A caption that says "I'm going to kill it today" is benign motivation. With an image of a weapon, it's a threat. So I need a model that can reason across modalities jointly.

### 4.1 Architecture Overview

```
                    ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
                    │  TEXT INPUT  │  │ IMAGE INPUT │  │ VIDEO INPUT  │
                    │  (caption,   │  │ (uploaded   │  │ (sampled     │
                    │   comment)   │  │  image)     │  │  frames)     │
                    └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
                           │                │                │
                           ▼                ▼                ▼
                    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                    │   Text       │ │   Vision     │ │   Video      │
                    │   Encoder    │ │   Encoder    │ │   Encoder    │
                    │  (DeBERTa    │ │  (ViT-L/14)  │ │  (TimeSformer│
                    │   V3-large)  │ │              │ │   or frame   │
                    │              │ │              │ │   sampling + │
                    │              │ │              │ │   ViT)       │
                    └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
                           │                │                │
                      d=1024           d=1024           d=1024
                           │                │                │
                           └────────┬───────┘────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   FUSION MODULE       │
                        │   Cross-Attention      │
                        │   Transformer          │
                        │   (4–6 layers)         │
                        └───────────┬───────────┘
                                    │
                              d_fused=1024
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  MULTI-TASK HEADS     │
                        │                       │
                        │  ┌─────────────────┐  │
                        │  │ Violence   → σ  │  │
                        │  │ Hate Speech→ σ  │  │
                        │  │ Nudity     → σ  │  │
                        │  │ Self-Harm  → σ  │  │
                        │  │ Bullying   → σ  │  │
                        │  │ Spam       → σ  │  │
                        │  │ Severity   → [1-5]│ │
                        │  └─────────────────┘  │
                        └───────────────────────┘
```

### 4.2 Unimodal Encoders

**Text Encoder — DeBERTa V3 Large**

Why DeBERTa over BERT? DeBERTa uses **disentangled attention** — it separates content and position embeddings and computes attention between them independently:

$$A_{i,j} = \underbrace{H_i^c {H_j^c}^T}_{\text{content-to-content}} + \underbrace{H_i^c {P_{i|j}}^T}_{\text{content-to-position}} + \underbrace{P_{j|i} {H_j^c}^T}_{\text{position-to-content}}$$

where $H^c$ are content vectors and $P$ are relative position embeddings. This is important for content moderation because **position matters** — "I hate you people" vs. "you people, I hate to say this but..." have different meanings based on syntactic structure.

DeBERTa V3 also uses **replaced token detection** (from ELECTRA) during pre-training, which is more sample-efficient than masked LM.

**Vision Encoder — ViT-Large**

I'd use a ViT-L/14 pre-trained via CLIP or DINOv2. The advantage of CLIP pre-training is that the vision encoder already has **semantic alignment with text**, which helps downstream fusion.

For a 224×224 image with 14×14 patches:

$$\text{Sequence length} = \frac{224}{14} \times \frac{224}{14} + 1 = 257 \text{ tokens (including [CLS])}$$

Each token is projected to $d = 1024$ and processed through 24 transformer layers.

**Video Encoder — Efficient Frame Sampling + ViT**

For short-form video (Reels, <60s), running a full video transformer on every frame is prohibitively expensive. My approach:

1. **Uniform temporal sampling**: Extract $K = 8$ frames evenly spaced across the video.
2. **Scene-change detection**: If a sharp visual change is detected (histogram difference > threshold), sample additional frames around that point.
3. **Per-frame ViT encoding**: Each frame → ViT → embedding.
4. **Temporal aggregation**: A lightweight temporal transformer (2 layers) over the $K$ frame embeddings to capture temporal relationships.

This is inspired by the **TimeSformer** "divided space-time attention" approach but cheaper — we avoid the quadratic cost of full spatiotemporal attention.

### 4.3 Multi-Modal Fusion

**Interviewer:** You mentioned cross-attention fusion. Why not just concatenate the embeddings?

**Candidate:** Great question. There are three main fusion strategies, each with tradeoffs:

**Strategy 1: Late Fusion (Concatenation)**
```
z_fused = MLP([z_text; z_image; z_video])
```
- ✅ Simple, modular, each encoder trainable independently
- ❌ Misses cross-modal interactions. "Kill" (text) + knife (image) = threat, but late fusion might not capture this synergy.

**Strategy 2: Early Fusion (Token-level)**
```
Concatenate all tokens from all modalities → one giant transformer
```
- ✅ Maximum expressiveness, can attend across all modality tokens
- ❌ Quadratic attention cost over combined sequence length. With 512 text tokens + 257 image tokens + 8×257 video tokens ≈ 2825 tokens → attention matrix is 2825² ≈ 8M entries per layer. Expensive.

**Strategy 3: Cross-Attention Fusion (my choice)**

Each modality attends to the others via cross-attention layers:

$$\text{CrossAttn}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}}) = \text{softmax}\left(\frac{Q_{\text{text}} K_{\text{image}}^T}{\sqrt{d_k}}\right) V_{\text{image}}$$

The text representation is enriched by attending to image features, and vice versa. This is a **bottleneck fusion** — we get cross-modal interaction without the quadratic cost of full early fusion.

I'd use **4–6 cross-attention layers** that alternate:
- Text attends to Image
- Image attends to Text
- Both attend to Video

This is similar to the **Flamingo** or **CoCa** architecture approach.

**Why this matters for harmful content**: Many adversarial attacks exploit single-modality models. Users post hate speech as text-in-image (screenshot of slur), or use coded language in captions that only becomes harmful with the accompanying image. Cross-attention lets the model reason about these interactions.

### 4.4 Multi-Task Classification Heads

Rather than training separate models per harm category, I use **multi-task learning** with shared representations:

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} w_t \cdot \mathcal{L}_t$$

where $T$ = number of harm categories and each $\mathcal{L}_t$ is binary cross-entropy:

$$\mathcal{L}_t = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i^t \log(\hat{y}_i^t) + (1 - y_i^t) \log(1 - \hat{y}_i^t) \right]$$

**Why multi-task?**

1. **Parameter efficiency**: Shared backbone, only separate classification heads (~1% of total params per head).
2. **Positive transfer**: Hate speech and bullying share linguistic features. Violence and gore share visual features. Shared representations learn these common patterns.
3. **Regularization**: Multi-task acts as implicit regularization — prevents any single task from overfitting.

**The weights $w_t$ are critical.** I'd set them based on:
- **Task importance**: CSAM gets higher weight than spam
- **Task difficulty**: Harder tasks (misinformation) may need up-weighted gradients
- **Dynamic weighting**: Use **uncertainty weighting** (Kendall et al., 2018):

$$w_t = \frac{1}{2\sigma_t^2}$$

where $\sigma_t$ is a learned task-specific uncertainty parameter. Tasks with higher uncertainty get lower weight, preventing noisy gradients from dominating.

### 4.5 Handling Class Imbalance

This is one of the biggest practical challenges. Harmful content is maybe **0.1–1%** of total content. Within that, CSAM is even rarer.

**Approach 1: Focal Loss**

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's predicted probability for the correct class. The $(1 - p_t)^\gamma$ term **down-weights easy (well-classified) examples** and focuses learning on hard examples. With $\gamma = 2$, an example classified with $p_t = 0.9$ contributes 100× less loss than one with $p_t = 0.1$.

**Approach 2: Stratified Sampling**
Oversample harmful content in training batches. Target a **harmful:benign ratio of ~1:5 to 1:10** in each batch, even though the natural ratio is ~1:1000.

**Approach 3: Two-Phase Training**
1. Pre-train on balanced data (heavily oversampled)
2. Fine-tune on natural distribution with adjusted thresholds

I'd combine Focal Loss + stratified sampling in practice.

---

> ### 🧠 Mental Model: "Fusion Strategy Selection"
>
> When you have multi-modal inputs, your fusion choice should depend on:
>
> | Factor | Late Fusion | Cross-Attention | Early Fusion |
> |---|---|---|---|
> | Cross-modal dependency | Low | Medium-High | High |
> | Compute budget | Low | Medium | High |
> | Modularity needs | High | Medium | Low |
> | Training data volume | Any | Medium+ | Large |
>
> For content moderation, cross-modal dependencies are high (text+image together define harm), so cross-attention is the sweet spot.

---

## Phase 5 — Training Pipeline (7–8 min)

**Interviewer:** How would you collect training data and set up the training pipeline?

**Candidate:** Data is the hardest part of content moderation. The model is only as good as the labels.

### 5.1 Data Sources

| Source | Volume | Quality | Bias Concern |
|---|---|---|---|
| **Human-labeled (from review queue)** | Medium | High (multi-annotator) | Selection bias — only flagged content |
| **User reports** | High | Noisy (many false reports) | Weaponized reporting |
| **Synthetic data** | Configurable | Medium | Distribution mismatch |
| **Public datasets** (HatEval, MMHS150K) | Fixed | Medium | Domain gap, outdated slurs |
| **Active learning samples** | Low but targeted | Very High | Focuses on model weakness |

### 5.2 Labeling Protocol

For high-quality labels, I'd use:

**Multi-annotator consensus:** Each example labeled by **3–5 annotators**. Use majority vote for the label, but critically, also **save the full distribution**.

If 3/5 annotators say "hate speech" and 2/5 say "not hate speech", the **soft label** is $y = 0.6$ rather than a hard $y = 1$. Training on soft labels with cross-entropy:

$$\mathcal{L} = -[0.6 \cdot \log(\hat{y}) + 0.4 \cdot \log(1 - \hat{y})]$$

This captures genuine **annotator disagreement**, which is a signal — it tells us the content is ambiguous, and the model should also be uncertain.

**Inter-annotator agreement**: Track **Cohen's Kappa** or **Krippendorff's Alpha** per category. If $\kappa < 0.6$, the labeling guidelines need refinement.

### 5.3 Active Learning Loop

The most valuable training examples are the ones the **model is uncertain about**. Active learning targets these.

```
┌───────────────┐
│ Unlabeled     │──── Model Inference ───→ Score each example
│ Content Pool  │                               │
└───────────────┘                               │
                                                ▼
                                     ┌────────────────────┐
                                     │ Acquisition Func   │
                                     │ (uncertainty,       │
                                     │  diversity,         │
                                     │  expected info gain)│
                                     └─────────┬──────────┘
                                               │
                                    Select top-K uncertain
                                               │
                                               ▼
                                     ┌────────────────────┐
                                     │  Human Labeling    │
                                     └─────────┬──────────┘
                                               │
                                               ▼
                                     ┌────────────────────┐
                                     │  Add to Training   │
                                     │  Set & Retrain     │
                                     └────────────────────┘
```

**Acquisition function — I'd use a combination of:**

1. **Uncertainty sampling**: Select examples where the model's predicted probability is closest to the decision boundary:

$$x^* = \arg\max_x \; H[\hat{y}(x)] = \arg\max_x \left[-\hat{p}\log\hat{p} - (1-\hat{p})\log(1-\hat{p})\right]$$

2. **Diversity sampling**: Use $k$-means on the embedding space to ensure we don't just label similar edge cases. Select one example per cluster.

### 5.4 Adversarial Training & Robustness

Bad actors will try to evade the model. Common attacks:

- **Text**: Leetspeak ("h4te"), Unicode homoglyphs (Cyrillic "а" replacing Latin "a"), whitespace injection
- **Image**: Overlay text on images, slight perturbations, steganography
- **Multi-modal**: Benign image + harmful text in separate channels

**Defense**: Include adversarial examples in training. Generate them via:

1. **Character-level perturbations**: Swap characters, insert zero-width Unicode
2. **Adversarial image patches**: PGD (Projected Gradient Descent) attacks on the image encoder
3. **Backtranslation**: Translate harmful text through another language and back to generate paraphrases

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y))$$

This is the **FGSM** (Fast Gradient Sign Method) perturbation. Including these in training makes the model robust to small evasion attempts.

### 5.5 Training Infrastructure

- **Distributed training** on 64–128 GPUs (A100/H100) using FSDP (Fully Sharded Data Parallel)
- **Mixed precision** (BF16) for ~2× throughput
- **Gradient accumulation** with effective batch size of 4096
- Training schedule: **Warmup (2K steps) → Cosine decay LR → Fine-tune heads**
- Full training run: **~3–5 days** for the deep model
- **Retraining cadence**: Weekly incremental (on new labeled data), monthly full retrain

---

## Phase 6 — Serving & Inference Optimization (5–7 min)

**Interviewer:** How do you serve this at 100M+ posts/day with sub-300ms latency?

**Candidate:** This is where the rubber meets the road. A big multi-modal transformer is expensive at inference time. Here's my optimization stack:

### 6.1 Model Optimization

**Knowledge Distillation:**
I'd train a **smaller student model** that mimics the large teacher:

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \hat{y}_S) + (1-\alpha) \cdot T^2 \cdot \text{KL}\left(\frac{\hat{y}_T}{T} \;\|\; \frac{\hat{y}_S}{T}\right)$$

where $T$ is the temperature (typically 3–5) that softens the teacher's probability distribution, and $\hat{y}_S, \hat{y}_T$ are student and teacher outputs.

The **student** is a smaller version — maybe a 6-layer ViT + 6-layer text encoder instead of 24+12. This cuts latency by 3–4× with typically <2% accuracy loss.

**Quantization:**
- INT8 quantization of weights and activations (post-training)
- Selective mixed-precision: Keep attention layers in FP16, quantize FFN layers to INT8

**Batched Inference:**
Use **dynamic batching** — accumulate requests for up to 10ms, then process as a batch. Even a batch of 16 amortizes the fixed costs of model loading.

### 6.2 Serving Architecture

```
                    ┌────────────────────┐
                    │    Load Balancer   │
                    └─────────┬──────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │  GPU Pod  │  │  GPU Pod  │  │  GPU Pod  │
         │  (8×H100) │  │  (8×H100) │  │  (8×H100) │
         │           │  │           │  │           │
         │ Model     │  │ Model     │  │ Model     │
         │ Replicas  │  │ Replicas  │  │ Replicas  │
         └──────────┘  └──────────┘  └──────────┘
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌────────────────────┐
                    │  Decision Engine   │
                    │  (threshold logic, │
                    │   action routing)  │
                    └────────────────────┘
```

**Key decisions:**
- **Model serving framework**: NVIDIA Triton or TorchServe with TensorRT optimization
- **Auto-scaling**: Scale GPU pods based on queue depth. During peak hours, scale up proactively.
- **Feature caching**: Cache embeddings for user profile features (account age, past violations) in Redis. These don't change per-post.
- **Async video processing**: Video is the most expensive modality. Process frames asynchronously — publish the post, process video in background, take action if flagged.

### 6.3 Latency Budget Breakdown

| Component | Latency (P99) |
|---|---|
| Request routing + preprocessing | ~10ms |
| Stage 1 (hash lookup + small model) | ~15ms |
| Image encoding (ViT, quantized) | ~40ms |
| Text encoding (DeBERTa, distilled) | ~25ms |
| Cross-attention fusion | ~30ms |
| Multi-task heads | ~5ms |
| Decision logic + action routing | ~5ms |
| **Total** | **~130ms** |

Comfortably under the 300ms budget, with headroom for tail latency.

---

> ### 🧠 Mental Model: "The Optimization Hierarchy"
>
> When optimizing ML inference, work **top-down**:
>
> 1. **Architecture** — Do you even need the big model? (Cascade eliminates 90% of traffic)
> 2. **Distillation** — Can a smaller model approximate it? (3–4× speedup)
> 3. **Quantization** — Can you use lower precision? (2× speedup)
> 4. **Runtime** — TensorRT, kernel fusion, batching (1.5–2× speedup)
> 5. **Hardware** — Better GPUs / specialized accelerators
>
> Each level is multiplicative. Architecture × Distillation alone might give 10–40× savings. Don't jump straight to "buy more GPUs."

---

## Phase 7 — Monitoring, Feedback Loops, & Edge Cases (5–7 min)

**Interviewer:** How do you ensure the system stays accurate over time?

**Candidate:** Content moderation is one of the most adversarial, non-stationary environments in ML. The distribution shifts constantly — new slang, new memes, new evasion tactics, new types of harm.

### 7.1 Monitoring Dashboard

I'd track the following in real-time:

**Model Health Metrics:**
- **Prediction distribution drift**: Use **Population Stability Index (PSI)** between today's score distribution and the baseline:

$$PSI = \sum_{i=1}^{B} (p_i^{\text{new}} - p_i^{\text{ref}}) \cdot \ln\left(\frac{p_i^{\text{new}}}{p_i^{\text{ref}}}\right)$$

If $PSI > 0.2$, the distribution has shifted significantly → trigger investigation.

- **Confidence calibration**: Are predicted probabilities well-calibrated? Use **Expected Calibration Error (ECE)**:

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

If the model says 80% probability of hate speech, ~80% of those should actually be hate speech.

**Operational Metrics:**
- Appeal overturn rate (spiking = model regression)
- Human reviewer agreement with model (dropping = drift)
- Latency percentiles (P50, P95, P99)
- Queue depth for human review (growing = model pushing too much to humans)

### 7.2 Feedback Loop Design

```
Model in Production
       │
       ├── Automated removals → Track appeal rate
       │                              │
       │                    User appeals → Human re-reviews
       │                                         │
       │                              ┌──────────┴──────────┐
       │                              │ Overturn = FP label  │
       │                              │ Uphold  = TP label   │
       │                              └──────────┬──────────┘
       │                                         │
       ├── Human review decisions ────────────────┤
       │                                         │
       │                              ┌──────────▼──────────┐
       │                              │  Training Data      │
       │                              │  Pipeline           │
       │                              └──────────┬──────────┘
       │                                         │
       └─── Weekly model retrain ◄───────────────┘
```

Every model decision generates a **feedback signal**. Appeals and human review decisions create **naturally labeled data** that flows back into the training pipeline. This is a self-improving loop.

**Danger**: Feedback loops can also reinforce biases. If the model disproportionately flags content in a particular language, those examples get over-represented in the retraining set, reinforcing the bias. **Countermeasure**: Monitor per-demographic-group false positive rates and add fairness constraints to retraining.

### 7.3 A/B Testing & Safe Deployment

Never deploy a new model to 100% of traffic immediately. I'd use:

1. **Shadow mode**: New model runs in parallel, scores are logged but not actioned. Compare with production model.
2. **Canary deployment**: Route 1% of traffic → 5% → 10% → 50% → 100%, monitoring appeal rates at each step.
3. **Guardrails**: If appeal overturn rate > threshold at any step, auto-rollback.

### 7.4 Critical Edge Cases

| Edge Case | Challenge | Mitigation |
|---|---|---|
| **Satire / Irony** | "I love how people think the earth is flat" — is this promoting flat earth or mocking it? | Context features: user history, comment thread, quote-tweet patterns |
| **Reclaimed language** | In-group use of slurs (e.g., within Black communities) | User demographic signals (carefully), community norms, engagement patterns |
| **Newsworthy violence** | War journalism, documentary content | Source verification, account type (news org vs. personal), content warnings vs. removal |
| **Art & Education** | Nudity in art history, medical images | Account type, caption context ("oil painting, 1862" vs. no context) |
| **Code-switching** | Harmful content in non-English languages, mixing languages | Multilingual models, per-language thresholds |
| **Evolving slang** | New coded language (e.g., "boogaloo", "🍕gate") | Trend monitoring, rapid few-shot adaptation |

**Interviewer:** How do you handle the fairness concern — making sure the model isn't biased against certain dialects like AAVE (African American Vernacular English)?

**Candidate:** This is a well-documented problem. NLP models trained on Standard English corpora flag AAVE text as more toxic. The approach:

1. **Disaggregated evaluation**: Compute precision/recall separately across demographic slices. If the FPR for AAVE text is 2× that of Standard English, we have a fairness gap.
2. **Counterfactual data augmentation**: Create pairs where only the dialect changes, not the meaning. Train the model to be invariant.
3. **Dialect-aware thresholds**: Adjust thresholds per-dialect to equalize FPR across groups (equalizing odds).
4. **Inclusive annotator pools**: Ensure labeling teams include speakers of diverse dialects.

This is related to the **equalized odds** fairness criterion:

$$P(\hat{Y}=1 | Y=0, A=a) = P(\hat{Y}=1 | Y=0, A=b) \quad \forall \; a, b$$

i.e., the false positive rate should be the same regardless of the protected attribute $A$.

---

> ### 🧠 Mental Model: "Non-Stationary Systems Need Closed Loops"
>
> Whenever your ML system operates in an adversarial or evolving environment, you need:
>
> 1. **Monitoring** that detects drift before it becomes a crisis
> 2. **Feedback loops** that turn model decisions into training data
> 3. **Safe deployment** that limits blast radius of bad models
> 4. **Fairness audits** that catch disparate impact across groups
>
> A model without monitoring is a ticking time bomb. A model without feedback loops gets stale. Design both from day one.

---

## Phase 8 — Wrap-Up & Summary (3 min)

**Interviewer:** Great discussion. Can you summarize the key design decisions and tradeoffs?

**Candidate:**

### Key Design Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| **Architecture** | Multi-stage cascade | Cost efficiency — heavy model on 10% of content |
| **Fusion strategy** | Cross-attention | Captures cross-modal harm signals without early fusion cost |
| **Multi-task vs separate models** | Multi-task | Parameter sharing, positive transfer, efficiency |
| **Loss function** | Focal loss + uncertainty-weighted multi-task | Handles imbalance + automatic task balancing |
| **Serving** | Distilled student model + INT8 quantization | Meets latency under 300ms |
| **Threshold strategy** | Per-category, asymmetric | CSAM: high recall / Spam: high precision |
| **Retraining** | Weekly incremental + monthly full | Tracks distribution shift without excessive cost |

### Key Tradeoffs I Navigated

1. **Safety vs. Freedom of Expression**: High recall catches more harm but also removes borderline creative expression. The human review queue is the pressure valve.
2. **Latency vs. Accuracy**: The cascade trades accuracy on Stage 1 (simple model) for speed, recovering accuracy in Stage 2 for ambiguous cases.
3. **Centralized vs. Per-Category Models**: Multi-task is more efficient but creates coupling — a bad update to hate speech detection could degrade violence detection. Mitigate with per-task eval gates in CI/CD.
4. **Automation vs. Human Judgment**: Fully automated moderation is fast but error-prone on nuance. Fully human moderation is accurate but doesn't scale. The system needs both.

### If I Had More Time, I'd Also Cover:

- **Explainability**: GradCAM for images, attention visualization for text — showing reviewers *why* the model flagged something
- **On-device pre-filtering**: Run a tiny model on-device before upload to warn creators proactively
- **Cross-platform intelligence sharing**: Hash databases shared between platforms (GIFCT)
- **Regulatory compliance pipeline**: Different content policies for EU (DSA), US, and other regions — policy-as-code layer on top of model scores

**Interviewer:** Thank you, that was very thorough.

---

## 📐 Reference: Key Equations Used

| Concept | Equation |
|---|---|
| Precision | $P = TP / (TP + FP)$ |
| Recall | $R = TP / (TP + FN)$ |
| $F_\beta$ score | $F_\beta = (1+\beta^2) \cdot PR / (\beta^2 P + R)$ |
| Binary Cross-Entropy | $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Focal Loss | $\mathcal{L} = -\alpha_t(1-p_t)^\gamma \log(p_t)$ |
| Distillation Loss | $\mathcal{L} = \alpha \cdot CE + (1-\alpha) \cdot T^2 \cdot KL(\hat{y}_T/T \| \hat{y}_S/T)$ |
| FGSM Attack | $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$ |
| PSI (Drift) | $PSI = \sum (p^{new}_i - p^{ref}_i) \cdot \ln(p^{new}_i / p^{ref}_i)$ |
| ECE (Calibration) | $ECE = \sum \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ |
| Equalized Odds | $P(\hat{Y}=1|Y=0,A=a) = P(\hat{Y}=1|Y=0,A=b)$ |

---

*This interview covers the expected depth for a Senior/Staff Data Scientist (L5/L6) at a FAANG company. The candidate demonstrated: problem scoping, metric selection with mathematical justification, end-to-end architecture design, deep modeling knowledge, production awareness, and fairness considerations.*
