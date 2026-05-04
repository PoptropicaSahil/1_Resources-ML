# ML System Design Interview: Harmful Content Detection on Social Platforms

> **Format:** Interviewer (I) ↔ Candidate (C) | **Level:** Senior MLE / Applied Scientist (L6+, ~10 YoE) | **Duration:** ~55 min

---

## Phase 1: Problem Scoping & Requirements (8 min)

**I:** We want to build a harmful content detection system for a large social media platform - think Meta/YouTube scale. Walk me through how you'd approach this.

**C:** Before I dive into modeling, I want to scope the problem carefully - these decisions will cascade through every architectural choice. Let me ask a few clarifying questions.

**First - what types of harmful content are in scope?** There are very different signal profiles for violence, nudity, hate speech, misinformation, spam, and cyberbullying. Are we building a unified system or targeting a specific category first?

**I:** Let's build a unified system that handles the main categories - violence, nudity/explicit content, and hate speech - as a starting point.

**C:** Good. Second - **what modalities does content include?** Text-only posts, images, videos, or fully multimodal (text + image + video in one post)?

**I:** Fully multimodal - users can post text, images, and videos, often in combination.

**C:** Third - **what's the action model?** Meaning, once we have a harm probability, what do we do - auto-remove with high confidence, demote in feeds at lower confidence, or route to human reviewers? This defines our precision/recall operating point.

**I:** A tiered action model: auto-remove above a high-confidence threshold, demote in feed in a mid-confidence band, and queue for human review otherwise.

**C:** Fourth - **what are the scale and latency assumptions?**

**I:** ~500M posts per day, roughly 10K posts/sec at peak. For newly uploaded posts, we have a couple of seconds before they're indexed and surfaced. Think of this as a near-real-time pipeline, not strictly synchronous.

**C:** Got it. Fifth - **do we have labeled data, and is there a user reporting mechanism?**

**I:** About 10K manually annotated posts per day from contracted reviewers, plus a much larger stream of user reports which are noisily labeled.

**C:** Perfect. Let me codify the requirements:

```
┌─────────────────────────────────────────────────────────────────┐
│                  System Requirements Summary                     │
├────────────────────┬────────────────────────────────────────────┤
│ ML Task            │ Multimodal multi-label classification       │
│ Input              │ Post (text + image + video + metadata)      │
│ Output             │ P(violence), P(nudity), P(hate_speech), ... │
│ Scale              │ 500M posts/day, ~10K QPS peak               │
│ Latency            │ Seconds (async post-upload pipeline)        │
│ Labeled data       │ 10K/day manual + large noisy user reports   │
│ Actions            │ Auto-remove / Demote / Human review queue   │
│ Explainability     │ Yes - reason shown to user on appeal        │
│ Languages          │ Multi-lingual support required              │
└────────────────────┴────────────────────────────────────────────┘
```

**I:** Good framing. Why does the tiered action model matter for your ML design?

**C:** It drives our operating point on the precision-recall curve. Auto-removal requires very high precision - false positives harm legitimate creators and create legal/trust liability. Demotion can tolerate lower precision because the cost of a mistake is lower. This means we want a **multi-threshold** decision boundary, not a single cutoff, and we need well-**calibrated** probabilities to make these thresholds meaningful. Calibration becomes as important as raw discriminative accuracy.

---

## Phase 2: Metrics (7 min)

**I:** How would you evaluate this system - both offline and online?

**C:** I'd structure metrics across three dimensions: model quality, business impact, and integrity health.

### 2.1 Offline Metrics

**1. F1 Score per harm category** - Given the class imbalance (harmful posts are a small fraction of all posts), raw accuracy is useless. F1 captures the precision-recall tradeoff:

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

But more importantly, we care about *where* on the curve we operate. We'd set category-specific operating points based on the cost asymmetry between false positives and false negatives.

**2. PR-AUC (Area Under Precision-Recall Curve)** - The primary ranking metric for imbalanced binary classification. Unlike ROC-AUC, PR-AUC is sensitive to performance on the minority (positive) class and doesn't inflate due to the large number of true negatives.

$$
\text{PR-AUC} = \int_0^1 P(R) \, dR
$$

**3. ROC-AUC** - Secondary metric. Useful for comparing across models at different operating thresholds. Less sensitive to class imbalance than PR-AUC but provides complementary signal.

**4. Calibration (Expected Calibration Error)** - Critical for the tiered decision system. For each predicted probability bucket $B_m$:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

A model predicting P(harmful)=0.9 should be wrong about 10% of the time in that bucket. Poor calibration breaks our confidence-threshold-based action tiers.

**I:** Why PR-AUC over ROC-AUC as the primary metric?

**C:** Consider a platform where 0.1% of posts are harmful. A model that predicts "safe" for everything gets ROC-AUC ~0.5 (random) but more critically, ROC-AUC is dominated by the enormous number of true negatives. PR-AUC specifically measures "of the posts I flagged, how many were actually harmful, and of all harmful posts, how many did I catch?" - which is exactly what the platform cares about. At very low positive rates, PR-AUC differences of 0.02 translate to thousands of missed harmful posts per day.

### 2.2 Online Metrics

| Metric | Definition | Captures |
|--------|-----------|---------|
| **Harmful prevalence** | % of harmful posts seen by users (escaped detection) | Core integrity health |
| **Proactive detection rate** | System-flagged / (System-flagged + User-reported) | Automation coverage |
| **Valid appeal rate** | % of removed posts successfully appealed | False positive rate at scale |
| **Mean time to detection** | Avg time from post upload to removal action | Freshness / latency |
| **Harmful impressions** | Total impressions on harmful posts before removal | User harm exposure |
| **Reviewer efficiency** | Avg decisions per reviewer per hour | Human-in-loop cost |

**I:** How do you connect offline metric improvements to online gains?

**C:** It's rarely a clean mapping, but two approaches help. First, **shadow scoring**: deploy the new model in parallel, log its scores without acting on them, and compare its decisions against eventual ground truth (user reports + reviewer decisions). Second, **A/B testing** content moderation changes is ethically tricky - you don't want to expose some users to more harmful content for an experiment. Instead, we use **backtest** evaluations: apply the new model retroactively to a held-out historical slice and measure what it would have done. For actions that don't affect exposure (e.g., routing to human review), safe A/B tests are possible.

---

## Phase 3: Data Collection & Annotation (5 min)

**I:** Let's talk data. What does your training pipeline look like?

**C:** We have three label sources, each with different quality/quantity tradeoffs:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Label Source Hierarchy                         │
│                                                                  │
│  HIGH QUALITY / LOW VOLUME                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Manual Annotation (~10K/day)                             │    │
│  │ • Contracted reviewers with detailed guidelines          │    │
│  │ • Multi-label: violence, nudity, hate speech, etc.       │    │
│  │ • Inter-annotator agreement tracked (Cohen's Kappa)      │    │
│  │ → Use for: validation set, eval benchmark                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ▼                                      │
│  MEDIUM QUALITY / MEDIUM VOLUME                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Human Review Decisions (~100K/day)                       │    │
│  │ • Reviewer decisions on queued posts                     │    │
│  │ • More consistent than user reports, less than experts   │    │
│  │ → Use for: training set augmentation                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ▼                                      │
│  LOW QUALITY / HIGH VOLUME                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ User Reports (~1M/day)                                   │    │
│  │ • Noisy: users report for many non-harmful reasons       │    │
│  │ • Can be adversarially manipulated                       │    │
│  │ → Use for: weak supervision signal, bootstrapping        │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Annotation challenges worth flagging:**

1. **Subjectivity** - "Hate speech" is more subjective than nudity. We need detailed labeling guidelines with examples, and we measure inter-annotator agreement. Posts with Cohen's Kappa < 0.6 should be re-adjudicated.

2. **Reviewer welfare** - Labelers are exposed to disturbing content at scale. Platforms need psychological support programs, exposure limits, and content warnings. This is both an ethical obligation and a retention issue.

3. **Label leakage through user reports** - If we train directly on user report rates, we encode reporting biases. Certain communities are reported at higher rates for reasons other than harm. We decouple the reporting signal from the harm signal.

4. **Class imbalance** - Harmful posts are typically <0.1% of all posts. We use stratified sampling in training batches and down-sample negatives.

---

## Phase 4: Feature Engineering (10 min)

**I:** Walk me through the features and multimodal representation strategy.

**C:** Harmful content detection is inherently multimodal. Let me break it down:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Taxonomy                            │
├────────────────┬──────────────────┬──────────────┬─────────────┤
│  Post Content  │  Post Metadata   │  Author      │  Context    │
│  (text/image/  │  & Interactions  │  Features    │  Signals    │
│   video)       │                  │              │             │
├────────────────┼──────────────────┼──────────────┼─────────────┤
│ Text embedding │ # likes/shares   │ Account age  │ Time of day │
│ (DistilmBERT)  │ # comments       │ # followers  │ Device type │
│ Image embedding│ # reports        │ Violation    │ Country/geo │
│ (CLIP visual)  │ Engagement rate  │  history     │ App version │
│ Video embedding│ Link destination │ Report rate  │             │
│ (VideoMAE)     │ Post timestamp   │ Profanity    │             │
│ OCR text       │ Cross-posts      │  rate        │             │
│ Audio transcript│                 │ Demographics │             │
└────────────────┴──────────────────┴──────────────┴─────────────┘
```

### 4.1 Text Representation

For text, we need semantic understanding (not just keyword matching - bad actors deliberately use misspellings, slang, and coded language). My choice:

- **DistilmBERT** - multilingual distilled BERT. ~40% smaller and 60% faster than mBERT, with 97% of its performance. Handles 104 languages, which is critical at global scale.
- We fine-tune on domain-specific harmful content data, not just use frozen features. Domain shift from pre-training corpus is significant - BERT was trained on Wikipedia/books, not social media posts.
- **Pooling strategy:** Use the `[CLS]` token embedding for post-level classification. For long posts or comment threads, chunk and mean-pool.

**Special handling for adversarial text:**

```
Original:         "I h@t3 y0u" → normalize → "I hate you"
Leet-speak:       "@$$hole" → normalize → "asshole"
Zero-width chars: "bad​content" (hidden chars) → strip unicode tricks
```

A preprocessing layer handles normalization before the BERT encoder sees the text.

### 4.2 Image Representation

- **CLIP visual encoder** - Pre-trained on 400M image-text pairs. Captures both visual semantics and, importantly, the relationship between text overlaid on images and the image content. Very useful for memes (which combine image + text in harmful ways).
- **SigLIP / ViT-L** - Alternatives with stronger visual classification performance.
- Fine-tune with a content-safety-specific head.

For explicit content (nudity) specifically: pre-trained NSFW classifiers like NudeNet or proprietary equivalents can serve as strong feature inputs or warm-start weights.

### 4.3 Video Representation

Video is the most challenging and expensive modality:

```
┌──────────────────────────────────────────────────────┐
│               Video Feature Extraction                │
│                                                       │
│  Raw Video                                            │
│      │                                                │
│      ├──► Frame Sampling (e.g., 1 fps)                │
│      │         │                                      │
│      │         ▼                                      │
│      │    Image features per frame (CLIP)             │
│      │         │                                      │
│      │         ▼                                      │
│      │    Temporal aggregation (mean/max pool)        │
│      │                                                │
│      ├──► Audio extraction → ASR transcript           │
│      │         │                                      │
│      │         ▼                                      │
│      │    Text features (DistilmBERT)                 │
│      │                                                │
│      └──► VideoMAE / TimeSformer (full temporal)      │
│               (reserved for flagged candidates)       │
└──────────────────────────────────────────────────────┘
```

**Practical tradeoff:** Full temporal video models (VideoMAE, TimeSformer) are expensive. In production, I'd use frame-level features for all videos, and only invoke the full temporal model for posts that exceed a risk threshold on the cheaper features. This is a **two-stage** inference approach.

### 4.4 Author Features

Author features are strong signals for repeated offenders:

| Feature | Description | Why it matters |
|---------|-------------|----------------|
| `violation_count_30d` | # of confirmed violations in last 30 days | Strong recidivism signal |
| `report_rate` | # user reports / # posts | Normalized harassment signal |
| `account_age_days` | Age of account | New accounts have higher risk |
| `follower_count` | Scale of potential harm | High-follower = high impact |
| `profanity_rate` | % of posts containing profane text | Behavioral fingerprint |
| `is_verified` | Verified account flag | Low-risk proxy (noisy) |

**Caution:** Author features can introduce **demographic bias**. Certain communities have historically higher report rates not because they post more harmful content, but because they are reported more (coordinated mass-reporting, etc.). We should monitor demographic disparities in model scores closely.

---

## Phase 5: Model Architecture (12 min)

**I:** Let's go deep on the architecture. How do you fuse multimodal inputs for multi-label classification?

**C:** This is the crux of the design. Let me walk through the fusion strategies and then commit to a recommended architecture.

### 5.1 Fusion Strategies

```
┌──────────────────────────────────────────────────────────────────┐
│              Multimodal Fusion Strategies                         │
│                                                                  │
│  EARLY FUSION                                                    │
│  ┌────────┐  ┌────────┐  ┌────────┐                             │
│  │  Text  │  │ Image  │  │ Video  │                             │
│  └───┬────┘  └───┬────┘  └───┬────┘                             │
│      └───────────┼───────────┘                                   │
│                  │ concat/project                                │
│                  ▼                                               │
│            ┌──────────┐                                          │
│            │ Joint MLP│ → P(harm)                                │
│            └──────────┘                                          │
│  ✓ Learns cross-modal interactions                               │
│  ✗ Modalities must always be present (missing modality problem)  │
│                                                                  │
│  LATE FUSION                                                     │
│  ┌────────┐  ┌────────┐  ┌────────┐                             │
│  │  Text  │  │ Image  │  │ Video  │                             │
│  └───┬────┘  └───┬────┘  └───┬────┘                             │
│      ▼           ▼           ▼                                   │
│   P(harm|T)  P(harm|I)  P(harm|V)                                │
│      └───────────┼───────────┘                                   │
│                  │ ensemble/weighted avg                         │
│                  ▼                                               │
│            P(harm) combined                                      │
│  ✓ Handles missing modalities gracefully                         │
│  ✗ Misses cross-modal signals (e.g., safe image + harmful text)  │
│                                                                  │
│  HYBRID FUSION (Recommended)                                     │
│  Per-modality encoders → Cross-attention fusion → Joint MLP      │
└──────────────────────────────────────────────────────────────────┘
```

**I:** What's the main failure case of late fusion you'd be worried about?

**C:** A classic adversarial case: an image of a peaceful protest (safe if seen alone) paired with hate speech text targeting the group pictured. Late fusion processes image and text independently - both score low - so the combined signal is diluted. The danger is emergent only in the combination. Early or hybrid fusion can learn to flag this pattern through cross-modal attention.

### 5.2 Recommended Architecture: Multi-Modal Multi-Task Transformer

```
                         ┌──────────────────────────────────┐
                         │    Multi-Task Output Heads        │
                         │                                  │
                    ┌────┴───┐ ┌────────┐ ┌──────────┐     │
                    │Violence│ │Nudity  │ │Hate      │ ... │
                    │Head    │ │Head    │ │Speech    │     │
                    │(MLP)   │ │(MLP)   │ │Head(MLP) │     │
                    └────┬───┘ └────┬───┘ └────┬─────┘     │
                         └─────────┴────────────┘           │
                                   │                        │
                         ┌─────────┴──────────┐             │
                         │  Shared Fusion MLP  │             │
                         │  [1024→512→256]     │             │
                         └─────────┬──────────┘             │
                                   │                        │
          ┌─────────────────────────┼────────────────────┐   │
          │                         │                    │   │
   ┌──────┴──────┐          ┌───────┴──────┐    ┌───────┴──┐│
   │Cross-Modal  │          │Cross-Modal   │    │Structured ││
   │Attention    │          │Attention     │    │Features   ││
   │(Text↔Image) │          │(Text↔Video)  │    │MLP        ││
   └──────┬──────┘          └───────┬──────┘    └───────┬──┘│
          │                         │                    │   │
   ┌──────┴──────┐          ┌───────┴──────┐    ┌───────┴──┐│
   │  Text       │          │  Video       │    │ Author + ││
   │  Encoder    │          │  Encoder     │    │ Context  ││
   │(DistilmBERT)│          │ (frame-level │    │ Features ││
   │             │          │  CLIP +      │    │          ││
   │  [CLS] tok  │          │  temporal    │    │          ││
   │  → 768d     │          │  pool) → 512d│    │ → 64d    ││
   └──────┬──────┘          └───────┬──────┘    └───────┬──┘│
          │                         │                    │   │
   ┌──────┴──────┐          ┌───────┴──────┐             │   │
   │  Text Input  │         │  Image Input │             │   │
   │  + Comments  │         │  + Video     │             │   │
   └─────────────┘          └─────────────┘             │   │
                                                         │   │
                                        Structured data ─┘   │
                                                             │
                                                             │
                                        ┌────────────────────┘
                                        │  Separate CLIP image
                                        │  encoder for images-only
                                        │  posts (no video)
                                        └──────────────────────
```

### 5.3 Multi-Task Learning Design

We use a **single shared backbone** (the multimodal encoder + fusion MLP) with **task-specific classification heads** per harm category:

**Why multi-task?**

1. **Data efficiency** - With 10K labels/day and 3+ tasks, each task gets only a few thousand samples. Shared layers allow knowledge transfer: patterns predictive of violence often overlap with hate speech (aggressive language, weapons).

2. **Regularization** - The shared backbone can't overfit to any single task. Shared representations generalize better.

3. **Single model, single inference pass** - One model produces all harm probabilities, which is operationally simpler and faster than running 3-4 separate models.

**Loss function:**

$$
\mathcal{L}_{total} = \sum_{k \in \text{tasks}} \lambda_k \cdot \mathcal{L}_{BCE}^{(k)}
$$

where each task $k$ uses binary cross-entropy:

$$
\mathcal{L}_{BCE}^{(k)} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i^{(k)} \log p_i^{(k)} + (1 - y_i^{(k)}) \log(1 - p_i^{(k)}) \right]
$$

The task weights $\lambda_k$ are tuned based on task importance and data availability. Rarer, higher-stakes categories (CSAM, terrorism) get higher $\lambda$.

**I:** What happens when modalities are missing? A text-only post has no image.

**C:** Great edge case. We handle this with **modality dropout and masking**:

1. During training, randomly zero out modality embeddings with probability $p_{mask}=0.3$ - this teaches the model to function with any subset of modalities.
2. At serving, missing modalities are replaced with **learned null embeddings** (a trainable vector representing "no image present").
3. The cross-attention layers naturally attend less to null embeddings since they contain no meaningful query/key signal.

This is preferable to training separate models per modality subset, which would be operationally unmanageable.

### 5.4 Handling Multimodal Training Imbalance

**I:** What's the gradient blending problem you mentioned in the spec?

**C:** In multimodal training, one modality (typically the most discriminative) can dominate gradient updates, causing other modalities to underfit. For example, explicit images have strong pixel-level signals - the vision encoder converges early and starts dominating the joint loss, while the text encoder stops learning.

Two mitigations:

**1. Gradient blending** (Wang et al., OGM-GE): Scale each modality's gradient by the ratio of its overfitting rate to the average, effectively slowing down the faster-learning modality:

$$
\tilde{g}_m = g_m \cdot \frac{\bar{\rho}}{\rho_m}
$$

where $\rho_m$ is modality $m$'s train/val performance ratio (overfitting indicator).

**2. Focal loss** - Down-weights easy examples (strong image signal already classifies correctly) so the model is forced to learn from harder multimodal cases:

$$
\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

$\gamma=2$ is standard. This naturally balances learning across modalities because easy visual positives get down-weighted, forcing the model to rely more on subtle text+image combinations.

---

## Phase 6: Training Pipeline (7 min)

**I:** Walk me through the training infrastructure and data pipeline.

**C:** Here's the end-to-end training pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Data Pipeline                       │
│                                                                  │
│  Raw Posts (Kafka stream)                                        │
│       │                                                          │
│       ▼                                                          │
│  Content Ingestion & Preprocessing                               │
│  (decode images/video, normalize text, ASR for audio)            │
│       │                                                          │
│       ▼                                                          │
│  Label Joining                                                   │
│  ┌─────────────────────────────────────┐                        │
│  │  Manual labels (10K/day)            │                        │
│  │  + Reviewer decisions (100K/day)    │  → Training labels     │
│  │  + Weak labels (user reports)       │                        │
│  └─────────────────────────────────────┘                        │
│       │                                                          │
│       ▼                                                          │
│  Stratified Sampling                                             │
│  (oversample harmful, balance across harm categories)            │
│       │                                                          │
│       ▼                                                          │
│  Feature Store                                                   │
│  (precomputed author features, cached modality embeddings)       │
│       │                                                          │
│       ├──► Offline Batch Training (weekly, full fine-tune)       │
│       │    - Full multimodal model on GPU cluster (A100s)        │
│       │    - Distributed: data parallelism for MLP,             │
│       │      model parallelism for large encoders                │
│       │                                                          │
│       └──► Online Incremental Fine-tuning (daily)               │
│            - Update task-specific heads only                     │
│            - Uses last 24h of reviewer decisions                 │
│            - Keeps backbone frozen to preserve generalization     │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1 Label Noise Management

We're training on a mix of label qualities. We handle this explicitly:

| Label source | Quality | Approach |
|---|---|---|
| Expert annotations | High (Kappa > 0.8) | Full weight in loss |
| Reviewer decisions | Medium (Kappa ~0.7) | Reduced weight (0.7x) |
| User reports | Low | Weak supervision only; use as soft labels with 0.3x weight |
| Model pseudo-labels | Variable | Curriculum: add only high-confidence pseudo-labels (P > 0.95) |

**Label smoothing** is applied to prevent overconfidence on noisy labels:

$$
y_{\text{smooth}} = y \cdot (1 - \epsilon) + \frac{\epsilon}{K}
$$

with $\epsilon = 0.1$ and $K$ classes.

### 6.2 Class Imbalance

Harmful posts are ~0.1–1% of all posts. Three strategies in combination:

1. **Stratified batch sampling** - Each training batch is constructed with a target positive ratio (e.g., 30% harmful), regardless of the true base rate.
2. **Negative downsampling** at data loading time, with calibration correction at inference.
3. **Asymmetric loss function** - Up-weight false negatives (missed harmful content) vs. false positives, because the cost asymmetry in the real system favors recall.

### 6.3 Avoiding Representation Bias

**I:** How do you handle bias? The model might learn to flag content from certain communities disproportionately.

**C:** This is a critical safety concern. Several approaches:

1. **Demographic parity audits** - Regularly measure false positive rates across demographic slices (language, country, topic). A model that flags Spanish-language content at 2x the rate of English with equal actual harm rates is biased.

2. **Counterfactual data augmentation** - For text, generate counterfactual versions: replace identity terms ("Black people" → "White people") and retrain. Model scores should be invariant to identity substitutions when the surrounding content is equivalent.

3. **Debiased label collection** - Annotator pool should reflect diversity. Track per-annotator bias metrics and weight annotators accordingly.

4. **Regularization toward fairness** - Add a fairness constraint to the loss:

$$
\mathcal{L}_{total} = \mathcal{L}_{BCE} + \mu \cdot \mathcal{L}_{fairness}
$$

where $\mathcal{L}_{fairness}$ penalizes demographic disparities in predicted scores.

---

## Phase 7: Prediction Service & Decision Pipeline (7 min)

**I:** Walk me through the serving architecture. How does a post go from upload to action?

**C:** Here's the end-to-end flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Post Lifecycle (Harm Detection)                │
│                                                                  │
│  User Uploads Post                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────┐                                            │
│  │ Ingestion Service│ → stores raw content, queues for scoring   │
│  └──────────┬───────┘                                            │
│             │  async (< 2s)                                       │
│             ▼                                                    │
│  ┌──────────────────────────────────────────────┐                │
│  │         Classifier Orchestrator              │                │
│  │                                              │                │
│  │  Stage 1: Cheap Heuristic Filters            │                │
│  │  (hash-match against known bad content,      │                │
│  │   keyword blocklists, account risk score)    │                │
│  │     → if match: fast-path to Violation Svc  │                │
│  │                                              │                │
│  │  Stage 2: Full Multimodal Inference          │                │
│  │  (DistilmBERT + CLIP + Video encoder)        │                │
│  │     → P(violence), P(nudity), P(hate)        │                │
│  │                                              │                │
│  │  Stage 3: Decision Thresholds                │                │
│  │  P(harm) > τ_high  → Violation Service       │                │
│  │  P(harm) > τ_mid   → Demotion Service        │                │
│  │  P(harm) > τ_low   → Human Review Queue      │                │
│  │  P(harm) < τ_low   → Publish post            │                │
│  └──────────────────────────────────────────────┘                │
│             │                                                    │
│    ┌────────┼────────┬────────────────┐                          │
│    ▼        ▼        ▼                ▼                          │
│  Remove  Demote  Queue for        Publish                        │
│  + notify  in   human review     (allow)                         │
│  user    feed   (with model                                      │
│                 explanation)                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 7.1 Two-Stage Inference for Latency

Full multimodal inference is expensive. We use a **cascaded approach**:

```
Stage 1: Cheap filters (< 10ms)
  - Perceptual hash match (known CSAM/violence hash databases)
  - Keyword blocklist scan
  - Account risk score lookup
  → ~20-30% of harmful content caught here at near-zero cost

Stage 2: Light multimodal model (< 100ms)
  - Text only (DistilmBERT) + image thumbnail (CLIP, 224px)
  - Catches majority of remaining cases

Stage 3: Full model (< 500ms)  [async, for borderline cases]
  - Full resolution image + video temporal model + all features
  - Only invoked for posts scoring 0.2 < P(harm) < 0.8 in Stage 2
```

This means full inference runs on perhaps 10-20% of posts, dramatically reducing compute cost.

### 7.2 Model Serving Infrastructure

| Component | Technology | Purpose |
|---|---|---|
| Model serving | TorchServe / Triton | GPU-accelerated inference |
| Feature store | Redis (hot) + Cassandra (warm) | Author features, cached embeddings |
| Message queue | Kafka | Async post scoring pipeline |
| Orchestration | Kubernetes | Auto-scaling inference workers |
| Hash database | In-memory bloom filter | Near-instant known-bad content lookup |

### 7.3 Explainability

**I:** You mentioned showing users a reason when content is removed. How?

**C:** Explainability is required both for user trust and regulatory compliance (e.g., EU Digital Services Act). We achieve it through:

1. **Category labels from multi-task heads** - Each head predicts a specific harm category. If P(violence)=0.95, we report "Removed: graphic violence."

2. **Attention-based saliency** - For text, highlight which tokens contributed most to the prediction (using attention rollout or integrated gradients). For images, generate a GradCAM heatmap showing which region triggered the flag.

3. **Human-readable templates** - The harm category + saliency map is translated into a user-facing message: *"This post was removed because it contained imagery that violates our violence policy [link to policy]."*

4. **Reviewer context** - If the post goes to human review, the reviewer sees the model's top predicted category and confidence, plus highlighted regions. This focuses reviewer attention and improves efficiency.

---

## Phase 8: Monitoring & Failure Modes (5 min)

**I:** What can go wrong in production?

**C:** Harmful content detection has some unique failure modes beyond standard ML monitoring:

### 8.1 Real-time Monitoring

```
┌──────────────────────────────────────────────────────────────┐
│                  Monitoring Dashboard                         │
│                                                              │
│  INTEGRITY HEALTH                 MODEL HEALTH               │
│  ┌────────────────────┐          ┌─────────────────────┐    │
│  │ Harmful prevalence │          │ Score distribution  │    │
│  │ █████████░ 92%     │          │ shift alert         │    │
│  │ (target: >95% det) │          │                     │    │
│  └────────────────────┘          │ P99 latency: 320ms  │    │
│                                  │ (alert if >500ms)   │    │
│  ┌────────────────────┐          └─────────────────────┘    │
│  │ Appeal rate        │                                     │
│  │ 1.2% of removals   │          DATA PIPELINE HEALTH       │
│  │ (target: <2%)      │          ┌─────────────────────┐    │
│  └────────────────────┘          │ Feature freshness   │    │
│                                  │ Author features: ✓  │    │
│  ┌────────────────────┐          │ Text encoder: ✓     │    │
│  │ Proactive rate     │          │ Video ingestion: ⚠  │    │
│  │ 87% system-found   │          └─────────────────────┘    │
│  │ 13% user-reported  │                                     │
│  └────────────────────┘                                     │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 Known Failure Modes

**1. Adversarial evasion**
Bad actors probe the system and learn to evade: mirroring images, adding text watermarks, splitting harmful content across multi-post threads.

*Mitigation:* Robustness training with adversarial augmentation; hash-resistant perceptual hashing (PDQ, PhotoDNA); thread-level context modeling.

**2. Concept drift - new harm categories**
A new type of harmful content emerges (e.g., deepfake porn, new coded language). The model has no training data.

*Mitigation:* Rapid annotation pipelines (24-hour emergency labeling); continuous monitoring of human reviewer override rates (high override rate = model blind spot); active learning to prioritize labeling of uncertain cases.

**3. Feedback loop - over-policing**
If the model over-removes content from certain communities, those users disengage or leave. This causes a distribution shift: the remaining posts from that community are now "cleaner," making the model appear well-calibrated when in fact it has driven out users. This is a classic **survivorship bias** loop.

*Mitigation:* Track demographic-stratified engagement metrics; compare flag rates vs. harm rates across communities; independent audits.

**4. Model staleness during events**
During major news events (e.g., war, political crisis), the volume of violent/disturbing content spikes. Models trained on normal-period data may have uncalibrated scores.

*Mitigation:* Event-triggered rapid retraining; increase human review queue capacity for high-volume events; temporary threshold adjustment with human-in-the-loop override.

**5. Cross-modal adversarial attacks**
Content that appears benign in each modality individually but is harmful in combination (e.g., innocent image + hate speech in comments).

*Mitigation:* Comment-text is included as a feature alongside the post; cross-modal attention mechanisms explicitly model these interactions.

### 8.3 Monitoring Stack

| Signal | Metric | Alert threshold |
|---|---|---|
| Score distribution | KL divergence from baseline | > 0.1 daily |
| Feature coverage | % non-null per feature | Drop > 5% |
| False positive rate | Valid appeal rate | > 3% weekly |
| Latency | P99 inference latency | > 500ms |
| Demographic disparity | FPR ratio across groups | > 1.5x |
| Label quality | Inter-annotator Kappa | < 0.6 |

---

## Phase 9: Scaling & Cold Start (4 min)

**I:** Two more areas - how do you scale this, and how do you handle new harm categories?

**C:**

### 9.1 Scaling to 500M Posts/Day

The key insight is that **not all posts need full inference**. Our two-stage cascade already handles ~80% of posts cheaply. For scaling the remaining 20%:

- **Horizontal scaling** of GPU inference workers - Kubernetes auto-scales based on queue depth. Each worker handles a batch of posts.
- **Batched inference** - Group posts into batches of 32-64 for GPU utilization. Text posts batch easily; video requires more careful sizing.
- **Embedding caching** - For repeat offenders or viral content, cache the encoded embeddings. If the same video hash appears 10K times in an hour, compute once.
- **Async processing** - The pipeline is fully asynchronous. A post is published immediately (unless caught by Stage 1 heuristics), then scored in the background, then actioned. This decouples serving latency from inference latency.

### 9.2 New Harm Category: Cold Start

**I:** A new type of harmful content is identified. How do you add a new detection category in 48 hours?

**C:** This is a real operational scenario. The key insight: **we don't need to retrain the backbone**.

```
Week 0: Zero-shot
  → Query existing model's shared representation space
  → Use CLIP-style zero-shot: encode text description of new harm
    category ("images of [new harm]") and compute cosine similarity
    to image embeddings. Works surprisingly well as a stopgap.

Day 1-2: Few-shot with lightweight head
  → Emergency annotation sprint: 200-500 examples of new category
  → Freeze backbone, train only a new classification head
  → Deploy new head in <24h
  → Backbone already learned general representations

Week 2+: Full fine-tune
  → Accumulate 10K+ labeled examples
  → Full multi-task retraining including new category
  → Replace emergency head with production-quality model
```

This tiered approach means we go from zero to some detection capability in hours, while the production-quality model is trained properly over weeks.

---

## Summary & Mental Model

**I:** Great walkthrough. Summarize the key design tradeoffs.

**C:** Here's how I think about the core tensions in harmful content detection:

```
┌──────────────────────────────────────────────────────────────────┐
│           Key Tradeoffs in Harmful Content Detection              │
│                                                                  │
│  Recall ◄──────────────────────────────────► Precision           │
│  (catch more harm)                          (fewer false removals)│
│                                                                  │
│  Model Complexity ◄───────────────────────► Serving Latency      │
│  (better MM fusion)                         (deeper = slower)    │
│                                                                  │
│  Single Model ◄───────────────────────────► Per-Category Model   │
│  (efficiency, shared repr.)                 (specialized accuracy)│
│                                                                  │
│  Human Reviewer Volume ◄──────────────────► Model Confidence     │
│  (safety net)                               (automation rate)    │
│                                                                  │
│  Fast Action ◄────────────────────────────► Due Process          │
│  (minimize harm spread)                     (avoid false removals)│
│                                                                  │
│  Generalization ◄─────────────────────────► Adversarial Robustness│
│  (perform across content)                   (handle evasion)     │
└──────────────────────────────────────────────────────────────────┘
```

**The 80/20 of harmful content detection:**

> **Multimodal fusion > model architecture > feature engineering > threshold tuning.**
> The hardest problem isn't the ML - it's the **data quality, annotation consistency, demographic fairness, and operational pipeline** that determines whether the system actually reduces harm in the real world.

### Quick Reference: Decision Map

| Design Choice | Recommendation | Key Reason |
|---|---|---|
| Fusion strategy | Hybrid (cross-attention) | Catches cross-modal harm |
| Text encoder | DistilmBERT (multilingual) | Speed + multilingual + context |
| Image encoder | CLIP ViT-L | Image-text alignment |
| Video strategy | Frame-sampling → full model cascade | Cost vs. coverage tradeoff |
| Classification | Multi-task shared backbone | Data efficiency, single inference |
| Class imbalance | Stratified sampling + label smoothing | Stable training |
| Inference pipeline | 3-stage cascade | 80% posts handled cheap |
| Monitoring | Demographic-stratified FPR | Fairness + quality |
| Cold start | Freeze backbone, train new head | 24h time to new category |

---

*Key papers: CLIP (Radford et al., 2021), DistilBERT (Sanh et al., 2019), VideoMAE (Tong et al., 2022), Gradient Blending (Wang et al., 2020), Focal Loss (Lin et al., 2017), DCN v2 (Wang et al., 2021).*
