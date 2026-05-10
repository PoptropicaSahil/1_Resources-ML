> READ the pdf -- nice simple only. But this thing from Claude is really nice too!


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


### 6.3 Avoiding Representation Bias

**I:** How do you handle bias? The model might learn to flag content from certain communities disproportionately.

**C:** This is a critical safety concern. Several approaches:

1. **Demographic parity audits** - Regularly measure false positive rates across demographic slices (language, country, topic). A model that flags Spanish-language content at 2x the rate of English with equal actual harm rates is biased.

2. **Counterfactual data augmentation** - For text, generate counterfactual versions: replace identity terms ("Black people" → "White people") and retrain. Model scores should be invariant to identity substitutions when the surrounding content is equivalent.

3. **Debiased label collection** - Annotator pool should reflect diversity. Track per-annotator bias metrics and weight annotators accordingly.

4. **Regularization toward fairness** - Add a fairness constraint to the loss:


### 7.3 Explainability

**I:** You mentioned showing users a reason when content is removed. How?

**C:** Explainability is required both for user trust and regulatory compliance (e.g., EU Digital Services Act). We achieve it through:

1. **Category labels from multi-task heads** - Each head predicts a specific harm category. If P(violence)=0.95, we report "Removed: graphic violence."


### 9.1 Scaling to 500M Posts/Day

`The key insight is that **not all posts need full inference**. Our two-stage cascade already handles ~80% of posts cheaply. For scaling the remaining 20%`:

- **Horizontal scaling** of GPU inference workers - Kubernetes auto-scales based on queue depth. Each worker handles a batch of posts.
- **Batched inference** - Group posts into batches of 32-64 for GPU utilization. Text posts batch easily; video requires more careful sizing.
- **Embedding caching** - For repeat offenders or viral content, cache the encoded embeddings. If the same video hash appears 10K times in an hour, compute once.
- **Async processing** - The pipeline is fully asynchronous. A post is published immediately (unless caught by Stage 1 heuristics), then scored in the background, then actioned. This decouples serving latency from inference latency.

### 9.2 New Harm Category: Cold Start

**I:** A new type of harmful content is identified. How do you add a new detection category in 48 hours?

**C:** This is a real operational scenario. `The key insight: **we don't need to retrain the backbone**.`

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
