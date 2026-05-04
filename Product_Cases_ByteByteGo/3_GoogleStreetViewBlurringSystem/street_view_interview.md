# ML System Design Interview: Google Street View Blurring System

**Role:** Senior Data Scientist  
**Duration:** 1.5 - 2 hours  
**Difficulty:** Hard  

---

## Phase 1: Requirements Gathering & Clarifications

### Interviewer:
"Let's start with the problem statement. We want to build a system that automatically detects and blurs faces and license plates in Google Street View images. Before we dive into the solution, let me understand your initial thoughts and ask some clarifying questions."

### Candidate:
"Sure! Based on the problem statement, I'm thinking we need a computer vision system that can detect sensitive information in images. But I have several clarifying questions before I propose a solution."

### Interviewer:
"Great! Go ahead."

### Candidate's Clarifying Questions:

**1. Scale & Scope:**
- How many images do we process daily/monthly?
- What's the geographic scope? (Global, specific regions)
- What image resolutions are we dealing with?
- Are we processing images in real-time or batch mode?

**2. Accuracy Requirements:**
- What's the tolerance for false negatives (unblurred faces)? Is 100% recall required, or is 90-95% acceptable?
- What's the tolerance for false positives (incorrectly blurred regions)?
- Do we care about false positives on artwork or billboards with faces?

**3. Privacy & Regulatory:**
- Are there regulatory requirements (GDPR, privacy laws)?
- What blur intensity is considered sufficient to prevent re-identification?
- Should we handle different object types (faces, license plates only, or other sensitive info)?

**4. Geographic Variation:**
- License plates vary by country/region - do we need different models?
- Do face styles, ethnicities, and appearances matter for model training?
- Any special cases like sunglasses, hats, or partial occlusions?

**5. Latency & Cost:**
- What's our latency budget per image?
- Do we have GPU availability, or must we optimize for CPU?
- What's our cost constraint?

---

## Phase 2: Interviewer Provides Context

### Interviewer:
"Good questions! Let me set the context:
- We have approximately **10 million images per day** from various countries
- Images are **5-megapixel, 2448×2048 resolution**
- We process in **batch mode** (not real-time, so latency is flexible)
- **Privacy requirement:** >89% recall for faces, >94% recall for license plates
- **False positive rate:** <1-2% pixel FPR (false positive rate) to maintain image quality
- **Scope:** Faces and license plates only
- **Geographic:** US and EU license plates (different aspect ratios)
- **Model deployment:** We'll have GPU clusters available
- **Timeline:** System must process 10M images daily"

### Interviewer:
"Now, knowing this context, what would be your high-level approach?"

---

## Phase 3: High-Level System Design

### Candidate:
"Given this context, here's my high-level approach:

**System Overview:**
1. **Detection Pipeline:** Use deep learning models to detect faces and license plates
2. **Post-Processing:** Implement a filtering mechanism to reduce false positives
3. **Blurring:** Apply blur to detected regions
4. **Batch Processing:** Process 10M images/day using a distributed system

**Key Design Choices:**
- **Detection Method:** Object detection (YOLO, Faster R-CNN, or two-stage detector)
- **Recall vs. Precision Trade-off:** Prioritize **recall over precision** since privacy is critical (missing a face is worse than blurring an artwork)
- **Post-Processing:** Use a secondary model or rule-based filtering to reduce false positives
- **Scalability:** Distribute processing across multiple GPUs using batch inference

**Why this approach?**
- YOLO-family detectors are fast (critical for 10M images/day)
- Two-stage approach (high-recall detector + post-processor NN) allows us to catch most faces first, then filter false positives
- Batch processing maximizes GPU utilization and throughput
- Separates recall optimization from precision optimization"

### Interviewer:
"Good! Now let's dig deeper. Why would you prioritize recall over precision?"

### Candidate:
"Excellent question. The asymmetric cost of errors:
- **False Negative (missed face):** Privacy violation → bad for users, regulatory risk, reputation damage
- **False Positive (blurred artwork):** Minor visual degradation → acceptable trade-off

If we think of this as a **cost function:**
- Cost(FN) >> Cost(FP)

This is different from typical ML applications where we might care equally about both metrics.

**Quantitatively:**
- With 10M images and even 5% miss rate, that's 500K unblurred faces
- Even 2% false positive rate means only ~100-200K incorrectly blurred pixels per image (out of 5M pixels)

The false positive cost is acceptable."

### Interviewer:
"Alright, I like this thinking. Now, what detection algorithm would you choose and why?"

### Candidate:
"This is interesting. There are multiple options:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **YOLO (Single-stage)** | Fast inference (7-10 ms per 5MP), Real-time capable | Lower precision at first pass, needs post-processing | ✓ Good for recall |
| **Faster R-CNN (Two-stage)** | High precision individually | Slower (30-50 ms), not ideal for 10M images/day | ✗ Too slow |
| **Sliding Window + Post-processor** | Can tune to very high recall, domain-specific features | Slower, custom implementation | ✓ Best for privacy |

**My recommendation: Hybrid Sliding Window + Neural Network Post-processor**

**Why?**
1. **Primary Detector (Sliding Window):** Fast integral image-based face detector, tuned for **high recall, low precision** (~78% recall baseline, can tune to >89%)
2. **Secondary Detector:** High-precision detector as a safety check
3. **Post-Processor NN:** Use domain-specific features to filter false positives without losing true positives
4. **License Plates:** Adapted sliding window detector with simplified post-processing

This is what Google does in their published research, and it's proven to work at scale."

### Interviewer:
"Fair point. Let's talk about the post-processor neural network. What features would you include?"

---

## Phase 4: Deep Dive - Post-Processor Design

### Candidate:
"Great question! The post-processor NN is critical for maintaining quality. Here are the key features:

### Feature Categories:

**1. Geometric Features (Location-based):**
- Height-to-bottom ratio: If a box is at an unusual depth (e.g., very large or very small), it might be a false positive (like a billboard)
- Camera index: Different cameras in the Street View rig might have different characteristics
- Box aspect ratio: Unusual ratios can indicate false positives
- 3D reasoning: Use camera height and image projection to estimate real-world object size

**Mathematical basis:**
If we assume constant camera height and ground plane, the ratio:
\(r = \frac{\text{box\_bottom}}{\text{box\_bottom} - \text{box\_top}}\)

This relates to real-world object height. Extreme ratios suggest false positives.

**2. Color Features:**
- Average face color probability: Faces have characteristic skin tones
  - Build color histogram from confirmed faces in HSV space
  - Compute p(face | color) for each pixel in candidate box
  - Use average probability as a feature
  - This helps distinguish faces from other colored objects (paintings, posters)

**3. Detector Confidence:**
- Primary detector score
- Secondary detector score (if available)
- Landmark detection score

**4. Deep Features (CNN-based):**
- Pass cropped box (64×64 grayscale) through a CNN
- Use CNN output as a feature to the post-processor
- Captures texture and patterns humans associate with faces

**5. Context Features (for license plates):**
- Car detector overlap: Is there a car near the license plate?
- Context helps disambiguate: plates usually appear on cars

### Post-Processor Architecture:
\[
\text{NN Input} = [\text{box\_coords}, \text{detector\_scores}, \text{color\_prob}, \text{cnn\_features}, \text{geometric\_feats}]
\]

\[
\text{Output} = \text{Sigmoid}(\text{FC layers}) \in [0, 1]
\]

**Training objective:**
Minimize modified cross-entropy that weights larger boxes more heavily (since missing a larger face is worse):

\[
L = \sum_{i=1}^{N} (\text{box\_area}_i \cdot \text{CrossEntropy}(\hat{y}_i, y_i))
\]

This ensures we don't just rely on box recall metrics but care about pixel-level coverage."

### Interviewer:
"Interesting! You mentioned tuning for high recall. How do you actually quantify recall in this problem? There's a subtlety here."

### Candidate:
"Excellent catch! There are two ways to measure recall:

**1. Box Recall (what most models report):**
\[
\text{Box Recall} = \frac{\text{# detected boxes overlapping ground truth}}{\text{# ground truth boxes}}
\]

With IoU threshold: box counts if IoU > threshold (typically 0.5)

**2. Pixel Recall (what matters for privacy):**
\[
\text{Pixel Recall} = \frac{\text{# pixels blurred in faces}}{\text{# pixels in ground truth faces}}
\]

**Why the difference?**
- A detection box that overlaps 10% with ground truth counts as 0 in Box Recall (if IoU threshold is 0.5)
- But those 10% of pixels are still blurred, so pixel-level recall is non-zero
- For privacy, pixel recall is what matters

**For our system:**
- We measure **hand-counted pixel recall**: Humans evaluate if faces are "sufficiently blurred" (features obscured)
- This is subjective but aligns with the business requirement
- Reported performance: **89% hand-counted pixel recall for faces, 94-96% for license plates**

This is why Google reports 89% rather than using standard box-based metrics."

### Interviewer:
"Good insight. Now let's talk about scalability. You said we need to process 10M images per day. How would you architect this?"

---

## Phase 5: Scalability & Infrastructure

### Candidate:
"Given 10M images/day with current inference speed of ~7-10 seconds per 5MP image, here's the math:

**Back-of-envelope calculation:**
\[
\text{Total compute needed} = 10^7 \text{ images} \times 9 \text{ sec} = 9 \times 10^7 \text{ seconds} = 1,041 \text{ days of compute}
\]

If we run 24/7 on a single GPU: 1,041 days = **2.8 years** - way too slow!

**Solution: Distributed Batch Processing**

**Batch inference optimization:**
- Instead of processing images one-by-one, batch them
- Batch size 32-64 significantly improves GPU utilization
- Estimated throughput: 50-100 images/sec per GPU with batching

\[
\text{Throughput} = \frac{100 \text{ images/sec}}{1 \text{ GPU}} = \frac{10^7 \text{ images}}{10^5 \text{ images/day}} = 100 \text{ GPUs needed}
\]

Actually, let me recalculate with realistic numbers:
- Batch inference: ~50 ms per batch of 64 images = 0.78 ms per image
- In 24 hours: 86,400 seconds × 1,282 images/sec = **110M images processed**
- For 10M images/day: need approximately **15-20 GPUs**

**System Architecture:**

```
Raw Images (S3/GCS)
    ↓
Message Queue (Kafka/RabbitMQ) with 10M messages
    ↓
[Preprocessing Workers] (CPU)
    ↓
Batching Layer (group into batches of 64)
    ↓
[GPU Worker Pool: 15-20 GPUs]
    ├─ Face Detection
    ├─ License Plate Detection
    ├─ Post-processing
    └─ Blurring
    ↓
Output Storage (S3/GCS)
```

**Key design decisions:**

1. **Separation of Preprocessing & GPU Work:**
   - Preprocessing (resize, normalize) is CPU-bound
   - Detection/blurring is GPU-bound
   - Separate services allow independent scaling

2. **Batching Strategy:**
   - Use dynamic batching: if batch not full after T milliseconds, process partial batch
   - Prevents excessive latency in low-traffic scenarios

3. **Error Handling:**
   - Failed images → retry queue
   - Timeout → fallback to single-image processing
   - Logging of confidence scores for monitoring

4. **Cost Optimization:**
   - Use spot instances for non-critical retries
   - Monitor GPU utilization (aim for >80%)
   - Consider cheaper models (quantized YOLOv5 nano) if accuracy allows

**Model Serving:**
- Use TensorFlow Serving or Triton Inference Server for model management
- Enables model versioning and A/B testing

**Monitoring:**
- Track latency (P50, P95, P99)
- Track recall/precision on holdout validation set
- Monitor false positive rate pixel-by-pixel"

### Interviewer:
"Good! But I want to dig into model choice more. Why not just use YOLOv8?"

### Candidate:
"Great question. YOLOv8 is modern, but there's a reason Google (when they published their work in 2008-2009) used sliding window detectors:

**Sliding Window Advantages for This Domain:**
1. **Tunable Recall-Precision Trade-off:** Can explicitly set detection threshold to maximize recall
2. **Interpretable:** Each detection has clear score indicating confidence
3. **Fast Pipeline:** Integral image features enable quick scanning
4. **Domain-specific Optimization:** Can add specialized features (color, geometric cues)

**YOLO Limitations:**
1. **Anchor-based (v5) / Anchor-free (v8):** Still trained with specific class distribution assumptions
2. **Post-processing Complexity:** Removing false positives requires external pipeline (NMS, filtering)
3. **Not optimized for extreme recall:** Designed for balanced precision-recall

**However, if I had to use YOLO:**

**YOLOv8 Comparison:**
- Inference speed: **~0.78 ms per image** (better than 9 seconds!)
- Accuracy: **mAP ~44.9 on COCO** for small model
- Advantage: Modern, well-maintained, easier to deploy

**Why I'd still choose sliding window for THIS problem:**
- Published results show **89% recall at acceptable precision** is achievable
- YOLO would need aggressive post-processing to reach this
- Sliding window specifically designed for face detection (decades of research)

**Hybrid approach (best of both):**
- Use YOLOv8 nano for initial fast pass (catches obvious cases)
- Use sliding window for detailed pass (catches edge cases)
- Post-processor NN filters false positives from both

This is similar to Google's strategy: multiple detectors at different operating points."

### Interviewer:
"Let me push back. Modern YOLO models are very good. What specific benchmark data do you have?"

### Candidate:
"Fair point. Let me look at recent benchmarks:

**YOLOv8 vs. YOLOv5 Performance:**
| Model | mAP@50-95 | Inference (CPU) | Inference (GPU) | Recall |
|-------|-----------|-----------------|-----------------|--------|
| YOLOv5n | 28.0 | 73.6 ms | 1.12 ms | ~73% (typical) |
| YOLOv5s | 37.4 | 120.7 ms | 1.92 ms | ~80% |
| YOLOv8n | 37.3 | 80.4 ms | 1.47 ms | ~76% |
| YOLOv8s | 44.9 | 128.4 ms | 2.66 ms | ~82% |

**Key insight:** Standard mAP doesn't tell us about recall at the extreme operating point we need.

For face detection specifically, older research (Google 2009) achieved:
- **89% recall** with sliding window + post-processor
- This was with much older hardware!

**Modern approach I'd propose:**
1. Fine-tune YOLOv8 nano on Street View face dataset (transfer learning)
2. Set detection threshold to maximum recall (even if it means many false positives initially)
3. Use post-processor NN to filter
4. Benchmark against hand-labeled test set

YOLOv8 might actually achieve similar or better results if trained on Street View data, due to:
- Better backbone architecture (CSPDarknet vs. older features)
- Anchor-free design (better for diverse face scales)
- Batch normalization and modern training techniques

**So revised recommendation:**
- **Start with YOLOv8 + transfer learning**
- **If recall doesn't meet 89% threshold, add sliding window as ensemble**
- **Use the two-stage post-processor approach regardless**"

### Interviewer:
"I like this pragmatic thinking. Now, let's talk about training data and evaluation."

---

## Phase 6: Data & Evaluation Strategy

### Candidate:
"Excellent. This is crucial.

### Training Data Requirements:

**Dataset composition:**
- 1 million annotated Street View images with:
  - Bounding boxes for all identifiable faces (face size > 12 pixels)
  - Bounding boxes for all license plates
  - Geographic diversity (US, EU, other regions)
  - Lighting conditions (day, night, shadows)
  - Challenging cases (sunglasses, hats, partial occlusions, rear-view mirrors)

**Data split:**
- Training: 700K images
- Validation: 150K images
- Test: 150K images (held out, never seen during training/tuning)

**Data quality considerations:**
- Avoid labeling billboards, artwork with faces (source of false positives)
- Only label faces/plates deemed 'identifiable' (subjective but consistent)
- Use multiple annotators with inter-annotator agreement checks

### Evaluation Metrics:

**Primary Metrics:**

1. **Hand-Counted Pixel Recall:**
   - Annotators manually review blurred outputs
   - Judge if faces are 'sufficiently blurred' (facial features not identifiable)
   - Report percentage of ground truth faces sufficiently blurred
   - Target: >89% for faces, >94% for plates

2. **Pixel False Positive Rate (FPR):**
   - Percentage of all blurred pixels that are outside ground truth boxes
   - Important because it reflects image quality degradation
   - Target: <1-2% FPR at operating point

3. **Per-class Metrics:**
   - Faces: frontal, profile, rear-view, with occlusions
   - Plates: US (frontal, angled), EU (frontal, angled)
   - Subset recalls for different difficulty levels

**Secondary Metrics:**

4. **Inference Latency:**
   - P50, P95, P99 latency per image
   - Monitor GPU utilization

5. **Model Size & Efficiency:**
   - Parameter count
   - FLOPs (for edge deployment considerations)

**Not using standard COCO metrics (mAP@0.5:0.95):**
- COCO metrics not optimized for privacy use case
- IoU threshold of 0.5 doesn't align with 'sufficient blur' definition
- Pixel-level recall more meaningful than box-level

### Validation Strategy:

**Cross-validation on geographic regions:**
- Train on images from Cities A, B, C
- Validate on Cities D, E, F
- Ensures model generalizes

**Temporal validation:**
- Train on images from Jan-June
- Validate on images from July-December
- Captures seasonal changes in lighting, etc.

**Bias evaluation:**
- Evaluate performance across:
  - Different ethnicities
  - Age groups
  - Lighting conditions (shadows, sunlight, night)
  - Occlusions (sunglasses, hats, masks)
- Goal: <5% variance in recall across groups

### Monitoring in Production:

Once deployed:

1. **User Reports:** Track unblurred faces/plates reported by users → auto-blur and log
2. **Sampling:** Regularly sample 0.1% of processed images → manual review
3. **Confidence Distribution:** Monitor if detector confidence scores shift (concept drift)
4. **Model Staleness:** Retrain every 3-6 months with new data

### Active Learning Strategy:

To improve model over time:
- Collect user-reported failures (unblurred faces)
- Collect hard false positives (incorrectly blurred non-faces)
- Prioritize labeling these for retraining
- This efficiently improves model where it's weakest"

### Interviewer:
"Let's talk about blur itself. What algorithm would you use to blur, and why?"

---

## Phase 7: Blurring Algorithm

### Candidate:
"Interesting question! Blurring seems trivial but has nuances:

### Requirements for Blurring:
1. **Privacy:** Must make face/plate unidentifiable (not reversible)
2. **Aesthetics:** Should blend smoothly, not look jarring
3. **Speed:** Should be fast relative to detection (~100ms per image max)

### Blur Algorithms:

| Method | Implementation | Pros | Cons |
|--------|-----------------|------|------|
| **Gaussian Blur** | \(\text{output}[x,y] = \sum \text{gaussian kernel} \times \text{input}\) | Simple, fast, natural look | May not be irreversible with sharp kernels |
| **Pixelation/Mosaic** | Replace each k×k block with average color | Very obvious blurring, clearly marked | Can be harsh looking |
| **Box Blur** | Average over local neighborhood | Fast, smooth | Similar to Gaussian |
| **Face Inpainting** | Use generative model to fill region | Beautiful result | Too slow, ethical concerns |

**Google's Approach (Published):**
- Combination of **Gaussian blur + noise**
- Alpha-blend smoothly with background at box edges
- Ensures irreversibility while maintaining aesthetics

**Formula:**
\[
\text{output} = \alpha \times (\text{blur}_\text{strong} + \text{noise}) + (1-\alpha) \times \text{input}
\]

Where:
- \(\alpha\) transitions smoothly from 1 inside box to 0 at edges
- \(\text{blur}_\text{strong}\) is aggressive Gaussian blur (sigma ~25-30 pixels)
- \(\text{noise}\) is random noise to prevent any recovery

### Implementation Details:

**Kernel Size:**
- For face: kernel size 30-50 pixels (to obscure facial features)
- Verify that face identification is impossible by testing with recognition models

**Irreversibility Check:**
- Can we reverse the blur using deconvolution? → Should fail
- Can a face recognition model identify the original face? → Should fail with <10% confidence

**Performance:**
- Gaussian blur: O(N × M) where N×M is image size with optimized separable kernels
- Time: ~50-100 ms per 5MP image on CPU
- Can be parallelized on GPU for batch processing

**Edge Handling:**
- Feather the blur at box boundaries using alpha blending
- Prevents harsh rectangular artifacts
- Makes the blurred region look natural

### Implementation Pseudocode:

```python
def blur_boxes(image, boxes, kernel_size=40):
    """
    Args:
        image: Input image (H×W×3)
        boxes: List of [x1, y1, x2, y2] coordinates
        kernel_size: Blur kernel size
    """
    output = image.copy()
    
    for (x1, y1, x2, y2) in boxes:
        # Add margin for smooth alpha blending
        margin = 10
        x1_margin = max(0, x1 - margin)
        y1_margin = max(0, y1 - margin)
        x2_margin = min(W, x2 + margin)
        y2_margin = min(H, y2 + margin)
        
        # Extract region
        region = image[y1_margin:y2_margin, x1_margin:x2_margin]
        
        # Strong blur
        blurred = cv2.GaussianBlur(region, (kernel_size, kernel_size), 25)
        
        # Add noise for irreversibility
        noise = np.random.normal(0, 25, blurred.shape)
        blurred = np.clip(blurred + noise, 0, 255)
        
        # Alpha blending for smooth edges
        alpha = create_alpha_mask(y1, y2, x1, x2, margin)
        
        # Composite back
        output[y1_margin:y2_margin, x1_margin:x2_margin] = \
            alpha * blurred + (1 - alpha) * region
    
    return output
```

### Quality Check:
- After blurring, verify that:
  1. Face recognition fails: `recognition_model.confidence < 0.1`
  2. License plate OCR fails: `ocr_confidence < 0.5`
  3. Blur looks natural (subjective, A/B test)"

### Interviewer:
"Good. Now, one more critical aspect: what about the edge cases and failure modes?"

---

## Phase 8: Edge Cases & Failure Modes

### Candidate:
"Excellent question. Let me think through the failure modes:

### 1. Small Faces (< 12 pixels)
**Problem:** Detector designed for 12+ pixel faces; smaller faces are ambiguous regarding identifiability

**Solution:**
- Explicitly handle: faces < 12 pixels → don't blur (user privacy is lower anyway)
- Add separate tiny-face detector if needed (lower priority)

### 2. Faces Behind Glass/Windows
**Problem:** Reflections, glass glare, changed appearance

**Solution:**
- Include training data with such cases
- Increase detector contrast sensitivity
- Post-processor color model captures glass reflections

### 3. Partial/Cut-off Faces
**Problem:** Face at image boundary, partially visible

**Solution:**
- Explicitly include such cases in training
- Box extends to image boundary (not rejected)
- Blur what's visible → sufficient for privacy

### 4. Multiple Overlapping Boxes
**Problem:** Two detected faces with overlapping bounding boxes

**Solution:**
- Use Non-Maximum Suppression (NMS):
  \[
  \text{suppress boxes if IoU} > 0.3 \text{ and confidence difference is large}
  \]
- Or merge overlapping boxes

### 5. Faces on Billboards/Artwork
**Problem:** Celebrity faces on posters → false positives (don't need privacy protection)

**Solution:**
- **Training:** Only label real people, not artwork (during labeling)
- **Features for post-processor:**
  - High saturation (artwork often has vivid colors)
  - Unusual color distributions (not natural skin tones)
  - Too-perfect geometry (symmetry)
- **Secondary detector:** Only detect real people facing camera

**Real-world trade-off:**
- Google's system sometimes blurs billboards → acceptable cost for privacy

### 6. License Plates at Angles/Reflections
**Problem:** Angled plates, reflections, shadows, mud/dirt cover

**Solution:**
- Train separate models for frontal (0-30°) and angled (30-90°) plates
- Use two-detector channels as mentioned earlier
- Post-processor uses car context to validate

### 7. Sunglasses/Hats
**Problem:** Occlusions change facial features

**Solution:**
- Include training data with occlusions
- Ensure faces with sunglasses are in both training and validation
- Evaluate recall separately for occluded vs. non-occluded

### 8. Privacy Regulations: Different Requirements
**Problem:** Some regions require stricter privacy than others

**Solution:**
- Parameter file per region:
  ```json
  {
    "US": {"min_face_size": 12, "blur_kernel": 40, "fpr_threshold": 0.02},
    "EU": {"min_face_size": 8, "blur_kernel": 50, "fpr_threshold": 0.01}
  }
  ```
- Adjust detection thresholds and blur intensity by region

### 9. Concept Drift
**Problem:** Model performance degrades over time as image distribution changes

**Solution:**
- Monitor confidence distribution over time
- Detect if distribution shifts significantly
- Trigger retraining pipeline
- Use user-reported failures as signals

### 10. Model Failure (e.g., all zeros)
**Problem:** Detection model outputs all zeros → no faces detected

**Solution:**
- Sanity checks:
  - Check if image loads correctly
  - Check if GPU is still responsive
  - Compare against baseline model
- Fall back to previous version if failures spike
- Alert on-call engineer

### Implementation:

```python
def detect_and_blur_with_safety(image, model, config):
    try:
        # Sanity check
        if image.size == 0:
            raise ValueError("Empty image")
        
        # Detect faces
        faces = model.detect(image, conf_threshold=config.threshold)
        
        # Safety: if no faces detected in typical street image
        # (heuristic: images usually have 0-5 faces), log it
        if len(faces) > 20:
            logger.warning(f"Unusual: {len(faces)} faces detected")
        
        # Filter very small boxes
        faces = [b for b in faces if (b[2]-b[0]) > config.min_face_size]
        
        # NMS
        faces = non_max_suppression(faces, iou_threshold=0.3)
        
        # Blur
        blurred_image = blur_boxes(image, faces, config.blur_kernel)
        
        return blurred_image
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        # Return original image (privacy risk but better than crash)
        # Or try with simpler model as fallback
        return image
```

### Monitoring & Alerting:

```python
# Track metrics
metrics = {
    'avg_faces_per_image': ...,
    'avg_plates_per_image': ...,
    'inference_latency_p99': ...,
    'model_crash_rate': ...,
    'confidence_distribution': ...
}

# Alert if:
if metrics['avg_faces_per_image'] > 10 or < 0.1:
    alert("Unusual face count distribution")
    
if metrics['inference_latency_p99'] > 30_000:  # 30 seconds
    alert("Latency degradation")
```"

### Interviewer:
"Excellent. Let me now ask about model optimization since we need to process 10M images/day. What compression techniques would you consider?"

---

## Phase 9: Model Optimization & Deployment

### Candidate:
"Great question. For 10M images/day, efficiency is critical. Let me discuss compression techniques:

### 1. Quantization

**Why:** Reduce model size from 32-bit floats to 8-bit integers

**Types:**

**Post-Training Quantization (PTQ):**
- Convert FP32 model → INT8 without retraining
- Fast, easy to implement
- 4× compression (FP32 = 4 bytes → INT8 = 1 byte)
- Typical accuracy loss: 0.5-2%

**Implementation:**
\[
\text{quantized\_value} = \text{round}\left(\frac{\text{FP32\_value} - \text{min}}{\text{max} - \text{min}} \times 255\right)
\]

**Results:**
- Model size: 50-100 MB → 12-25 MB
- Inference speedup: 2-4× on CPU
- Negligible loss on COCO validation set

**Quantization-Aware Training (QAT):**
- Train model with quantization in mind
- Slightly slower to train but better accuracy
- Better for aggressive quantization (4-bit)
- For our case, 8-bit QAT recommended

### 2. Pruning

**Structured Pruning:**
- Remove entire channels/filters, not individual weights
- Better hardware compatibility (doesn't require special inference engines)
- Example: Remove 30% of filters that contribute least to output
- Result: 3-5× speedup, 50-70% parameter reduction

**Unstructured Pruning:**
- Remove individual weights based on magnitude
- Requires special inference hardware (not compatible with standard GPUs)
- Better compression but harder to deploy

**Iterative Magnitude Pruning:**
- Remove bottom K% of weights by magnitude
- Retrain to recover accuracy
- Repeat until target compression reached

**Example:**
```python
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.structured_prune(module, 'weight', 
                               pruning_method='magnitude',
                               amount=0.3)  # Remove 30%
```

### 3. Knowledge Distillation

**Concept:** Train smaller 'student' model to mimic larger 'teacher' model

**Benefit:** Student is 10-50× smaller but maintains accuracy

**Process:**
\[
L_{\text{distillation}} = \alpha L_{\text{CE}}(y, \hat{y}_{\text{student}}) + (1-\alpha) L_{\text{KL}}(T_{\text{teacher}}, T_{\text{student}})
\]

Where:
- \(T = \frac{\text{logits}}{temperature}\) (soften predictions)
- \(\alpha\) = 0.1-0.3 typically

**Trade-off:** Takes time to train but results in much smaller model

### 4. Architecture Search / Model Selection

**Current Options:**
| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|------------------|
| YOLOv8n | 3.2 MB | 80 ms (CPU) | mAP 37.3 | ✓ Good for edge |
| YOLOv8s | 11.2 MB | 128 ms (CPU) | mAP 44.9 | ✓ Default |
| YOLOv8m | 25.9 MB | 234 ms (CPU) | mAP 50.2 | ✗ Too slow for CPU |
| MobileNetV3 | 4.2 MB | 15 ms | Lower acc | ✓ Edge devices |

**My recommendation:**
- Primary: **YOLOv8s + 8-bit quantization**
  - Balanced accuracy and speed
  - 12 MB → 3 MB after quantization
  - Inference: ~50 ms per image on GPU

- Fallback: **YOLOv8n (nano)**
  - If inference budget is tighter
  - Slightly lower accuracy trade-off

### 5. Batch Inference Optimization

**Key insight:** Modern GPUs are designed for parallel processing

**Batching effect on throughput:**

\[
\text{Throughput per GPU} = \frac{\text{Batch Size}}{\text{Latency per Batch}}
\]

Example with YOLOv8s:
- Single image: 2.66 ms
- Batch 64: 150 ms total = 2.3 ms per image (slight overhead)
- Throughput: 64 / 0.0023 = **27,800 images/second**

**For 10M images/day:**
\[
\text{GPUs needed} = \frac{10^7 \text{ images}}{86,400 \text{ sec/day} \times 27,800 \text{ img/GPU/sec}} = 0.004 \text{ GPU}
\]

Wait, this seems too good. Let me recalculate more realistically:

- Batch 32: ~85 ms per batch = 2.7 ms per image
- Throughput: 32 / 0.0027 = **11,800 images/second per GPU**
- For 10M images: 10^7 / 11,800 = **847 seconds** = **14 minutes** on a single GPU!

With overhead and batching delays: approximately **1-2 GPUs needed**.

But we'd want redundancy:
- Aim for 4-6 GPUs for headroom, failover, model updates

### 6. Export & Deployment Formats

**TensorFlow SavedModel:**
- Native TensorFlow format
- Good for TensorFlow Serving
- TensorRT conversion available

**ONNX (Open Neural Network Exchange):**
- Hardware-agnostic
- Can convert to various runtimes (TensorRT, CoreML, ONNX Runtime)
- Good for model portability

**TensorRT (NVIDIA):**
- Optimized for NVIDIA GPUs
- Auto kernel fusion, layer optimization
- Up to 2× speedup vs. standard inference
- Requires NVIDIA hardware

**Recommendation:**
- Export to both ONNX and TensorRT
- Use TensorRT for GPU production
- ONNX runtime for CPU/validation

### 7. Real-world Latency Breakdown

For a complete pipeline:

```
Raw Image (~25 MB JPEG)
    ↓ [50 ms] Image decode + resize
Preprocessed (1×3×640×640)
    ↓ [2.7 ms] Inference (batch 32)
Detected boxes [x1, y1, x2, y2, conf]
    ↓ [5 ms] Post-processing NN
Filtered boxes
    ↓ [15 ms] Blurring
Blurred image
    ↓ [10 ms] JPEG encode + write

Total: ~80-90 ms per image
For batch 32: ~77 ms per image (amortized)
```

**With 10M images and 77 ms per image:**
\[
\text{Total time} = 10^7 \times 0.077 \text{ sec} = 770,000 \text{ sec} ≈ 9 \text{ days on single GPU}
\]

**With 2 GPUs:** ~4.5 days
**With 4 GPUs:** ~2.25 days
**With 6 GPUs:** ~1.5 days

This fits our batch processing requirement comfortably."

### Interviewer:
"Let me challenge this. What if we need to process in real-time? How would the design change?"

---

## Phase 10: Real-Time Processing (Bonus)

### Candidate:
"Interesting pivot! Real-time changes the entire architecture:

### Real-Time Requirements:
- Latency: <100 ms per image (10 FPS)
- Throughput: Depends on number of video streams

### Architecture Changes:

**From batch to streaming:**

```
Video Stream
    ↓
Frame Buffer (queue recent frames)
    ↓
Lightweight Detector (e.g., YOLOv8n)
    ↓
Track faces across frames (Kalman filter)
    ↓
Blur & Display
```

### Key Optimizations:

1. **Temporal Tracking:** Don't re-detect every frame
   - Use object tracking (DeepSORT, ByteTrack)
   - Only run detector every K frames
   - Interpolate boxes between detection frames
   - Reduces computation by 3-5×

2. **Model Pruning:** Use smallest viable model
   - YOLOv8n: 37.3 mAP but 3.2 MB, 80 ms inference
   - Trade slightly lower accuracy for speed

3. **Quantization + TensorRT:** Critical for real-time
   - INT8 quantization: 2-4× speedup
   - TensorRT: Another 1-2× speedup
   - Combined: 4-8× speedup vs. baseline

4. **GPU Inference in Parallel:**
   - Video decode on CPU
   - Detection on GPU
   - Blur on GPU
   - Encode on CPU
   - Pipelined execution

### Latency Budget:

```
Frame capture: 33 ms (30 FPS video)
  Decode: 10 ms
  Resize: 5 ms
  Inference (detection): 20 ms (YOLOv8n + quantization)
  Post-processing: 3 ms
  Blur: 10 ms
  Encode: 10 ms
Total: 48 ms → 30 FPS achievable
```

### Mathematical Model:

For \(N\) cameras streaming at \(F\) FPS:
\[
\text{Compute needed} = N \times F \times \text{latency per frame}
\]

Example: 100 cameras at 30 FPS with 20 ms detection:
\[
100 \times 30 \times 0.020 = 60 \text{ GPU-seconds per second}
\]

With batch processing (batch 8):
\[
\text{Effective latency} = 20 \text{ ms}, \text{ throughput} = 400 \text{ frames/sec per GPU}
\]

For 3,000 frames/second (100 cameras × 30 FPS):
\[
\text{GPUs needed} = \frac{3,000}{400} = 7.5 → 8 \text{ GPUs}
\]

### Challenges:

1. **Tracking drift:** Bounding boxes drift over time if not re-detected
2. **Ghost boxes:** Tracker creates boxes for non-existent objects
3. **ID switching:** Objects switch identities

**Solution:** Re-detect on every Kth frame (K=3-5), update tracks

### Architecture Diagram:

```
Camera inputs (30 FPS × 100 cameras)
    ↓
Frame queues (FIFO, max 3 frames per camera)
    ↓
Detection Worker (GPU)
    ├─ Every 3rd frame: Full detection
    ├─ Other frames: Track-only
    ↓
Tracking Module (CPU) - Kalman filter
    ↓
Blur Worker (GPU)
    ↓
Display / Output
```

---

## Summary & Candidate Wrap-up

### Candidate:
"Let me summarize the complete design:

### High-Level Solution:
1. **Batch processing system** for 10M images/day using 4-6 GPUs
2. **Two-stage detection:** High-recall primary detector + Neural network post-processor
3. **Dual detection:** Separate models for faces and license plates
4. **Quality metrics:** Hand-counted pixel recall (89%+ target) + Pixel FPR (<2%)
5. **Blur:** Gaussian blur + noise with alpha blending for natural look
6. **Optimization:** Quantization, batching, TensorRT for efficiency
7. **Monitoring:** User reports, sampling, confidence tracking for production
8. **Edge cases:** Handled through training data diversity and per-region configurations

### Trade-offs Accepted:
- **Privacy over precision:** Better to blur a billboard than miss a face
- **Manual review overhead:** Use human-in-the-loop for uncertain cases
- **Regional customization:** Different models/thresholds for different markets

### Future Improvements:
- Active learning with user-reported failures
- Federated learning if multiple regions maintain separate models
- Integration with facial recognition models to test irreversibility
- Real-time streaming version if requirement changes"

### Interviewer:
"Excellent. You've shown deep technical understanding of:
- Object detection algorithms and trade-offs
- System design at scale
- Evaluation metrics for privacy-critical systems
- Model optimization and deployment

One final question: **What would you do differently if you were building this system today vs. 2008 when Google did?**"

### Candidate:
"Great question!

**2008 Approach:**
- Limited GPU availability → sliding window was faster
- Limited training data → hand-crafted features (Haar, HOG, Gabor)
- Limited model diversity → single detector per object type

**2024 Approach (today):**
- Abundant GPU resources → can run large models
- Massive labeled datasets available → deep learning dominates
- Model zoo → can choose from YOLO, RT-DETR, etc.
- Transfer learning → pre-trained on COCO, fine-tune on Street View data
- Advanced deployment → TensorRT, quantization, knowledge distillation

**I would:**
1. Start with **pre-trained YOLOv8** fine-tuned on Street View
2. Use **quantization-aware training** from day one
3. Leverage **transfer learning** from existing face detection datasets
4. Use **automated architecture search** if Pixel recall doesn't hit target
5. Deploy with **TensorRT** and **batch inference**
6. Integrate **monitoring from day one** (not as afterthought)

**However,** the core insight from 2008 remains valid: For privacy-critical systems, sometimes the asymmetric cost of errors requires a **specialized two-stage approach** rather than relying on a single general-purpose model."

---

## Interview Conclusion

**Interviewer:**
"Fantastic. You've demonstrated:
- ✓ Clear requirement gathering and clarification
- ✓ Systematic high-level design with trade-off analysis
- ✓ Deep technical knowledge of computer vision and ML
- ✓ Practical system design thinking (scalability, monitoring)
- ✓ Mathematical rigor and ability to quantify claims
- ✓ Understanding of privacy as a first-class metric
- ✓ Willingness to challenge assumptions and adapt

This is a strong interview. Questions?"

