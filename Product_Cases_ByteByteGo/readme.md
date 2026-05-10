# Product Cases Prep

> SEE ALSO: junfanz1 Awesome-AI-Engineer-Review main System Design -- `ML System Design Interview.md`
> SEE ALSO: alirezadir's ML System Design  [notes](https://github.com/alirezadir/Machine-Learning-Interviews/tree/main/src/MLSD)

> Ideally should have purchased Alex Xu's book but all good

Note that the following Chapters are already available on ByteByteGo. I've taken them as-is

- Visual Search System
- YouTube Video Search
- Video Recommendation System
- Personalized News Feed

The other .md files are generated with Claude Opus 4.6 Extended. Have also included Alreza's notes for these. Credits to Salu for access to Claude!

Things that I have liked -

- Three clarifying questions is the sweet spot [source](https://buildml.substack.com/p/data-science-case-study-design-a) `ACRONYM: BSLV`
  - (1) business objective -- drive engagement/increase revenue
  - (2) scale -- # DAU, # items, # interactions per day
  - (3) latency -- # p90 during inference
  - (4) variables avaiable (JUST TELL) -- impressions, watch times, likes, user featurres. item features, session context etc
- Metrics
  - Offline: AUC, NDCG, MRR, etc
  - Online: CTR, time spent (*dwell time*), increase in revenue, user 
  satisfaction (survey)
  - SYSTEM HEALTH: P99 latency, cache hit rate, throughput
- Ask for population demographics (global or local)
- Ask for negative signals (report/ dislike/ hide/ block)
  - How do +ves and -ves look like
- Input features can be batch processed (user profile, item features), or real-time (session context, interaction features)

- Determine if continual training is needed
- Is personalization needed? (if not, can use simpler models)
satisfaction (survey)
  - How to marry Offline and Online metrics? A/B test for external traffic / hold-out friends and family for internal traffic

- `Event Recommendation System is the most complete one`

- **two tower architectures are used for asymmetric problems**. If query is just an item ID, or is it an item + context?
  - If "similar listings" = listings similar to this listing (item-item) → Listing2Vec / single encoder
  - If "similar listings" = listings similar to this listing, for this user, in this session → Two-tower, because the query is now (listing, user, session_context) which is structurally different from a candidate listing

- For images/videos
  - ResNet
  - CLIP
  - ViT - after sampling frames for videos
  - ViViT - for full videos

Advantages/Disadvantages

- Multi-tower can take into account many types of inputs, have shared learning body, and have multiple heads for different tasks (click, like, etc.)

| **Reaction** | Click | Like | Comment | Share | Friendship request | Hide | Block |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Predicted probability** | 23% | 48% | 12% | 4% | 0.1% | 0.005% | 0.0003% |
| **Value** | 1 | 5 | 10 | 20 | 30 | -20 | -50 |
| **Score** | 0.23 | 2.4 | 1.2 | 0.8 | 0.03 | -0.001 | -0.00015 |
| Engagement score = 4.65885 |

- **DCN-v2** for feature crossing (e.g. user_id x post_id) 

- Can have a distilled/lightweight (XGBoost works) pre-ranker before the heavier reranker -- need not do if retreiever is already good enough
  - Inputs to XGBoost can be tabular features (engagement rate, recency, historical rates, topic match score, session activity score)

- Calibration of scores -- scaling/plotting predicted scores against actual probabilities, or using techniques like Platt scaling or isotonic regression -- to ensure that the predicted scores can be interpreted as probabilities of engagement

- Index refresh depending on how fast content changes - for newsfeed, can be every few hours, for similar listings, can be every few days

- USE COMPANY-LEVEL FEATURES
  - LinkedIn has professional connections
  - Youtube has watch history, subscriptions, etc.

- How to ensure fast inference? (200 ms latency)
  - ANN searches for retriever
  - Index creation by FAISS (IVF + HNSW), Qdrant for LLM-level vector searches (whole internet)
  - Index store Redis
  - Train over multiple GPUs (model parallelism) or data parallelism (multiple copies of model)
  - Distillation for faster inference
- Postprocessing/ re reranking
  - Diversity injection (not same host/seller, topic etc.) -- cap at 2 from top 10, show varied prices, etc.
  - Confidence boost: add model thresholds + minor boost scores
  - Promote sponsored content (ads) -- cap at 1 from top 10
  - Geographical rules
  - Business rules (promote new listings)
  - Stale content removal, data freshness
  - Remove hate speech, misinformation etc (assurance layer)

COMMON QUESTIONS

- Positional bias in ranking -- mitigate by randomizing the order of candidates during training, or by dropping the position feature during training (but can use it during inference)
  - Inverse Propensity Scoring: Click at Position1 is cheap (because everyone sees it) -- downweight training sample; Click at Position20 is GOLD -- upweight training sample
- Why Multi-task? -- Data efficiency (rare actions have lesser training data); shared learning; maintenance
- Cold start -- (FOR USERS): signup survey, basic info like industry, preferences - then use these to map into existing clusters/profiles ; (FOR ITEMS): extract title, description, metadata, image embeddings - then use them to map into existing user clusters/profiles
- When to know if enough history is available -- use rules like (atleast 10 interactions, or 1 interaction/day for 7 days) 
- How long to run tests -- at least 2 weeks to capture weekly patterns, we usually reach sample sizes within 1-2 days; min significant effect over baseline (2%, 5%)




In ML system design interviews, selecting the correct architecture is less about "the best model" and more about the **constraints of the data and the latency budget**. Below are the crisp patterns to identify when to use specific architectures based on the provided sources.

### 1. The Multi-Stage Funnel Pattern
**When to use:** Use this anytime the candidate pool is large (e.g., >100k items) and the latency budget is tight (e.g., <200ms).
*   **The Flow:**
    *   **Retrieval (Candidate Generation):** Fast, high-recall models (Two-Tower) narrow down billions of items to hundreds or thousands.
    *   **Ranking (Scoring):** Heavy, high-precision models (DCN v2, MTL) score the narrowed list using rich features.
    *   **Re-ranking:** Apply business logic (diversity, safety filters, freshness) to the top ~50 results.

### 2. Two-Tower Architecture
**When to use:** Use for **Retrieval** stages or when you need to serve results from a massive corpus.
*   **The Pattern:**
    *   **High Latency/Scale:** Decoupling user and item towers allows you to **pre-compute and index** item embeddings into an Approximate Nearest Neighbor (ANN) store.
    *   **Severe Cold Start:** Excellent for new items because the item tower can generate embeddings from **content features alone** (e.g., title, category) without needing interaction history.
    *   **Domain Separation:** When inputs come from distinct entities (e.g., User features vs. Video features) that don't interact until the final dot product.

### 3. DCN v2 (Deep & Cross Network)
**When to use:** Use for **Ranking** stages where feature interactions are critical but structured.
*   **The Pattern:**
    *   **Structured/Tabular Data:** When you have many categorical features where the **multiplicative interaction** (e.g., "Age_Bucket × City × Category") is a stronger signal than the features alone.
    *   **Efficiency:** MLPs approximate interactions inefficiently via many ReLU layers; DCN v2 models bounded-degree polynomial interactions explicitly and more parameter-efficiently.

### 4. Multi-Task Learning (MTL)
**When to use:** Use when the system must optimize for multiple competing objectives or deal with sparse labels.
*   **The Pattern:**
    *   **Multiple Objectives:** In News Feeds or Video systems, you often want to maximize **Clicks, Likes, Shares, and Watch Time** simultaneously.
    *   **Label Sparsity (Regularization):** A dense task (e.g., Clicks) can provide a "shared representation" that helps a sparse task (e.g., Purchases or RSVPs) learn faster.
    *   **Passive Users:** Add specific heads for **Dwell-time** or **P(Skip)** to capture engagement from users who never click or like anything.

### 5. Multi-Tower (Multi-Modal)
**When to use:** Use when the input itself is heterogeneous (e.g., a post with text, image, and video).
*   **The Pattern:**
    *   **Late Fusion:** Run separate models for each modality and combine scores at the end. Use this for **simplicity and independent model iteration**.
    *   **Early/Hybrid Fusion:** Combine features before the final model layers. Use this for **complex interactions**, such as detecting harmful memes where the image is benign but the text makes it harmful.

### 6. Single vs. Multiple Models: Decision Tricks
*   **Use a Single Model (MTL/Shared Backbone) when:**
    *   Tasks are related (e.g., Violence vs. Hate Speech detection).
    *   You have limited training data per category and want to leverage "knowledge transfer".
    *   You have a tight inference budget—running one backbone is cheaper than three.
*   **Use Multiple Models (Independent) when:**
    *   Tasks are unrelated or have vastly different data distributions.
    *   Different teams are responsible for different outputs (e.g., Ads ranking vs. Organic Feed ranking).
    *   Latency is not the bottleneck, and you want to avoid the "gradient dominance" problem where one task overpowers others during training.

### Architect's Summary Table

| Requirement | Architecture Choice |
| :--- | :--- |
| **Search 1B items in 50ms** | Two-Tower + ANN (Retrieval) |
| **Complex category crosses** | DCN v2 (Ranking) |
| **Clicks vs. Conversions** | Multi-Task Learning (MTL) |
| **New items with no clicks** | Two-Tower (Content-based) |
| **Social context/FoF** | GNN or Graph Features |
| **Passive user engagement** | MTL with Dwell-time/Skip heads |


Based on the patterns established in ML system design, architectures are selected based on **scale**, **latency**, and **data heterogeneity**. Below is the bucketed breakdown of various systems, their architectural choices, and the technical "niches" that define them.

### ML System Architecture Identification Table

| System Example | Key Architecture | Why this Pattern? | Brief Architecture Overview | Specific Niche / Detail |
| :--- | :--- | :--- | :--- | :--- |
| **Video/Movie Recommendation** | **Two-Tower Funnel** | **Scale (10B items) & Latency.** Decoupling user/item towers allows pre-computing 10B video embeddings for ANN search. | A retrieval stage (Two-Tower) narrows billions to thousands, followed by a heavy ranking stage (DCN/NN) for high-precision scoring. | **Cold Start Heuristics:** For new videos, use embeddings of geographically nearby items until interaction data is gathered. |
| **Ad Click Prediction** | **DCN v2 / DeepFM** | **Structured Interactions & Sparsity.** Categorical crosses (e.g., City × Category) are critical; LR/MLPs are too inefficient for these multiplicative signals. | Combines a Deep network for generalization and a Cross/FM network to automatically model complex feature interactions. | **Continual Learning:** Performance degrades significantly with even a 5-minute delay; requires online feature computation and streaming updates. |
| **Event Recommendation** | **Content-Based Two-Tower** | **Temporal Perishability.** Events expire quickly, so "Collaborative Filtering" fails; towers must rely on **content features** (title, venue) for immediate retrieval. | Multi-stage funnel using geographic pre-filtering, Two-Tower retrieval for new items, and DCN-v2 for personalized ranking. | **Incremental Index Refresh:** Requires real-time inserts/deletes (e.g., Milvus/Vespa) as items are created and expire hourly. |
| **Harmful Content Detection** | **Multi-Task (MTL) & Hybrid Fusion** | **Emergent Harm & Explainability.** Harm often arises from the *combination* of image and text (Early/Hybrid fusion) and requires separate heads for explanation. | A shared multimodal backbone (DistilmBERT/CLIP) extracts features used by multiple classification heads (Violence, Nudity, etc.). | **Cascaded Inference:** To handle 500M posts/day, use Stage 1 heuristics (hashes) to catch 80% of bad content cheaply before full inference. |
| **People You May Know (PYMK)** | **GNN / Edge Prediction** | **Graph Context.** Social connections (Friends-of-Friends) are the strongest signal; a simple binary classifier misses 2-hop neighborhood insights. | Uses Graph Neural Networks to produce node embeddings, calculating dot-product similarities to predict the probability of a new edge. | **Pre-computation Efficiency:** Unlike Ads, social graphs are stable; recommendations can be batched daily since user connections don't change fast. |
| **Visual Search (Pinterest style)** | **Representation Learning** | **Visual Similarity.** Only pixel data is available (no text query), requiring a model to map images into a shared N-dimensional embedding space. | A deep CNN/Transformer encoder (ResNet/ViT) transforms the query image into an embedding for ANN lookup against billions of items. | **Self-Supervised Labeling:** Use data augmentation (rotating/cropping query images) to create "positive" training pairs without human annotators. |

---

### Patterns for Identifying Architectures

1.  **When to use Two-Tower vs. Single Model:**
    *   **Two-Tower:** Use when you have a **huge corpus** (1M+ items) and need to retrieve in <50ms. It allows you to separate the item tower, pre-compute its embeddings, and store them in an ANN index.
    *   **Single Model (MTL):** Use when you have **multiple related labels** (e.g., Click, Like, Share). It improves performance on sparse labels by sharing the backbone's knowledge from dense labels.

2.  **When to use a Funnel (Retriever + Ranker):**
    *   Use this when the **Candidate Pool > 100k**. Scoring 100k items with a heavy model (DCN v2) would take seconds; retrieval narrows it to ~500 items that the ranker can score in ~50ms.

3.  **When to use DCN v2 vs. Standard MLP:**
    *   **DCN v2:** Use for **ranking tabular data** with many categorical IDs. It captures multiplicative interactions (e.g., "User_History × Ad_Context") much more parameter-efficiently than standard ReLU layers.

4.  **When to use GNNs:**
    *   Use when the **relationship between entities** (edges) is more important than the entities' individual features (nodes). This is standard for PYMK or fraud detection where "who you know" defines "who you are".

**Would you like me to create a quiz to test your ability to match these patterns to new ML design scenarios?**