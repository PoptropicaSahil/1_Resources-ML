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