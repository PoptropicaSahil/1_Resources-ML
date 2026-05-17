# FROM GEMINI. Clearing MAJOR doubt

> Comprehension is Compression!

## LTR for Amazon Books/Audible (<1s latency)

* **Two-Tower architecture for Candidate Retrieval** (Stage 1)
* **Multi-Task, Multi-Tower architecture for Reranking** (Stage 2)

---

### 1. Latency Budget Allocation (Target: < 500ms server-side)

* **User Feature Fetching (Redis/Feature Store):** ~15-20ms
* **Stage 1: Retrieval (ANN Search):** ~30-50ms
* **Item Feature Fetching (for candidates):** ~30-50ms
* **Stage 2: Reranking (Model Inference):** ~150-200ms
* **Post-processing (Business logic, deduplication):** ~20ms

---

### 2. Stage 1: Candidate Retrieval (The Two-Tower Model)

* **User Tower:** Historical interactions, search query text embeddings, genre preferences, Kindle/Audible usage metrics $\rightarrow$ dense layers $\rightarrow$ embedding vector $E_u \in \mathbb{R}^d$
* **Item Tower:** Title/Abstract embeddings via a frozen MiniLM, author ID, format type (audio vs. paperback), category tags, popularity score $\rightarrow$ embedding vector $E_i \in \mathbb{R}^d$
* **Interaction:** Similarity using dot product/ cosine similarity: $s(u, i) = \langle E_u, E_i \rangle$

> Remember $ \langle \text{user, item} \rangle $ pair

**Training Strategy:**

* **Loss**: InfoNCE (contrastive)
* **Sampling**: **In-Batch Negative Sampling**

```math
\mathcal{L} = -\log \frac{\exp(\langle E_u, E_{i^+} \rangle / \tau)}{\exp(\langle E_u, E_{i^+} \rangle / \tau) + \sum_{j \in \text{batch}} \exp(\langle E_u, E_{i_j^-} \rangle / \tau)}
```

*(where $\tau$ is temperature)*

* **Hard Negatives:** Items the user saw but didn't click/purchase
* *Mixed Negatives:* Prevent the model from only learning trivial distinctions

**Inference & Latency Optimization:**

* **Offline:** Index all books/audiobooks using through Item Tower offline. **Approximate Nearest Neighbor (ANN)** index -  **FAISS (HNSW)** or Amazon OpenSearch k-NN
* **Online:** `At request time, only the User Tower executes`. Returning the top 1,000 from ANN index (<20ms)

---

### 3. Stage 2: Reranking (Multi-Tower / Multi-Task Model)

Since Audible & Amazon Books have different conversion funnels, **Multi-Task Learning (MTL) Multi-Tower** is ideal

**Architecture** : Multi-gate Mixture-of-Experts (MMoE) + DCN v2

* **Feature Crosses:** `Pass dense user, item, and contextual features into a DCN layer` to explicitly learn bounded-degree feature interactions (e.g., *User prefers Sci-Fi* $\times$ *Book narrated by Ray Porter*).
* Tasks by final gates: CTR; CVR - Book (purchase); CVR - Audible (using Audible credit / streaming > 30 minutes)

**Loss Function combines:**

* Pointwise (Binary Cross Entropy for conversions)
* Listwise loss (e.g., ListMLE or LambdaRank)

```math
\mathcal{L}_{total} = \alpha \mathcal{L}_{CTR} + \beta \mathcal{L}_{CVR\_Book} + \gamma \mathcal{L}_{CVR\_Audible} + \delta \mathcal{L}_{Rank}
```

**Final Sorting:**
Calibrated linear combination of MTL outputs for the final sort

```math
Score = p(Click) \times [w_1 \cdot p(Purchase) + w_2 \cdot p(Listen)]
```

---

### 4. Critical Feature Engineering

* **Behavioral Sequences:** `Attention (like DIN - Deep Interest Network) over the user's last 20 viewed/purchased ISBNs`
* **Cross-Domain Features:** Bridge reading and listening. Does the user typically read on Kindle but switch to Audible for non-fiction?
* **Temporal Context:** Time of day/day of week
* **Cold-Start Mitigation:** Node2Vec on the author-book graph for new inventory

---

### 5. Achieving Sub-Second Latency in Stage 2 (Engineering)

1. **Concurrent Scoring:** Batch the 1,000 user-item pairs and parallelly on GPU + Triton Server
2. **Model Compilation & Quantization:** **ONNX** or TensorRT dump. For MTL models, INT8 usually maintains NDCG metrics while halving latency.
3. **Two-Phase Reranking (Optional but recommended):** If 1,000 is too heavy for MTL model, insert a intermediate LightGBM/XGBoost $\rightarrow$ 200 candidates
4. **Feature Caching:** Cache item-side features (Redis cluster with RedisJSON) $\rightarrow$ reranker only fetchs user profile and the interaction history at runtime

### DCN-v2 WORKING

Explicitly learns bounded-degree feature interactions while simultaneously learning implicit, non-linear interactions.

#### 1. Inputs

Flattened, concatenated 1D vector $\leftrightarrow$ user-item-context interaction

Concat of

* **Categorical features' Embeddings:** (User ID, Item ID, Author ID, Genre, Device Type) $\rightarrow$ embedding layers $\rightarrow$ dense vectors (e.g., 32/64 dims)
* **Continuous Features Embeddings:** (Item price, historical user CTR, audiobook length, time of day) $\rightarrow$ normalized or bucketed $\rightarrow$ dense vectors

> Concat into single vector

#### 2. How DCN-v2 Works

Passes input vector through two parallel networks

**A. Deep Network (Implicit Interactions)**
MLP + ReLU $implies$ generalization, non-linear patterns

**B. Cross Network (Explicit Interactions)**
`Applies polynomial feature crossing at each layer`

```math
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
```

Where:

* $x_l$ = previous layer output
* $x_0$ = original input vector `(CRUCIALLY, original input is injected into **EVERY** layer to create higher-order crosses)`
* $W_l$ = (learnable) weight matrix; $b_l$ = bias; $\odot$ is the element-wise product
* If 3 cross layers $\implies$ up to 4th-order feature interactions (e.g., *Gender Male* $\times$ *Time Evening* $\times$ *Genre Sci-Fi* $\times$ *Format Audio*).

#### 3. Architecture

```text
=======================================================================
                        DCN-v2 PARALLEL ARCHITECTURE
=======================================================================

[Dense Features]  [Sparse Embeddings (User, Item, Context)]
       \                        /
        \                      /
         [Concatenated Input Vector: x_0]
                 |
        +--------+---------------------------+
        |                                    |
[Cross Network]                       [Deep Network]
| Layer 1: x_1 = x_0*(W_0*x_0)+x_0 |          | Dense Layer + ReLU |
| Layer 2: x_2 = x_0*(W_1*x_1)+x_1 |          | Dense Layer + ReLU |
| Layer 3: x_3 = x_0*(W_2*x_2)+x_2 |          | Dense Layer + ReLU |
        |                                     |
        +--------+----------------------------+
                 |
         [Concatenate (or Add)]
                 |
          [DCN Output Vector]

```

#### 4. Placement in the Reranker Multi-Tower System

```text
=======================================================================
               FULL RERANKER INTEGRATION (DCN -> MMoE)
=======================================================================

              [Raw User & Item Features]
                         |
                 [DCN (Shared Bottom)]  <-- DCN sits here
                         |
                 [DCN Output Vector]
                         |
           +-------------+-------------+
           |             |             |
       [Expert 1]    [Expert 2]    [Expert 3]
           |             |             |
       +---+-------------+-------------+---+
       |                 |                 |
    [Gate 1]          [Gate 2]          [Gate 3]
       |                 |                 |
 [CTR Tower]     [Book CVR Tower]  [Audible CVR Tower]
       |                 |                 |
  P(Click)          P(Buy Book)       P(Buy Audio)

```

---