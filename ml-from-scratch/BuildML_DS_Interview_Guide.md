# Notes from BUILD ML's Data Science Interview Guide

> <https://buildml.substack.com/p/data-science-interview-guide-part-fbc>

## Part 4 Decision Trees

Series of yes/no questions aimed to obtain purer nodes. Since no assumptions about data, can capture complex, non-linear relationships.

### (Implicit) Assumptions

- IID data: Time series violates this
- Sufficient data: With small data, trees can easily overfit to noise
- Feature relevance: Irrelevant features = poor splits
- Axis-aligned decisions: Age >35 or Income>50k, but not Age + Income > 100k. *Random Forests can overcome*

### Splitting criteria

> Majority classes are favoured - since Gini/entropy  are biased towards features that separate large groups (majorly moajorty class)

- **Entropy ($H(S)$)**: Level of disorder. For a dataset $S$ with $c$ classes

```math
H(S) = - \sum_{i=1}^{c} p_i \log_2 p_i
```

where $p_i$ is the proportion of examples in class $i$. Lower entropy = more pure

**Information Gain ($IG(S, A)$)** = Reduction in entropy after split based on feature $A$. Higher gain = better split

```math
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
```

where $S_v$ is the subset of $S$ where feature $A$ takes value $v$  

- **Gini Impurity**: How often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. Lower Gini = more pure

> *BRO WHAT?*

```math
G(S) = 1 - \sum_{i=1}^{c} p_i^2
```

- **Variance Reduction (For regression trees)**: Minimise variance of target values in each split

```math
\text{Var(S)} = \frac{1}{|S|} \sum_{i=1}^{|S|} (y_i - \bar{y})^2
```

where $\bar{y}$ is the mean of target values in $S$. Higher variance reduction = better split

### Other points

- Feature scaling: Not needed since based on thresholds
- Encoding: Mandatory
- **Feature importance**: Based on how much a feature reduces impurity across all splits where it is used. Higher reduction = more important
- Stopping criteria: Max depth, Min samples per split/leaf, min impurity decrease
- Pruning: Pre-pruning (early stopping) vs Post-pruning (grow full tree then trim branches that add little predictive power)

```py
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
```

- Other hyperparams: max_leaf_nodes
- Cross validation for model stability
-

---

## Part 6 XGBoost

Uses gradient of the loss function (i.e. direction) to decide what next tree learns (it learns residuals). Gradient becomes the target of the next tree i.e. each tree models gradient of the loss wrt current predictions

Final model is the sum of many small trees. Loss has (i) penalty for bad predictions AND (ii) penalty for building overly complex trees --> XGBoost gives simpler, more generalisable models

Two values for each training example -

- Gradient: (for squared error) gradient is pred - actual
- Hessian: (for squared error) Hessian is constant -- info about **CURVATURE** of the loss

Once known, examine potential splits for next tree, sum up their gradients and hessians. **XGBoost does not need original labels anymore**. Gradients and hessians carry enough information to evaluate splits.

For each leaf, we calculate Leaf Score = correction to be added to predictions for any point landing in that leaf

```math
w^{*} = - \dfrac{G}{H + \lambda}
```

where

- $G$ is sum of gradients
- $H$ is sum of hessians for the leaf

> It is a regularised second order Taylor expansion of the loss

To check if a split is worth, information gain is checked

```math
\text{Gain} = \dfrac{1}{2} \left( \dfrac{G_L^2}{H_L + \lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right) - \gamma
```

where

- $G_L$ and $H_L$ are the sum of gradients and hessians for the left child
- $G_R$ and $H_R$ are the sum of gradients and hessians for the right child
- $\lambda$ is a regularization parameter that controls the complexity of the model
- $\gamma$ is a penalty term for adding a new leaf - ensuring only useful splits are made

> Everything the tree needs i.e. leaf scores, split quality, corrections is determined by gradients and hessians. No need for repeated scans over original labels **This is the key insight that allows XGBoost to be efficient and powerful**

Leaf scores added up via a learning rate, to slow down updates. The final training loop becomes

1. Gradients and hessians identify errors
1. Leaf scores compute the best corrections
1. Gain calculations determine the best splits
1. Leaf-wise growth focuses the model where improvement matters most
1. Learning rate ensures steady progress over abrupt jumps

### What makes it special

- Regularisation: L1 and L2 penalties directly in objective function
- Identifying Split Points: **Weighted quantile sketch algorithm** to efficiently find split points, even for large datasets, instead of sorting all data points. The sketch approximates the feature distribution while including gradient and hessian of each house. *Points with large gradients and hessians have more influence on the sketch, ensuring important splits are not missed*
- Handling Missing Values: Learns a default direction (left/right) for missing values during training. During inference, missing values are sent down the default path
- Parallel Training: Cores can evaluate splits across different features at once
- DMatrix for Optimised Storage: Provides fast access to gradients and hessians

### Important Hyperparameters

- max_depth
- min_child_weight: Minimum sum of hessians required to create a new leaf. High values --> more evidence needed to create a leaf --> simpler model
- gamma: Minimum loss reduction required to make a split. Higher gamma --> more conservative model
- lambda: L2 regularisation
- alpha: L1 regularisation - weak/noisy leaf nodes pushed to zero
- max_delta_step: Max step size for leaf score updates
- eta: Learning rate
- n_estimators: Number of boosting rounds (trees)
- early_stopping_rounds: Stop if no improvement after this many rounds
- scale_pos_weight: Useful for imbalanced datasets
- subsample: Fraction of training data for each tree

**histogram-based splits** are faster for large datasets, but may reduce accuracy

### Feature importance

- **Weight**: Number of times a feature was used in splits across all the trees. Higher appearances --> more important. *Can be misleading* if a feature is used in many shallow splits but has low impact on predictions
- **Gain**: Average gain across all splits where the feature is used. Higher gain --> more important. *More informative* than weight as it reflects contribution to reducing loss
- **Cover**: Average number of samples affected across all splits where the feature is used. Higher cover --> more important. *Can be misleading* if a feature is used in splits that affect many samples but has low gain

> Importance scores do not reveal directionality. Correlated features can distort importance. Best to compare across cross validation folds. SHAP is more informative

---

## Part 6.5 CatBoost

- Raw categoricals: **Hashes and created ordered target statistics (CTR)** - smooth target encodings without the samples own target
- Category combinations (cross features)
- Missing categories are treated as a separate category
- Builds symmetric (oblivious) trees: At each depth, one best splot is chosen and apploed to all current leaves
- Uses ordered boosting: Random permuation of data to compute gradients. For each training example, only uses previous examples in the permutation to compute gradient. Prevents target leakage and overfitting
- Built-in SHAP

---

## Part 7 kNN

New point arrives, distances are computed, neighbors selected, their outputs combined. (both regression and classification)

> Since kNN makes no assumptions about data, it is very flexible and can adapt to complex, nonlinear patterns

Distance computation is the key. Scaling is mandatory

- **Euclidean**: Neighborhoods are cicular. Overall stright line distance is taken
- **Manhattan**: Neighborhoods are diamond shaped. Distance grows by moving along each feature independently (rather than diagonally in space) -- *more intuitive*
- **Cosine**: Only direction, NOT distance
- **Hamming**: Categorical/binary features. It counts how many features differ.

![alt text](https://substackcdn.com/image/fetch/$s_!oqBR!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3ffb7839-3c5b-415b-ba78-488c3bf1cf58_576x559.png)

> Bias Variance fine only: small k --> less neighbors check --> fit to noise --> high variance

### Curse of dimensionality

- Each feature can add its change. kNN adds all differences together
- Total distance between points becomes large - not because they are extremely different in one aspect, but because they are slightly different in many aspects
- *Distance loses its ability to separate good neighbors from bad ones*

> **INTUITION:**In high dimensions, small distances add up, distances become large and similat, and nearest neighbors stop being meaningfully similar

### Weighted kNN (nice!)

- Instead of each point having equal weight, allow closer points more weight. Use *inverse distance weighting*. Helps to resolve ambiguous cases too
- Allows robustness to noisy data. Isolated data points get less weight automatically

### Faster Searches

- **KD trees**: Recursively splits feature space along individual dimensions to make many regions. For inference, first searches across splits to reach a small region
- **Ball**: Groups points into nested balls - each with its own center and radius
- Again, with high dimensions, distances between points become lesser and small differences accummulate across many features. Many regions appear equally close

#### Approximate nearest neighbors

- Relax the pruning rules from exact search. Skip regions when they are *unlikely* to contain a much better neighbor
- *Locality sensitive hashing* - hashes points in a way that makes similar points more likely to collide in the same buckets

> These methods work because in reality similarity is often fuzzy and *almost nearest* is good enough

#### Imbalanced data

- kNN does not inherently handle these well (ofc majority class can bias it)
- Smaller k helps (can cause overfitting)
- Distance weighted kNN is effective
- Modify decision rule too - incorporate class priors

---

## Part 8 kMeans

The averaging step (while finding new centroid locations) is the reason the algorithm is called k-means.

### The TWO Algo steps **NON-CONVEX ALGORITHM**

- Given centers, assign each point to its closest center
- Given the points in the cluster, find the center (mean) of the cluster

> Both steps are minimising the total squared distance

### Objective

- **Inertia**: For each point, how far is tha point from the center of its cluster. Then add up the **squares of distances of all points**.
- Mean is THE POINT that minimises the total squared (Euclidean) distance
- If absolute distance was considered, the Median would have minimised.

> Lower inertia not necessarily the best. Inertia always goes down with more clusters

### Elbow method

Checks inertia vs k. Inertia always goes down with k. \
**INTUITION**: Before the elbow, adding clusters gives large gains in compactness. After the elbow, gains are marginal

#### Silhouette score

Instead of only looking at how close points are to own cluster, it compares the average distance to points in its own cluster with the average distance to points in the nearnest neighboring cluster.

> More balanced view of cluster quality than inertia alone

#### Gap statistic

Compares clustering result with completely random data with no structure. If increasing k isn't improving over random data - then possibly overclustering

> *There is no single best k. Always decide as per business rules + elbow + interpretability*

### Initialisation

- Starting with random points if very close OR starting points are in unrepresentative part of the data

#### kMeans++

- First centroid is chosen randomly
- For each subsequent centroid, points that are far away from existing centroids are given higher chance of being selected. High likelihood that each new centroid starts in a different dense region of the data

> **INTUITION:** k-Means++ tries to cover the space before the algorithm refines clusters. Reducing changes \
> Running kMeans once is rarely enough. Multiple runs and checking stability is standard practice

### Assumptions

- Spherical clusters: **IMPORTANT** Since we take Euclidean distance and summarise each cluster by its centroid
- Equal variance across clusters: Squared distance penalises faraway points. Might split a wide cluster to reduce objective function
- Similar cluster sizes *not strict*: Smaller clusters can get pulled toward larger ones or disappear entirely
- Euclidean feature space: Features should be numeric and comparable with Euclidean distance. Categorical, ordinal with mixed spacing, mixed data types violate. Embeddings help!

### Alternatives

- kMeans always gives results. **When assumption fails, better to use density based, heirarchical clusterings better choice**

### Processing

- Scaling + Normalising MANDATORY
- Outliers + Noise distort clusters
- High dimensional data hard to cluster -- distances between points tend to become similar. (**Distance concentration**). Notion of centroid becomes weak.
- Dimensionality reduction: reduces noise and redundant features

### Training time complexity

- n points, k clusters, d features --> n *k* d operations since compute the distance between every data point and every centroid

### Mini batch k Means for large datasets

- Instead of computing distances to all points at every iteration, update centroids using small random subsets of the data (mini batch)

### Clustering Evaluation

- **Subjective**
- Internal metrics like Inertia + Silouhette score fine
- Business metrics important
- Centroids are cluster centers - not real data points

### VS Other Clustering Algorithms

- **vs Hierarchical Clustering:** Repeatedly mergins smaller clusters together or splitting larger clusters apart - producing dendrograms. Cut this tree at different levels and obtain different number of clusters
- **vs DBSCAN:** *Looks at regions of high point density separated by regions of low density i.e. dense neighbourhoods form clusters*. Performs better where kMeans struggles - elongated clusters/outliers
- **vs Gaussian Mixture Models:** Softer version of kMeans. Assume data is generated from a fixed number of clusters. But each cluster is modelled as a probability distribution

---

## Part 9 SVM

Solving a constrained **CONVEX** optimisation problem that balances two competing goals

1. Keep the decision boundary as wide as possible
2. Allow some classification errors (avoid too large mistakes) when the data is noisy and not perfectly separable

These are contrastive because, wider boundary --leads to--> misclassifying more points

> **INTUITION**: \
> Even for a linear seperation line between two classes, many lines work. Not all seem equally *safe*. SVMs selects the one with max breathing room (margin) i.e. max distance b/w decision boundary and closest points from each class \
> On the margin points matter. If we move one of these close points - boundary shifts - hence called support vectors \
> SVMs care about *which side* of boundary and *how far* from boundary. Aren't trying to predict probabilities \
> SVMs work well with many features - because (out of many possible ways to seperate the data) the margin acts like a built-in preference for simpler, more stable boundaries instead of fragile ones

### Hard vs Soft Margin SVM

- Hard Margin: Cannot find a solution if even one point violates assumption of lying on the correct side of the boundary
- Soft Margin: Allows some points to be on wrong side of margin. **KEY IDEA**: Not all mistakes are equally bad - barely crossing the margin is better than deeply misclassified point. **SLACK VARIABLES**: Measures how badly a point violates the margin. High slack = clear errors

### Primal vs Dual

- Primal: Think wrt decision boundary's coefficients (weights and bias)
- Dual: Boundary is stored as a weighted combination of few critical points (support vectors). Each training point gets a number (its influence). Most points have zero. Only support vectors have non-zero.

### Parameter C

- How strict about misclassifications. How much model hates to misclassify
- Large C: high penalty to any misclassifications. Boundary tries to classify every point correctly - even becoming sensitive to noise -- overfitting
- Small C: allows wider margin - smoother and *stable* boundary -- underfitting

> Bias variance tradeoff!!

### Kernel Trick

- Instead of manual feature engineering, kernels allow to map data into a much higher dimensional space. Here the max margin, linear decision bounday logic applies
- **KEY**: We never actually perform this transformation. In the SVM dual form, model only needs a way to measure similarity b/w data points. **Kernels provide this similarity function. The compute what the dot product looks like in the transformed space - without every constructing the space explicitly**

> Hence kernels are elegant and efficient!

#### Limitations due to Kernels

- Kernels rely on computing similarities between pairs of points - often involving many support vectors. Training + inference time up because each new point must be compared against many support vectors

### RBF Kernel

Polynomial kernels overfit to noise as the degree increases. RBF is most popular. **It measures similarity based on distance**. Close points strongly influence each other - distant points barely interact - leading to smooth and localised boundaries.

> Other kernels include Linear, Polynomial, Sigmoid etc. Since overfitting to noise is common - kernels are almost always discussed with Regulatisation

#### Gamma parameter

> Only useful for a non-linear kernel. Sensitive to feature scaling. **SCALING IS MANDATORY WHEN USING RBF BASED SVMs**

- Low gamma: each point has wide area of influence = smoother, global decision boundaries -- underfitting
- High gamma: local influence = tighter boundaries -- overfitting

> High gamma + high C = overfitting. Low gamma + low C = underfitting

### Feature Scaling

- MANDATORY
- SVMs are based on margins, for nonlinear kernels it is distances between points.
- Super important in interviews - signal of practical experience

### Outliers

- Only Soft margin SVM works since it allows violations - controlled by C. Small C = small hate for misclassification = stable overall boundary
- Less sensitive to outliers - only to support vectors

### High dimensions

- High number of features --> points are more spread out --> *easier* to find a seperating boundary --> SVMs with linear kernels perform well on text classification
- SVMs only use support vectors for decision boundary
- High dimensional data if sparse, easy for SVMs (check how)

### Multiclass SVMs (= combining many Binary SVMs)

- One vs rest: Train one SVM per class. Each learns to separate one vs all other classes combined. *Inference* --> run all models and pick the class whose classifier is most confident
- One vs one: Train one for every pair of classes. *Inference* --> Each models casts a vote, class with most votes wins

### SVR

- Fits a function while ignoring small errors and focussing only on large deviations. Defines a tube around the prediction function and only cares about points outside the tube (controlled by epsilon). Points outside the epsilon affects the solution (support vector). Points within the tube are ignored. C holds same meaning - large C forces model to fit data tightly

### Limitations

- Interpretability
- KERNEL BASED SVMs - slow and memory intensive as the dataset grows
- Heavily dependent on C and gamma and feature scaling

### VS Logistic Regression

- LR: Probabilistic - what prob. of a point belonging to a class. Decision boundary is the byproduct of this probability
- SVMs: Not modelling probabilities at all. Focussed on finding a decision boundary with widest margin. Only cares about which side and how far. **Usually leads to more robust boundaries, especially when classes are well separated.**

### FAQs

- When SVMs over other classification algos: When robust decision boundaries needed - small/medium size of data - high dimensional feature space
- Feature scaling imp? YES. SVMs rely on distances and margins. Scaling ensures all features conrtibute fairly
