# High-Depth ML & Data Science Interview Questions (Math-Focused)


---

## 1. Logistic Regression: Loss Function Derivation

**Question:**
Derive the Binary Cross-Entropy (Log-Loss) function for Logistic Regression using Maximum Likelihood Estimation (MLE). Why do we take the log?

**Answer:**

Assume the target  $y \in {0,1}$  follows a Bernoulli distribution with probability
$$ p = \hat{y} = \sigma(w^T x) $$

The probability mass function is:
$$ P(y \mid x; w) = \hat{y}^y (1 - \hat{y})^{1 - y} $$

For $N$ independent samples, the likelihood is:
$$ L(w) = \prod_{i=1}^{N} (\hat{y}_i)^{y_i} (1 - \hat{y}_i)^{1 - y_i} $$

We take the log to simplify differentiation and avoid numerical underflow:
$$ \ell(w) = \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right] $$

Negating and averaging gives the Binary Cross-Entropy loss:
$$ J(w) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right] $$

---

## 2. Random Forest: Feature Importance Mathematics

**Question:**
How does a Random Forest compute Gini Importance (Mean Decrease Impurity) for a feature?

**Answer:**

The Gini Impurity of a node $t$ with class probabilities $p_i$ is:
$$ G(t) = 1 - \sum_{i=1}^{C} p_i^2 $$

If node $t$ is split into left and right children using feature $X_j$, the impurity decrease is:
$$ \Delta G(t, X_j) = G(t) - \left( \frac{N_{t_L}}{N_t} G(t_L) + \frac{N_{t_R}}{N_t} G(t_R) \right) $$

The importance of feature $X_j$ is the sum of $\Delta G$ over all splits where it is used, averaged across all trees.

---

## 3. XGBoost: Objective Function and Taylor Expansion

**Question:**
How does XGBoost differ from standard Gradient Boosting in its objective formulation?

**Answer:**

Standard Gradient Boosting uses first-order gradients. XGBoost uses a second-order Taylor approximation.

At iteration $t$, the objective is approximated as:
$$ \mathcal{L}^{(t)} \approx \sum_{i=1}^{N} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t) $$

Where:

* $ g_i = \frac{\partial l}{\partial \hat{y}} $
* $ h_i = \frac{\partial^2 l}{\partial \hat{y}^2} $
* $ \Omega(f_t) $ is a regularization term

Second-order information enables faster convergence and more accurate optimization.

---

## 4. Attention Mechanism: Scaling Factor

**Question:**
Why is self-attention scaled by $ \sqrt{d_k} $ in Transformers?

**Answer:**

If $ Q $ and $ K $ have zero mean and unit variance, then:
$$ \text{Var}(QK^T) = d_k $$

Large dot products cause the Softmax to saturate, leading to near-zero gradients. Dividing by $ \sqrt{d_k} $ normalizes the variance, keeping Softmax in its sensitive region.

---

## 5. SVM: Primal vs Dual Formulation

**Question:**
Explain the objective of a Hard Margin SVM and the motivation for the dual form.

**Answer:**

Primal problem:
$$ \min_{w,b} \frac{1}{2} |w|^2 $$
subject to:
$$ y_i (w^T x_i + b) \geq 1 \quad \forall i $$

The dual formulation:

* Depends only on dot products $ x_i^T x_j $ 
* Enables the kernel trick
* Is computationally efficient in high dimensions

---

## 6. PCA: Eigenvalues and Variance

**Question:**
Why do eigenvectors of the covariance matrix represent directions of maximum variance?

**Answer:**

Variance along direction $ u $ is:
$$ \text{Var}(Xu) = u^T \Sigma u $$

Maximize subject to $ u^T u = 1 $. Using Lagrange multipliers:
$$ \Sigma u = \lambda u $$

Thus, eigenvectors define principal directions, and eigenvalues represent variance magnitude.

---

## 7. Regularization: Bayesian Interpretation

**Question:**
Show that L1 corresponds to a Laplace prior and L2 to a Gaussian prior.

**Answer:**

Posterior:
$$ \log P(w \mid D) = \log P(D \mid w) + \log P(w) $$

* L2: $ w \sim \mathcal{N}(0, \tau^2) \Rightarrow \log P(w) \propto -|w|^2 $
* L1: $ w \sim \text{Laplace}(0, b) \Rightarrow \log P(w) \propto -|w|_1 $

L1 encourages sparsity due to a sharp peak at zero.

---

## 8. Backpropagation: Vanishing Gradient

**Question:**
Why does Sigmoid cause vanishing gradients?

**Answer:**

Sigmoid derivative:
$$ \sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 0.25 $$

In a deep network, gradients scale as:
$$ (0.25)^L $$

This exponential decay causes early layers to stop learning.

---

## 9. Naive Bayes: Independence Assumption

**Question:**
Where is the naive assumption applied in Naive Bayes?

**Answer:**

Bayes theorem:
$$ P(y \mid x_1, \dots, x_n) \propto P(x_1, \dots, x_n \mid y) P(y) $$

Naive assumption:
$$ P(x_1, \dots, x_n \mid y) \approx \prod_{i=1}^{n} P(x_i \mid y) $$

Final rule:
$$ \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y) $$

---

## 10. K-Means: Convergence

**Question:**
What is the K-Means objective and does it guarantee a global optimum?

**Answer:**

Objective:
$$ J = \sum_{j=1}^{K} \sum_{x_i \in C_j} |x_i - \mu_j|^2 $$

K-Means alternates between:

* Assignment step
* Centroid update step

The objective decreases monotonically and is bounded below, guaranteeing convergence to a local optimum, not the global one.

---

## 11. Adam Optimizer: Bias Correction

**Question:** In the Adam optimizer, we calculate moving averages of gradients ($m_t$) and squared gradients ($v_t$). Why do we perform "bias correction" steps $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$?

**Answer:**
We initialize the moving average vectors $m_0$ and $v_0$ as vectors of **zeros**.
If the decay rates $\beta_1$ and $\beta_2$ are close to 1 (e.g., typical 0.9 and 0.999), the moving averages $m_t$ and $v_t$ are heavily **biased towards zero**, especially during the initial time steps.
Mathematically, the expected value $E[m_t]$ corresponds to $E[g_t] \cdot (1 - \beta_1^t)$. By dividing by $(1-\beta_1^t)$, we unbias the estimate so that the expected value matches the true first moment of the gradient, ensuring the optimizer doesn't take tiny steps at the start of training.

## 12. Bias-Variance Decomposition: The Math

**Question:** Derive the decomposition of the Mean Squared Error (MSE) for an estimator $\hat{f}(x)$ into Bias, Variance, and Irreducible Error.

**Answer:**
Let $y = f(x) + \epsilon$, where $\epsilon$ has mean 0 and variance $\sigma^2$.
The expected MSE at a point $x$ is $E[(y - \hat{f}(x))^2]$.
By adding and subtracting $E[\hat{f}(x)]$ (the average prediction over many dataset realizations):
 $$ E[(y - \hat{f})^2] = E[(y - f + f - E[\hat{f}] + E[\hat{f}] - \hat{f})^2]  $$
Expanding the square and using the independence of $\epsilon$, cross-terms cancel out, leaving:
 $$ \text{MSE} = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f}(x) - E[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}  $$
This proves that reducing error requires balancing the model's ability to fit data (Bias) vs. stability across datasets (Variance).

## 13. AdaBoost: Exponential Loss & Update Rule

**Question:** Why does AdaBoost use the Exponential Loss function $L(y, f(x)) = e^{-y f(x)}$? How does this choice lead to the specific weight update formula $\alpha_t = \frac{1}{2} \ln(\frac{1-\epsilon_t}{\epsilon_t})$?

**Answer:**
AdaBoost greedily minimizes the Exponential Loss. The loss is differentiable and acts as an upper bound to the 0-1 loss (classification error).
When we solve for the optimal weight $\alpha_t$ for the new weak learner $h_t$ that minimizes this loss:
 $$ \frac{\partial L}{\partial \alpha_t} = 0 \implies \sum_{y_i = h_t(x_i)} w_i e^{-\alpha} - \sum_{y_i \neq h_t(x_i)} w_i e^{\alpha} = 0  $$
Let $\epsilon_t$ be the weighted error rate. Solving for $\alpha$ gives the exact closed-form solution:
 $$ \alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)  $$
This specific update ensures that the weights of misclassified points are increased exactly enough so that the new weak learner would have 50% accuracy on the _re-weighted_ data.

## 14. Batch Normalization: Backpropagation

**Question:** During backpropagation through a Batch Normalization layer, why do we need to calculate gradients w.r.t the batch mean $\mu$ and batch variance $\sigma^2$? What happens if we ignore them?

**Answer:**
In Batch Norm, the output $y_i$ depends on input $x_i$ **directly** (via normalization) and **indirectly** (because $x_i$ contributes to $\mu$ and $\sigma^2$ of the batch).
 $$ \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}  $$
The chain rule must account for all paths:
 $$ \frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \left( \frac{\partial \hat{x}_j}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial \hat{x}_j}{\partial \sigma_B^2} \frac{\partial \sigma_B^2}{\partial x_i} \right)  $$
If you ignore the terms involving $\mu$ and $\sigma$, the network "cheats" by manipulating the batch statistics to minimize loss without learning useful features, leading to training instability or collapse.

## 15. CNN: Max Pooling Gradient

**Question:** Max Pooling layers have no learnable parameters. How is the gradient calculated for them during backpropagation?

**Answer:**
The gradient flows only through the **maximum** value.
During the forward pass, we store the **indices** (locations) of the maximum values in each pooling window.
During the backward pass (Upsampling/Unpooling):

1. The gradient from the next layer is passed **only** to the neuron at the stored index (the "winner").
2. All non-max neurons in that window receive a gradient of **0**.
Mathematically, this acts as a "router" for gradients, directing the learning signal only to the features that were most salient (max).

## 16. Dropout: Scaling Factor

**Question:** In "Inverted Dropout", why do we scale the activations by $\frac{1}{1-p}$ during _training_?

**Answer:**
We want the expected value of the neuron's output to remain the same during training and testing.
During training, a neuron is kept with probability $(1-p)$. Thus, the expected output is $E[x_{train}] = (1-p) x$.
During testing, we use all neurons ($p=0$), so the output is $x$.
To match expectations ($E[x_{train}] \approx x_{test}$), we scale training outputs by $\frac{1}{1-p}$.
This ensures we don't need to modify the weights at test time, making deployment easier and statistically consistent.

## 17. AUC-ROC: Probabilistic Interpretation

**Question:** Prove or explain why the Area Under the ROC Curve (AUC) is equivalent to the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

**Answer:**
The AUC calculates the integral of TPR (True Positive Rate) vs. FPR (False Positive Rate).
Consider a dataset with $N^+$ positive and $N^-$ negative samples.
We can view the AUC as counting pairs:
 $$ \text{AUC} = \frac{1}{N^+ N^-} \sum_{i=1}^{N^+} \sum_{j=1}^{N^-} \mathbb{I}(Score(x_i^+) > Score(x_j^-))  $$
This sum represents the normalized count of all pairs where the classifier correctly assigns a higher probability to the positive sample than the negative one. This is exactly the probability $P(Score(x^+) > Score(x^-))$ (equivalent to the **Mann-Whitney U Test** statistic).

## 18. GMM vs K-Means: Soft Assignment

**Question:** In Gaussian Mixture Models (EM Algorithm), how does the E-step differ mathematically from the assignment step in K-Means?

**Answer:**
K-Means performs **Hard Assignment**: A point $x_i$ belongs to cluster $k$ with probability 1 or 0 (indicator function).
GMM performs **Soft Assignment** by calculating "Responsibility" $\gamma(z_{nk})$:
 $$ \gamma(z_{nk}) = P(z_k=1 | x_n) = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}  $$
This is the posterior probability that point $x_n$ came from component $k$. This allows GMM to model uncertainty and overlapping clusters, whereas K-Means assumes spherical, non-overlapping clusters.

## 19. Positional Encodings: Linear Property

**Question:** In Transformers, we use Sine/Cosine positional encodings. What specific mathematical property allows the model to learn to attend by _relative_ positions?

**Answer:**
We use sinusoidal functions of different frequencies: $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$.
The key property is that for any fixed offset $k$, $PE(pos+k)$ can be represented as a **linear function** of $PE(pos)$.
Using the rotation formula $\sin(A+B) = \sin A \cos B + \cos A \sin B$:
 $$ \begin{bmatrix} \sin(pos+k) \\ \cos(pos+k) \end{bmatrix} = \begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}  $$
This rotation matrix depends only on $k$, allowing the model's self-attention weights (which are linear projections) to easily learn patterns based on relative distance $k$ regardless of absolute position $pos$.

## 20. Loss Functions: MSE for Classification

**Question:** Why is Mean Squared Error (MSE) generally a bad choice for binary classification (Logistic Regression) compared to Cross-Entropy? Give two mathematical reasons.

**Answer:**

1. **Non-Convexity:** When combining the Sigmoid activation $\sigma(z)$ with MSE, the resulting cost function $J(w)$ is **non-convex** with respect to weights $w$. This creates many local minima, making Gradient Descent get stuck easily. Cross-Entropy is strictly convex.
2. **Vanishing Gradients:**
    * MSE Gradient: $\partial J \propto (\hat{y}-y) \cdot \sigma'(z)$. If the prediction is completely wrong (e.g., $y=1, \hat{y} \approx 0$), the gradient $\sigma'(z) \approx 0$, causing learning to stall.
    * Cross-Entropy Gradient: $\partial J \propto (\hat{y}-y)$. The derivative $\sigma'(z)$ cancels out. If the prediction is wrong, the gradient is large ($|\hat{y}-y| \approx 1$), leading to fast correction.

----

## 21. ResNet: Gradient Flow

**Question:** Mathematically, how do Skip Connections (Identity Mappings) in ResNets specifically solve the vanishing gradient problem?

**Answer:**
Let a residual block be defined as $H(x) = F(x) + x$, where $F(x)$ is the learnable non-linear function.
During backpropagation, the gradient of the loss $L$ with respect to input $x$ is:
$$  \frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial L}{\partial H} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)  $$
$$  \frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} \cdot \frac{\partial F}{\partial x} + \frac{\partial L}{\partial H}  $$
The term $+ \frac{\partial L}{\partial H}$ acts as a "gradient superhighway," ensuring that the gradient can flow directly from deeper layers to shallower layers without being diminished (multiplied by small numbers), even if $\frac{\partial F}{\partial x}$ approaches zero.

### 22. Linear Regression: Invertibility

**Question:** In the Normal Equation $\theta = (X^T X)^{-1} X^T y$, under what specific mathematical condition is $(X^T X)$ **not** invertible, and what does this imply about your data?

**Answer:**
The matrix $(X^T X)$ is not invertible (singular) if it is **rank-deficient**.
This occurs if the columns of $X$ are **linearly dependent** (Multicollinearity).
Mathematically, this means one feature is a perfect linear combination of other features (e.g., Feature A = 2 * Feature B).
In this case, the determinant $|X^T X| = 0$, and there are infinitely many solutions for $\theta$ that minimize the cost. To fix this, we typically add Regularization (Ridge), which adds $\lambda I$ to $X^T X$, making it full-rank and invertible.

### 23. Weight Initialization: Xavier/Glorot

**Question:** Why does Xavier Initialization require the variance of weights to be $Var(W) = \frac{1}{n_{in}}$ (or $\frac{2}{n_{in}+n_{out}}$)? What mathematical property are we trying to preserve?

**Answer:**
We want to preserve the **variance of activations** across layers to prevent signals from exploding or vanishing.
For a linear neuron $y = \sum w_i x_i$:
$$  Var(y) = Var(\sum w_i x_i)  $$
Assuming inputs $x$ and weights $w$ are independent and have mean 0:
$$  Var(y) = n_{in} \cdot Var(w_i) \cdot Var(x_i)  $$
To keep the signal strength constant ($Var(y) = Var(x)$), we must enforce:
$$  n_{in} \cdot Var(w_i) = 1 \implies Var(w_i) = \frac{1}{n_{in}}  $$
If $Var(w)$ is too small, activations shrink to 0; if too large, they explode to infinity (or saturate activations).

### 24. t-SNE: KL Divergence Asymmetry

**Question:** The t-SNE objective function minimizes the Kullback-Leibler (KL) Divergence between high-dimensional probabilities $P$ and low-dimensional probabilities $Q$. Why is $KL(P||Q)$ used instead of $KL(Q||P)$?

**Answer:**
$KL(P||Q) = \sum p_i \log(p_i / q_i)$.

* **KL(P||Q)** penalizes cases where $p_i$ is high (points are close in high-dim) but $q_i$ is low (points are far in low-dim). This forces the algorithm to **preserve local structure** (keep neighbors together).
* **KL(Q||P)** would penalize cases where $q_i$ is high but $p_i$ is low. This would force far-away points to remain far, but wouldn't care if neighbors were separated.
t-SNE prioritizes keeping similar points together, hence $KL(P||Q)$.

### 25. 1x1 Convolution: Mathematical Purpose

**Question:** How exactly does a 1x1 Convolution reduce dimensionality? Isn't it just multiplying by a scalar?

**Answer:**
It is **not** just a scalar multiplication if the input has multiple channels.
If the input tensor is $H \times W \times C_{in}$ and we apply $C_{out}$ filters of size $1 \times 1 \times C_{in}$:
Each $1 \times 1$ filter performs a **linear combination** across the depth (channels) for every pixel position $(i, j)$:
$$  \text{Output}_{i,j,k} = \sum_{c=1}^{C_{in}} w_{k,c} \cdot \text{Input}_{i,j,c} + b_k  $$
If $C_{out} < C_{in}$, this effectively projects the high-dimensional feature vector at each pixel into a lower-dimensional space, reducing computational cost for subsequent layers (like in Inception/GoogLeNet).

### 26. Softmax: Shift Invariance

**Question:** Prove that the Softmax function is invariant to constant shifts in the input. Why is this property useful for numerical stability?

**Answer:**
Let $z$ be the input vector and $c$ be a scalar constant.
$$  \text{Softmax}(z+c)_i = \frac{e^{z_i + c}}{\sum_j e^{z_j + c}} = \frac{e^{z_i} e^c}{\sum_j e^{z_j} e^c}  $$
The $e^c$ term factors out of the sum in the denominator and cancels with the numerator:
$$  = \frac{e^{z_i} e^c}{e^c \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} = \text{Softmax}(z)_i  $$
**Usefulness:** In implementation, we calculate $\text{Softmax}(z - \max(z))$. By subtracting the maximum value, all exponents become $\le 0$. This prevents floating-point **overflow** (calculating $e^{1000}$), which would otherwise result in NaNs.

### 27. EM Algorithm: Jensen's Inequality

**Question:** In the Expectation-Maximization (EM) algorithm, we maximize a lower bound of the Log-Likelihood. How is this lower bound derived using Jensen's Inequality?

**Answer:**
We want to maximize $l(\theta) = \log P(X|\theta) = \log \sum_Z P(X, Z|\theta)$.
Direct maximization is hard due to the sum inside the log. We introduce a distribution $Q(Z)$ (posterior approximation).
$$  \log \sum_Z P(X, Z|\theta) = \log \sum_Z Q(Z) \frac{P(X, Z|\theta)}{Q(Z)} = \log E_Q \left[ \frac{P(X, Z|\theta)}{Q(Z)} \right]  $$
Since $\log$ is a **concave** function, Jensen's Inequality states $\log E[Y] \ge E[\log Y]$.
$$  \log E_Q \left[ \dots \right] \ge E_Q \left[ \log \frac{P(X, Z|\theta)}{Q(Z)} \right]  $$
This term (the **ELBO**) is easier to maximize. The E-step aligns the bound with the likelihood, and the M-step maximizes the bound.

### 28. LSTM: Gating Derivative

**Question:** Why do LSTMs use the `Sigmoid` function for gates (Forget, Input, Output) but `Tanh` for the cell state update?

**Answer:**

* **Sigmoid for Gates:** Gates are switches. We need values strictly between **0 (closed)** and **1 (open)**. Sigmoid outputs $(0, 1)$, making it probabilistically interpretable as "how much information to let through."
* **Tanh for State Update:** The cell state needs to increase _or_ decrease. Tanh outputs $(-1, 1)$, allowing the network to **add or subtract** information from the cell state.
* _Why not Sigmoid for state?_ If we used Sigmoid (0, 1) for updates, the cell state would only grow positively over time, leading to potential explosion or bias. Tanh keeps the state centered around 0.

### 29. Momentum: Velocity Update

**Question:** Write the mathematical update rule for Gradient Descent with Momentum. How does the "velocity" term physically help convergence?

**Answer:**
Standard SGD update: $\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$.
Momentum update:

1. $v_{t+1} = \gamma v_t + \alpha \nabla J(\theta_t)$
2. $\theta_{t+1} = \theta_t - v_{t+1}$
Where $\gamma$ (usually 0.9) is the momentum coefficient (friction).
**Physical Intuition:** The velocity $v_t$ accumulates gradients over time.

* In "valleys" (oscillating gradients), the alternating signs (+/-) cancel out in the sum $v_t$, reducing oscillation.
* In "flat" directions (consistent small gradients), the terms add up, accelerating speed towards the minimum.

### 30. F1 Score: Harmonic Mean

**Question:** Why is the F1 Score defined as the Harmonic Mean of Precision and Recall, rather than the Arithmetic Mean?

**Answer:**
$$  F1 = \frac{2}{\frac{1}{P} + \frac{1}{R}} = \frac{2PR}{P+R}  $$
The Harmonic Mean is dominated by the **minimum** of the two numbers.

* If Precision = 1.0 and Recall = 0.0 (e.g., predicting nothing):
  * Arithmetic Mean = 0.5 (Misleadingly decent).
  * Harmonic Mean (F1) = 0 (Correctly indicates total failure).
This property forces the model to balance **both** metrics simultaneously. You cannot "cheat" the metric by maximizing one at the total expense of the other.
