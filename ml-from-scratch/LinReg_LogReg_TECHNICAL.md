<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Linear and Logistic Regression: Comprehensive Technical Analysis

This comprehensive report provides a detailed mathematical treatment of linear and logistic regression, including regularization techniques, optimization methods, and complete pure Python implementations.

## Introduction to Regression Models

Regression analysis forms the cornerstone of supervised machine learning, providing a framework for modeling relationships between input features and output variables. Linear regression addresses continuous prediction tasks, while logistic regression handles binary classification problems. Both models share fundamental optimization principles but differ significantly in their probabilistic interpretations and loss functions.

The power of these models lies not only in their predictive capabilities but also in their mathematical tractability and interpretability. Linear regression serves as the foundation for understanding more complex models, while logistic regression introduces concepts crucial for neural networks and deep learning.

## Linear Regression: Mathematical Foundations

### Model Specification and Assumptions

Linear regression models the relationship between a dependent variable $y$ and independent variables $\mathbf{x}$ through a linear function. For a dataset with $m$ samples and $n$ features, the model is expressed as:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_n x_{in} + \varepsilon_i
$$

> **Feature vector for observation $i$: written as COLUMN VECTORS** *(stupid onlyy)*
$$
\mathbf{x}_i = \begin{bmatrix} 1 \\ x_{i1} \\ x_{i2} \\ \vdots \\ x_{in} \end{bmatrix}

~ \text{and} ~

\mathbf{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n \end{bmatrix}
$$

The same equation can be written as an inner product:

$$ y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \varepsilon_i$$

Stack all $y_i$ values into a vector:

$$ 
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix}
$$

Stack all feature vectors $ \mathbf{x}_i^T $ as rows of a matrix:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_m^T \end{bmatrix}
$$

In matrix notation, this becomes:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where $\mathbf{y} \in \mathbb{R}^m$ is the target vector, $\mathbf{X} \in \mathbb{R}^{m \times n}$ is the design matrix, $\boldsymbol{\beta} \in \mathbb{R}^n$ represents the coefficients, and $\boldsymbol{\varepsilon}$ captures the error term.

The classical assumptions underlying linear regression include: 

- (1) linearity between features and target
- (2) independence of observations
- (3) homoscedasticity (constant variance of errors)
- (4) normality of error distribution
- (5) absence of multicollinearity among features

These assumptions, while often violated in practice, provide the theoretical foundation for inference and prediction.

### Ordinary Least Squares Derivation

The Ordinary Least Squares (OLS) method estimates parameters by minimizing the residual sum of squares (RSS). The objective function is:

$$
\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^m (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2
$$

Taking the derivative with respect to $\boldsymbol{\beta}$ and setting it to zero yields the normal equations:

$$
\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0
$$

$$
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
$$

When $\mathbf{X}^T\mathbf{X}$ is invertible, the closed-form solution is:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This elegant solution represents the **best linear unbiased estimator (BLUE)** under the Gauss-Markov theorem, meaning it has the minimum variance among all linear unbiased estimators.

For simple linear regression with a single predictor, the slope and intercept have explicit formulas:

$$
\hat{\beta}_1 = \frac{s_{xy}}{s_x^2} = \frac{\sum_{i=1}^m (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^m (x_i - \bar{x})^2}
$$

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}
$$

where $s_{xy}$ is the sample covariance and $s_x^2$ is the sample variance of $x$.

### Probabilistic Interpretation and Maximum Likelihood

Linear regression can be derived from a probabilistic perspective by assuming the errors follow a Gaussian distribution. Specifically, we assume:

$$
y_i = \mathbf{x}_i^T\boldsymbol{\beta} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

This implies that $y_i | \mathbf{x}_i; \boldsymbol{\beta} \sim \mathcal{N}(\mathbf{x}_i^T\boldsymbol{\beta}, \sigma^2)$. The likelihood of observing the data given parameters $\boldsymbol{\beta}$ is:

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2}{2\sigma^2}\right)
$$

Taking the logarithm yields the log-likelihood:

$$
\log L(\boldsymbol{\beta}) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^m (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2
$$

Maximizing this log-likelihood with respect to $\boldsymbol{\beta}$ is equivalent to minimizing the sum of squared errors, thus **maximum likelihood estimation (MLE) under Gaussian noise yields the OLS solution**. This connection reveals that choosing squared error as the loss function implicitly assumes Gaussian-distributed noise. Alternative noise distributions (e.g., Laplacian) would lead to different loss functions, such as absolute error.

### Gradient Descent Optimization

While OLS provides a closed-form solution, gradient descent offers an iterative alternative that scales better for large datasets and naturally extends to more complex models. The algorithm updates parameters in the direction of steepest descent of the cost function:

$$
\text{Cost}(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^m (h_{\boldsymbol{\beta}}(\mathbf{x}_i) - y_i)^2
$$

where $h_{\boldsymbol{\beta}}(\mathbf{x}) = \mathbf{x}^T\boldsymbol{\beta}$ is the hypothesis function. The gradient is:

$$
\nabla_{\boldsymbol{\beta}} \text{Cost} = \frac{1}{m}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta} - \mathbf{y})
$$

The update rule at iteration $t$ is:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \nabla_{\boldsymbol{\beta}} \text{Cost}
$$

where $\alpha$ is the **learning rate**, a critical hyperparameter that controls the step size. If $\alpha$ is too small, convergence is slow; if too large, the algorithm may diverge or oscillate. For linear regression with a convex loss surface, gradient descent is guaranteed to converge to the global minimum given an appropriate learning rate.

The convergence rate of gradient descent depends on the condition number of $\mathbf{X}^T\mathbf{X}$. Poor conditioning (high condition number) leads to slow convergence, as the loss function forms elongated elliptical contours. This motivates the use of feature scaling or normalization to improve convergence speed.

## Regularization: Controlling Model Complexity

### The Bias-Variance Tradeoff

Machine learning models face a fundamental tradeoff between bias and variance. **Bias** measures the error from overly simplistic model assumptions, while **variance** captures the model's sensitivity to fluctuations in the training data. The expected prediction error decomposes as:

$$
\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2[\hat{f}(\mathbf{x})] + \text{Var}[\hat{f}(\mathbf{x})] + \sigma^2
$$

where $\sigma^2$ is the irreducible error from noise. **Underfitting** (high bias, low variance) occurs when the model is too simple to capture data patterns, while **overfitting** (low bias, high variance) happens when the model fits noise rather than underlying trends.

Regularization techniques address this tradeoff by adding a penalty term to the loss function, deliberately introducing bias to reduce variance and improve generalization. The optimal regularization strength balances these competing objectives, typically selected through cross-validation.

### Ridge Regression (L2 Regularization)

Ridge regression adds a penalty proportional to the squared magnitude of coefficients to the OLS objective:

$$
\text{Cost}_{\text{Ridge}}(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^m (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \frac{\lambda}{2}\sum_{j=1}^n \beta_j^2
$$

where $\lambda \geq 0$ is the regularization parameter. The closed-form solution is:

$$
\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$

The penalty term $\lambda\mathbf{I}$ improves the conditioning of $\mathbf{X}^T\mathbf{X}$, ensuring invertibility even when $\mathbf{X}^T\mathbf{X}$ is singular or near-singular. This makes ridge regression particularly effective for handling **multicollinearity**, where predictor variables are highly correlated.

![The sigmoid function transforms linear combinations of features into probabilities between 0 and 1, forming the basis of logistic regression.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/f30b4774216d0c09e80fe970f6d968eb/94c45171-6790-4dfd-9eb8-36288241d11c/e31a656f.png)

The sigmoid function transforms linear combinations of features into probabilities between 0 and 1, forming the basis of logistic regression.

**Why Ridge Shrinks Coefficients to Zero:** The L2 penalty induces a gradient proportional to the coefficient value: $\frac{\partial}{\partial \beta_j}(\lambda\beta_j^2) = 2\lambda\beta_j$. This creates a **proportional shrinkage** effect—larger coefficients receive stronger penalties, but coefficients near zero experience minimal shrinkage. Consequently, ridge regression shrinks coefficients toward (but never exactly to) zero. Geometrically, the constraint $\sum_{j=1}^n \beta_j^2 \leq t$ defines a hypersphere in parameter space. The optimal solution occurs where the loss function contours first touch this spherical constraint region, rarely occurring exactly at zero.

Ridge regression also addresses multicollinearity through the **variance inflation factor (VIF)**. In the presence of high correlation among predictors, OLS estimates have inflated variance: $\text{Var}(\hat{\beta}_j) \propto 1/(1-R_j^2)$, where $R_j^2$ is the coefficient of determination when regressing predictor $j$ on all others. Ridge regularization reduces these variances by stabilizing coefficient estimates.

### Lasso Regression (L1 Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) regression uses an L1 penalty based on the absolute values of coefficients:

$$
\text{Cost}_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^m (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^n |\beta_j|
$$

Unlike ridge regression, lasso has no closed-form solution due to the non-differentiability of the absolute value function at zero. Optimization typically employs coordinate descent or proximal gradient methods.

**Why Lasso Produces Sparse Solutions:** The L1 penalty gradient is constant: $\frac{\partial}{\partial \beta_j}(\lambda|\beta_j|) = \lambda \cdot \text{sign}(\beta_j)$. This **constant shrinkage** applies uniformly regardless of coefficient magnitude, pushing small coefficients exactly to zero. Geometrically, the constraint $\sum_{j=1}^n |\beta_j| \leq t$ forms a rhombus (in 2D) or hyperdiamond (in higher dimensions) with sharp corners on the coordinate axes. Loss function contours are more likely to intersect these corners where some coordinates equal zero, yielding sparse solutions that perform automatic feature selection.

![Geometric interpretation of L1 and L2 regularization showing why L1 leads to sparse solutions (coefficients at zero) while L2 only shrinks coefficients.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/f30b4774216d0c09e80fe970f6d968eb/3a180afa-f6a1-4873-b140-51a6744f355f/9331d380.png)

Geometric interpretation of L1 and L2 regularization showing why L1 leads to sparse solutions (coefficients at zero) while L2 only shrinks coefficients.

The soft-thresholding operator implements lasso updates in coordinate descent:

$$
\beta_j = \begin{cases} \rho - \lambda & \text{if } \rho > \lambda \\ 0 & \text{if } |\rho| \leq \lambda \\ \rho + \lambda & \text{if } \rho < -\lambda \end{cases}
$$

where $\rho$ is the partial residual correlation. This explicitly sets coefficients to zero when their contribution to reducing loss is insufficient to overcome the penalty.

### Elastic Net: Combining L1 and L2

Elastic Net regularization combines L1 and L2 penalties to leverage advantages of both approaches:

$$
\text{Cost}_{\text{EN}}(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^m (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\left[\alpha\sum_{j=1}^n |\beta_j| + \frac{1-\alpha}{2}\sum_{j=1}^n \beta_j^2\right]
$$

where $\alpha \in [0, 1]$ controls the mixing ratio. When $\alpha = 1$, elastic net reduces to lasso; when $\alpha = 0$, it becomes ridge regression.

Elastic net addresses limitations of both methods. Unlike lasso, which arbitrarily selects one variable from a group of correlated predictors, elastic net tends to include or exclude correlated variables together. Unlike ridge regression, elastic net can still perform feature selection through sparsity. However, elastic net introduces an additional hyperparameter requiring careful tuning, increasing computational complexity.

## Logistic Regression: Mathematical Foundations

### Model Specification and the Sigmoid Function

Logistic regression extends linear models to binary classification by modeling the probability that an observation belongs to a particular class. The model applies the sigmoid (logistic) function to a linear combination of features:

$$
P(y=1|\mathbf{x}; \boldsymbol{\beta}) = \sigma(\mathbf{x}^T\boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}}
$$

where $\sigma(z)$ is the sigmoid function. The sigmoid maps real-valued inputs to the interval $(0, 1)$, making it suitable for probability estimation. Key properties include:

1. **Range:** $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$
2. **Symmetry:** $\sigma(-z) = 1 - \sigma(z)$
3. **Derivative:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
4. **Boundary behavior:** $\lim_{z \to \infty} \sigma(z) = 1$, $\lim_{z \to -\infty} \sigma(z) = 0$

The derivative property is particularly important for optimization, as it simplifies gradient computations.

### Log-Odds and the Linear Decision Boundary

The logistic model can be expressed through the log-odds (logit) transformation:

$$
\log\left(\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})}\right) = \log\left(\frac{P(y=1|\mathbf{x})}{1 - P(y=1|\mathbf{x})}\right) = \mathbf{x}^T\boldsymbol{\beta}
$$

This reveals that logistic regression models the log-odds as a linear function of features. The **decision boundary** occurs where $P(y=1|\mathbf{x}) = 0.5$, which corresponds to $\mathbf{x}^T\boldsymbol{\beta} = 0$. This defines a hyperplane in feature space:

$$
\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n = 0
$$

> Logistic Regression doesn't assume a linear relationship between predictors and the target variable itself. Instead, it *assumes a linear relationship between the independent variables and the log-odds of the dependent variable*.

For two features, this is a line; for three features, a plane; and for $n$ features, an $(n-1)$-dimensional hyperplane. **Logistic regression is thus a linear classifier**, despite the nonlinear sigmoid transformation. Points on one side of the hyperplane are classified as class 1, while points on the other side are classified as class 0.

![The decision boundary in logistic regression is a hyperplane that separates the feature space into regions where P(y=1|x) > 0.5 and P(y=1|x) < 0.5.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/f30b4774216d0c09e80fe970f6d968eb/dc9237c7-4093-4094-84c5-3ce9f01dcd7a/2c5673b4.png)

The decision boundary in logistic regression is a hyperplane that separates the feature space into regions where P(y=1|x) > 0.5 and P(y=1|x) < 0.5.

Changing the classification threshold $T$ shifts the decision boundary. If $T < 0.5$, the model becomes more liberal in predicting class 1; if $T > 0.5$, more conservative. The decision boundary equation becomes $\mathbf{x}^T\boldsymbol{\beta} = \log(T/(1-T))$.

### Maximum Likelihood Estimation

Unlike linear regression, logistic regression has no closed-form solution. Parameters are estimated using **maximum likelihood estimation (MLE)**. For binary outcomes $y_i \in \{0, 1\}$, the likelihood function is:

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^m P(y=y_i|\mathbf{x}_i; \boldsymbol{\beta}) = \prod_{i=1}^m \sigma(\mathbf{x}_i^T\boldsymbol{\beta})^{y_i}(1 - \sigma(\mathbf{x}_i^T\boldsymbol{\beta}))^{1-y_i}
$$

The log-likelihood simplifies to:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^m \left[y_i \log(\sigma(\mathbf{x}_i^T\boldsymbol{\beta})) + (1-y_i)\log(1 - \sigma(\mathbf{x}_i^T\boldsymbol{\beta}))\right]
$$

Maximizing log-likelihood is equivalent to minimizing the negative log-likelihood, which becomes the **cross-entropy loss function**.

### Cross-Entropy Loss and its Derivation

The cross-entropy loss measures the divergence between the predicted probability distribution and the true distribution. For logistic regression, the loss for a single sample is:

$$
\mathcal{L}(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

where $\hat{y} = \sigma(\mathbf{x}^T\boldsymbol{\beta})$ is the predicted probability. This loss function penalizes confident wrong predictions heavily while rewarding correct predictions. When $y = 1$, the loss is $-\log(\hat{y})$, which approaches infinity as $\hat{y} \to 0$; when $y = 0$, the loss is $-\log(1-\hat{y})$, which approaches infinity as $\hat{y} \to 1$.

The average cross-entropy over all samples is:

$$
J(\boldsymbol{\beta}) = -\frac{1}{m}\sum_{i=1}^m [y_i\log(h_i) + (1-y_i)\log(1-h_i)]
$$

where $h_i = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$. Unlike mean squared error, cross-entropy produces a convex loss surface for logistic regression, ensuring a unique global minimum.

**Information-Theoretic Interpretation:** Cross-entropy originates from information theory as a measure of the average number of bits needed to encode events from one distribution using an optimal code for another distribution. Minimizing cross-entropy is equivalent to minimizing the Kullback-Leibler (KL) divergence between the true and predicted distributions.

### Gradient Computation

The gradient of the cross-entropy loss with respect to parameters has an elegant form due to the sigmoid derivative property:

$$
\frac{\partial J}{\partial \boldsymbol{\beta}} = \frac{1}{m}\mathbf{X}^T(\mathbf{h} - \mathbf{y})
$$

where $\mathbf{h} = \sigma(\mathbf{X}\boldsymbol{\beta})$ is the vector of predicted probabilities. This remarkably simple gradient—identical in form to linear regression despite the nonlinear model—results from the derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ canceling terms in the cross-entropy gradient. The update rule for gradient ascent (maximizing likelihood) is:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \alpha \frac{1}{m}\mathbf{X}^T(\mathbf{y} - \mathbf{h})
$$

For gradient descent (minimizing negative log-likelihood), the sign flips.

### Newton's Method and Second-Order Optimization

While first-order methods like gradient descent are effective, **Newton's method** (also called Newton-Raphson) uses second-order information for faster convergence. Newton's method approximates the objective function with a second-order Taylor expansion:

$$
J(\boldsymbol{\beta}) \approx J(\boldsymbol{\beta}^{(t)}) + \nabla J^T(\boldsymbol{\beta} - \boldsymbol{\beta}^{(t)}) + \frac{1}{2}(\boldsymbol{\beta} - \boldsymbol{\beta}^{(t)})^T\mathbf{H}(\boldsymbol{\beta} - \boldsymbol{\beta}^{(t)})
$$

where $\mathbf{H}$ is the Hessian matrix of second derivatives. Setting the gradient of this approximation to zero yields the Newton update:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1}\nabla J
$$

For logistic regression, the Hessian is:

$$
\mathbf{H} = \frac{1}{m}\mathbf{X}^T\mathbf{W}\mathbf{X}
$$

where $\mathbf{W} = \text{diag}(h_i(1-h_i))$ is a diagonal weight matrix. This form leads to an algorithm called **Iteratively Reweighted Least Squares (IRLS)**, as each iteration solves a weighted least squares problem.

**Advantages and Disadvantages:** Newton's method typically requires far fewer iterations than gradient descent—often converging in 5-10 steps versus hundreds or thousands. However, each iteration is more expensive, requiring $O(n^3)$ operations to invert the Hessian versus $O(n^2)$ for a gradient step. Newton's method also lacks a natural online or stochastic variant, requiring batch processing. For very high-dimensional problems, quasi-Newton methods like BFGS or L-BFGS approximate the Hessian more efficiently.

### Regularized Logistic Regression

Logistic regression benefits from regularization to prevent overfitting, especially with high-dimensional features or limited data. The regularized objective combines cross-entropy loss with L1 or L2 penalties:

**L2 (Ridge) Logistic Regression:**

$$
J(\boldsymbol{\beta}) = -\frac{1}{m}\sum_{i=1}^m [y_i\log(h_i) + (1-y_i)\log(1-h_i)] + \frac{\lambda}{2}\sum_{j=1}^n \beta_j^2
$$

**L1 (Lasso) Logistic Regression:**

$$
J(\boldsymbol{\beta}) = -\frac{1}{m}\sum_{i=1}^m [y_i\log(h_i) + (1-y_i)\log(1-h_i)] + \lambda\sum_{j=1}^n |\beta_j|
$$

The gradient includes an additional regularization term. For L2: $\nabla_{\text{reg}} = (\lambda/m)\boldsymbol{\beta}$; for L1: $\nabla_{\text{reg}} = (\lambda/m)\text{sign}(\boldsymbol{\beta})$. Regularized logistic regression prevents the weights from growing unboundedly, which is especially important when data is linearly separable—in such cases, unregularized logistic regression would have weights diverging to infinity to drive predicted probabilities to exactly 0 or 1.

<span style="display:none"></span>

<div align="center">⁂</div>

```text
[^1]: https://statproofbook.github.io/P/slr-ols.html
[^2]: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
[^3]: https://en.wikipedia.org/wiki/Logistic_regression
[^4]: https://www.youtube.com/watch?v=TM1lijyQnaI
[^5]: https://nandeshwar.in/100-days-of-deep-learning/what-is-linear-regression-with-derivation/
[^6]: http://spia.uga.edu/faculty_pages/mlynch/teaching/ols/OLSDerivation.pdf
[^7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6900425/
[^8]: https://www.cs.toronto.edu/~fleet/courses/C11/Handouts/ProbabilityandLS_NG.pdf
[^9]: https://en.wikipedia.org/wiki/Ordinary_least_squares
[^10]: https://www.youtube.com/watch?v=gLGQokSrtDI
[^11]: https://www.scss.tcd.ie/doug.leith/ST3009/slides_10b.pdf
[^12]: https://noiselab.ucsd.edu/ECE285/lecture3.pdf
[^13]: https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
[^14]: https://www.datacamp.com/tutorial/tutorial-gradient-descent
[^15]: https://developers.google.com/machine-learning/crash-course/linear-regression/hyperparameters
[^16]: https://sdgniser.github.io/coding_club_blogs/blogs/2024/03/24/linear-regression-and-gradient-descent.html
[^17]: https://www.cs.cornell.edu/~bindel/class/sjtu-summer18/lec/2018-06-14.pdf
[^18]: https://cs229.stanford.edu/summer2019/BiasVarianceAnalysis.pdf
[^19]: https://www.geeksforgeeks.org/machine-learning/ml-bias-variance-trade-off/
[^20]: https://towardsdatascience.com/machine-learning-bias-variance-tradeoff-and-regularization-94846f945131/
[^21]: https://builtin.com/data-science/l2-regularization
[^22]: https://www.geeksforgeeks.org/machine-learning/ridge-regression-vs-lasso-regression/
[^23]: https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b/
[^24]: https://www.ibm.com/think/topics/ridge-regression
[^25]: https://www.datacamp.com/tutorial/variance-inflation-factor
[^26]: https://www.geeksforgeeks.org/machine-learning/multicollinearity-in-regression-analysis/
[^27]: https://www.reddit.com/r/statistics/comments/mfc0ph/d_is_there_any_geometric_intuition_behind_why_l1/
[^28]: https://www.linkedin.com/pulse/intuitive-visual-explanation-differences-between-l1-l2-xiaoli-chen
[^29]: https://apxml.com/courses/deep-learning-regularization-optimization/chapter-2-weight-regularization/comparing-l1-l2
[^30]: https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf
[^31]: https://www.pinecone.io/learn/regularization-in-neural-networks/
[^32]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
[^33]: https://towardsdatascience.com/courage-to-learn-ml-demystifying-l1-l2-regularization-part-3-ee27cd4b557a/
[^34]: https://www.geeksforgeeks.org/machine-learning/what-is-elasticnet-in-sklearn/
[^35]: https://en.wikipedia.org/wiki/Elastic_net_regularization
[^36]: https://peterroelants.github.io/posts/cross-entropy-logistic/
[^37]: https://ds100.org/course-notes-su23/logistic_regression_2/logistic_reg_2.html
[^38]: https://homes.cs.washington.edu/~marcotcr/blog/linear-classifiers/
[^39]: https://community.deeplearning.ai/t/linearity-of-logistic-regression/457178
[^40]: https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html
[^41]: https://www.sciencedirect.com/topics/computer-science/decision-hyperplane
[^42]: https://www.geeksforgeeks.org/machine-learning/differentiate-between-support-vector-machine-and-logistic-regression/
[^43]: https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf
[^44]: https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning
[^45]: https://www.linkedin.com/pulse/deriving-cross-entropy-function-logistic-regression-nimmagadda
[^46]: https://en.wikipedia.org/wiki/Cross-entropy
[^47]: https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
[^48]: https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/readings/L04_second_order.pdf
[^49]: https://web.stanford.edu/class/msande311/lecture12.pdf
[^50]: https://www.geeksforgeeks.org/machine-learning/newtons-method-in-machine-learning/
[^51]: https://www.cs.mcgill.ca/~dprecup/courses/ML/Lectures/ml-lecture05.pdf
[^52]: https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/14-newton-scribed.pdf
[^53]: https://www.youtube.com/watch?v=VqKq78PVO9g
[^54]: https://www.youtube.com/watch?v=EEwzx9UpgsY
[^55]: https://www.ibm.com/think/topics/logistic-regression
[^56]: https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/
[^57]: https://www.sfu.ca/~dsignori/buec333/lecture 8.pdf
[^58]: https://dev.to/harsimranjit_singh_0133dc/maximum-likelihood-estimation-with-logistic-regression-2g3n
[^59]: https://blog.dailydoseofds.com/p/2-mathematical-proofs-of-ordinary
[^60]: https://mlu-explain.github.io/logistic-regression/
[^61]: https://www.cmrr.umn.edu/~kendrick/statsmatlab/StatsLecture7_Classification.pdf
[^62]: https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method
[^63]: https://towardsdatascience.com/understanding-l1-and-l2-regularization-93918a5ac8d0/
[^64]: https://maitbayev.github.io/posts/why-l1-loss-encourage-coefficients-to-shrink-to-zero/
[^65]: https://www.stat.cmu.edu/~larry/=stat401/lecture-06.pdf
[^66]: https://codesignal.com/learn/courses/regression-and-gradient-descent/lessons/gradient-descent-optimization-in-linear-regression
[^67]: https://www.tandfonline.com/doi/full/10.1080/02664763.2014.980789
[^68]: https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/
[^69]: https://onefishy.github.io/ML_notes/what-are-probablistic-and-non-probablistic-regression.html
[^70]: https://home.iitk.ac.in/~shalab/regression/Chapter9-Regression-Multicollinearity.pdf
[^71]: https://en.wikipedia.org/wiki/Gradient_descent
[^72]: https://bjlkeng.github.io/posts/a-probabilistic-view-of-regression/
[^73]: https://en.wikipedia.org/wiki/Bias–variance_tradeoff
[^74]: https://dev.to/harsimranjit_singh_0133dc/elastic-net-regularization-balancing-between-l1-and-l2-penalties-3ib7
[^75]: https://www.exxactcorp.com/blog/deep-learning/overfitting-generalization-the-bias-variance-tradeoff
[^76]: https://www.geeksforgeeks.org/machine-learning/ml-cost-function-in-logistic-regression/
[^77]: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-l1-l2-and-elastic-net-regularization-with-pytorch.md
[^78]: https://www.ibm.com/think/topics/bias-variance-tradeoff
[^79]: https://www.youtube.com/watch?v=7rR1L7t2EnA
```
