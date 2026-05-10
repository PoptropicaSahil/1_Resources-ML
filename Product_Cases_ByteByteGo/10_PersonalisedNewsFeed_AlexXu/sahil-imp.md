1. Why is a Multi-task Neural Network used?

`In a news feed system, user engagement isn't just a simple "click."` The system aims to maximize a weighted score of multiple explicit and implicit reactions, such as clicks, likes, shares, comments, dwell-time, and skips.To predict all of these probabilities for a given $\langle \text{user}, \text{post} \rangle$ pair, the system uses a Multi-task Deep Neural Network (DNN) instead of $N$ independent neural networks for several reasons:

- Efficiency: Training several independent DNNs (one for likes, one for shares, etc.) is highly compute-intensive, time-consuming, and expensive to maintain.
- Handling limited data: For less frequent reactions (like "shares"), there might not be enough training data for an independent model to learn accurately.
- Shared learning: A multi-task DNN solves this by using shared layers that process the input features and learn similarities between the different tasks simultaneously.
- Task-specific predictions: After the shared layers, the network splits into separate classification heads (e.g., a "like classification head," a "dwell-time prediction head"), each outputting the specific probability for that reaction.


`2. What is the Retrieval step?In this news feed design, the retrieval step (often called candidate generation) is rule-based`. Because a news feed only displays content from a user's network, the retrieval service simply filters and fetches posts that the user has not seen yet, or posts that have new comments that the user hasn't seen.