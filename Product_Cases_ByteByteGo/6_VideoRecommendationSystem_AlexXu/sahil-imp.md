> Useful for main doubts

To train a single two-tower model jointly for video recommendations, you would design an architecture with two parallel encoders: a user tower and a video tower

## Training Data, Labels, and Loss

- **Training Data:** `The dataset is constructed from ⟨user, video⟩ pairs`
  - **User features** include demographics (age, gender, location), contextual information (time of day, device), and historical interactions (search history, watched videos)
  - **Video features** include metadata such as video ID, duration, language, titles, and tags
- **Labels**: The system uses `binary labels` based on user feedback
  - **Positive (1)**: When a user `explicitly likes a video or watches at least half of it`
  - **Negative (0)**: To handle the imbalanced nature of the data, `negative samples are generated from random videos` the user hasn't interacted with or `videos they explicitly disliked`
- **Loss Function**: Since this is framed as a classification task, you use `Cross-Entropy loss` to optimize the encoders

## Calculating User and Content Embeddings (JOINT)

The joint training process forces both towers to map their respective inputs into a shared embedding space

- **User Embeddings (u)**: The user tower takes the raw user features and passes them through embedding layers (for categorical data) and neural network layers to produce a `single vector u`
- **Content Embeddings (v)**: Similarly, the video tower encodes video metadata and features into a `vector v` of the `same dimension`
- **Similarity**: During training, the model learns parameters so that the `dot product` (or similarity) between u and v is high for positive pairs and low for negative pairs

## Candidate Generation (Retrieval)

During this stage, the goal is to quickly find a subset of videos from billions of possibilities

- **Obtaining User Embeddings**: The system takes the current user's profile and real-time context (e.g., they are on a mobile device at 8:00 PM) and passes them through the `trained User Tower` to compute a live user embedding vector
- **Retrieval**: The video embeddings for the entire library are typically pre-computed and stored in an `Approximate Nearest Neighbor (ANN)` index. The `live user vector is used as a query` against this index to retrieve the top k most similar video embeddings

## Reranking (Scoring)

Once you have retrieved thousands of candidates via ANN, the model is used as a reranker to provide more precise scores

- **Scoring Process**: For the few thousand candidates, the system `feeds the user features and the specific video features` of each candidate into the two-tower model
- **Precision over Efficiency**: While candidate generation often uses a simplified version (like an embedding layer for videos), the ranking stage utilizes the `full "heavy" model` with `more parameters` and `richer video features` to calculate a specific relevance score for every candidate. The videos are then sorted by these scores and presented to the user
