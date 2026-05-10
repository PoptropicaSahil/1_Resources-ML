## 1. Matching Pre-trained Embeddings (The Two-Tower Architecture)

If you have a pre-trained video encoder and a pre-trained text encoder, they naturally produce embeddings in two completely different vector spaces. To match them, you utilize a two-tower architecture.

* The Encoders (Towers): One tower is the text encoder (e.g., a Transformer like BERT), and the other tower is the video encoder (e.g., a frame-level model like ViT)
`* The Shared Space: The goal of the model is to project the outputs of both towers into the same multi-dimensional embedding space.`
* Matching Mechanism: Once the text and video are projected into this shared space as embedding vectors, the similarity score between the video and the text is calculated using the dot product of their representations.
  * If the dot product is high, the text and video are semantically similar.

## 2. How the Model is Trained (Contrastive Learning)

You cannot just plug in two pre-trained encoders and expect their outputs to align perfectly. The two-tower model must be trained using a contrastive learning approach.The core idea is to train the model to pull the embeddings of matching $\langle \text{video}, \text{text} \rangle$ pairs closer together while pushing unmatched pairs further apart.

* The Batch Setup: For a given video embedding ($E_v$), you provide the model with $1$ positive text query (the actual matching query) and $n-1$ negative text queries (random, unrelated queries from the dataset).
* Similarity Computation: The model computes the dot product similarity between the video embedding and every single text embedding in that batch: $E_v \cdot E_1, E_v \cdot E_2, \dots, E_v \cdot E_n$.
* Probability Distribution: These similarity scores are passed through a Softmax function, which converts them into a probability distribution representing the likelihood of each text query matching the video.
* The Loss Function: Finally, the model uses cross-entropy loss to compare these predictions against the ground truth. The ground truth is a one-hot vector where the positive match is $1$ and all negative matches are $0$.By minimizing this loss during training, the network updates its weights so that true matches yield the highest similarity scores.

## 3. Fusing the Retrievers

`Once your two-tower visual search system` and your `Elasticsearch text search system` have both retrieved their high-recall candidates, you need to `merge them`. `**The Fusing layer takes the two different lists of relevant videos and combines them into a single list**`.

You mentioned fusing them via another neural network. While possible, the documentation outlines two specific approaches:

* Weighted Sum (Recommended): The easiest and most common way to implement this is to simply re-rank the videos based on a weighted sum of their predicted relevance scores from both systems.
* ML Model (Complex): A more complex approach is to adopt an additional model to re-rank the videos. However, the documentation notes this is often avoided in the fusing stage because it is more expensive (requires additional model training) and makes serving slower.

## 4. The Re-ranking Component

After the high-recall retrieval and the fusion of the lists, the final step before showing results to the user is the Re-ranking service.Even if a video is perfectly relevant semantically, it might not be the best video to show the user from a business perspective. The re-ranking service modifies the ranked list of videos by incorporating business-level logic and policies.This might include logic such as:Filtering out unsafe, restricted, or duplicated content.Boosting newer videos (freshness) or highly popular videos.Applying regional or legal compliance rules.
