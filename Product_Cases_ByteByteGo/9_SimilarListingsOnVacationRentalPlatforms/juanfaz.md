8. Airbnb: Similar Listings on Vacation Rental
traditional recommendation systems: contect-independent user interests, not changing frequently
session-based recommendation systems for similar listings: user interest evolve fast, we can learn item embeddings using co-occurrences of items in user's browsing history.
Model

predicting neighboring listings: input (central listing) -> shallow neural network -> predict context -> booked listing
training: embeddings are gradually learned by reading through search sessions, use sliding window to predict
construct positive and negative pairs dataset by negative sampling (negative pairs: dissimilar embeddings)
loss function
compute distance (dot product) between two embeddings, use sigmoid to convert distance to prob value, use cross-entropy as classification loss to measure predicted prob and ground truth label. But it's good for user click, not optimal for user to book.
add eventually booked listing as global context, to update central listing vector embedding as positive pairs. Add negative pairs from same region to training data.
Evaluation: update average rank of eventually-booked listing

Serving

training pipeline: fine-tuning -> trained model
indexing pipeline: trained model + listings -> indexer -> index table
prediction pipeline: currently-viewing listing + index table -> embedding featcher service (input listing is seen or not) + listing embedding -> nearest neighbor service -> reranking service -> recommended similar listings