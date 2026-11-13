# Machine learning glossary (sample RAG corpus)

A small, original glossary for trying out RAG Studio. Paste it in as a text source, or copy this file into `backend/storage/` and load it as a file.

## Data preparation

**Missing values.** Cells with no recorded value. Common fixes are dropping the rows, filling numbers with the median, or filling categories with the most frequent value. Tree models such as gradient boosting can often handle them directly.

**Outlier.** A value far from the rest of the distribution, for example a monthly charge of 999 when most customers pay between 20 and 140. Outliers can be real or data-entry errors, so inspect them before removing them.

**Duplicate rows.** Identical records that appear more than once. They inflate some patterns and can leak between the train and test split, which makes scores look better than they are.

**Scaling.** Rescaling numeric features to a similar range. Distance-based models (k-nearest neighbours, SVMs) and linear models with regularisation need it. Tree models do not.

**One-hot encoding.** Turning a category column such as contract type into one 0/1 column per category, so models that need numbers can use it.

**Data leakage.** When information that would not be available at prediction time sneaks into training, for example using total charges to predict whether a customer has already churned.

## Modelling

**Train/test split.** Holding back part of the data to measure how well a model works on rows it has never seen.

**Cross-validation.** Splitting the data into k folds, training k times and averaging the scores. It gives a more stable estimate than a single split, at k times the cost.

**Stratified split.** A split that keeps the share of each class the same in every fold. Important when one class is rare.

**Logistic regression.** A linear model for classification that outputs probabilities. Fast, easy to explain, and a strong baseline.

**Random forest.** An average of many decision trees, each trained on a bootstrap sample of the rows and a random subset of the features. Robust and needs little tuning.

**Gradient boosting.** Builds shallow trees one after another, each one correcting the errors of the trees before it. XGBoost and LightGBM are popular, fast implementations.

**Hyperparameter tuning.** Searching for the model settings (tree depth, learning rate, regularisation strength) that give the best cross-validated score, using a grid or random search.

**Overfitting.** When a model learns noise in the training data, so it scores well in training but badly on new data. More data, simpler models, and regularisation all help.

## Evaluation

**Accuracy.** The share of predictions that are correct. Misleading when classes are imbalanced.

**Precision and recall.** Precision is how many predicted positives were right. Recall is how many real positives were found. For churn, recall tells you how many leaving customers you would catch.

**F1 score.** The harmonic mean of precision and recall, useful when you need one number that balances both.

**ROC AUC.** How well the model ranks positives above negatives across all thresholds. 0.5 is random guessing, 1.0 is perfect.

## Retrieval-augmented generation

**Chunking.** Splitting documents into passages small enough to retrieve and fit in a prompt. Sentence and recursive chunking keep passages readable.

**Dense retrieval.** Finding passages whose vector representation is close to the question's vector.

**BM25.** A classic keyword-based ranking function. It is strong when the question shares exact words with the answer.

**Hybrid retrieval.** Combining dense and keyword results, often with reciprocal rank fusion, to get the strengths of both.

**MMR (maximal marginal relevance).** Picks results that are relevant but not near-duplicates of each other, so the context covers more ground.
