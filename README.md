# Recommendation system for MovieLens dataset

Next movie predictions based on the [Movielens dataset](https://grouplens.org/datasets/movielens/100k/). Descriptions of the dataset can be found in the [original page](https://grouplens.org/datasets/movielens), [tensorflow tutorials](https://www.tensorflow.org/datasets/catalog/movielens), [kaggle](https://www.kaggle.com/c/movielens-100k).


## Table of Contents

* [About](#about)
* [Installation](#Installation)
* [Project Structure](#project-structure)
  * [Run](#run)
* [Possible Approaches](#approaches)

## About

The task is to find patterns on what movies different types of users like. This can be achieved by examining the rankings each user has already made and try to predict how they will rank the rest of the movies. One method to predict these rankings is with Singular Value Decomposition. Deep Learning can also be employed in this case using context-aware neural networks to take into account the timestamp of the ratings of the users.

## Installation


1. Clone the repo
```sh
git clone https://github.com/Kostis-S-Z/moviestream_task.git
```

2. _(Optional, but highly recommended)_ Make a virtual environment

```python3 -m venv meta_env``` or ``` virtualenv meta_env```


3. Install core dependencies

```pip install -r requirements.txt```

## Project structure

**ml-100k/**: Directory containing the MovieLens-100k dataset

**data.py**: Module to load the dataset, either from local files or from the tensorflow dataset.

**mf.py**: Matrix Factorization (SVD) method to rank (predict) the next movie(s) a user will watch.

**top_k.csv**: The output of the script in a CSV format containing the next top K movies a user will see. The csv is indexed by user ID.

***Draft***

_context_nn.py_: Context-aware Neural Net based on timestamp of users' ratings. Not yet deployed.

_models.py_: Embedding models.

### Run

To run the script and get the output recommendations, simply run:

```python3 mf.py```


## Approaches

In this repo a popular and simple Matrix Factorisation technique was used named Singular Value Decomposition (SVD). Other approaches to handle this problem are mentioned below.

#### Simple Matrix Factorization (ML, not DL)

[SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) or [NMF (Non-Negative MF)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)


#### Simple embeddings of users and movies (barely DL)

[embeddings shallow network](https://github.com/tensorflow/recommenders/blob/main/docs/examples/featurization.ipynb)

#### Adding context i.e timestamp (DL)

- [context-shallow](https://github.com/tensorflow/recommenders/blob/9f08160ab58cb19e5360e3c83f2aac555b7d4dd0/docs/examples/context_features.ipynb)
- [context-deep](https://github.com/tensorflow/recommenders/blob/82feca08f5cecdd925dd99d44e3fa3c13692c616/docs/examples/deep_recommenders.ipynb)

Implement DL approach using context feature i.e time
adding more layers lets use search for more complex connections / patterns across the data, such as the time each user liked a movie


#### Performance > Accuracy

Google published a new model for Large-Scale recommendations user_id with a focus in low computation cost in ICML 2020. [ScaNN blog](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) + [ScaNN implementation](https://github.com/tensorflow/recommenders/blob/main/docs/examples/efficient_serving.ipynb)

#### Other 

- Collaborative filtering with MF ("standard", without NN) [link1](https://www.kaggle.com/riyadhrazzaq/collaborative-filtering-w-mf-from-scratch), [link2](https://www.kaggle.com/premstroke95/basics-of-collaborative-factorization) 

- Low Rank MF with NN [link](https://www.kaggle.com/rajmehra03/cf-based-recsys-by-low-rank-matrix-factorization)

- Neural Collaborative Filtering (NCF) [[paper](https://arxiv.org/abs/1708.05031), [NeuMF](https://github.com/tensorflow/models/tree/08bb9eb5ad79e6bceffc71aeea6af809cc78694b/official/recommendation)]

- If we want to focus on the sequence pattern on how a user's taste "evolves" through time we could look into Sequence / Recurrent models e.g RNN (LSTM) to capture time dependencies.

- We could look into Graph Neural Nets if we want to scale up, have access to more data (+meta-data) in order to capture complex relationships across movies, users and their shared interests.

- During cold-start problems our predictions might be not good enough. We could use some more probablistic / bayesian approaches to provide a confidence interval of how well we think our recommendation is. 

- [Deploying a Rec system on GCP](https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview)
