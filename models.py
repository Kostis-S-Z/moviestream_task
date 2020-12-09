"""
This code has been adapted from a tensorflow tutorial at
https://github.com/tensorflow/recommenders
"""
import tensorflow as tf
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
    """
    Embedding model encoding user IDs and timestamps.
    """

    def __init__(self, user_ids, timestamps, timestamp_buckets):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(user_ids) + 1, 32),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        ])
        self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

        self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        # Pass input dictionary to each embedding layer accordingly and return a concatenated results.
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            self.normalized_timestamp(inputs["timestamp"]),
        ], axis=1)


class QueryModel(tf.keras.Model):
    """
    Embedding model encoding user queries.
    Uses as input the output of the UserModel.
    """

    def __init__(self, net_layers, user_ids, timestamps, timestamp_buckets):
        super().__init__()

        self.embedding_model = UserModel(user_ids, timestamps, timestamp_buckets)

        self.query_model = tf.keras.Sequential()

        for layer_size in net_layers[:-1]:
            self.query_model.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # Leave last layer as linear since we will use is as input to the next embedding.
        for layer_size in net_layers[-1:]:
            self.query_model.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        # Use UserModel embedding as input
        feature_embedding = self.embedding_model(inputs)
        return self.query_model(feature_embedding)


class MovieModel(tf.keras.Model):
    """
    Embedding model encoding movie titles.
    """

    def __init__(self, movie_titles, movies, max_tokens=10_000):
        super().__init__()

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """
    Embedding model for encoding movie ratings.
    Uses as input the output of the MovieModel.
    """

    def __init__(self, layer_sizes, movie_titles, movies):
        super().__init__()

        self.embedding_model = MovieModel(movie_titles, movies)

        self.candidate_model = tf.keras.Sequential()

        for layer_size in layer_sizes[:-1]:
            self.candidate_model.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # Leave last layer as linear since we will use is as input to the next embedding.
        for layer_size in layer_sizes[-1:]:
            self.candidate_model.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        # Use MovieModel embedding as input
        feature_embedding = self.embedding_model(inputs)
        return self.candidate_model(feature_embedding)


class MovielensModel(tfrs.models.Model):
    """
    Put all the pieces together. Use the Query and Movie embeddings to train.
    """

    def __init__(self, layer_sizes, movie_titles, movies, user_ids, timestamps, timestamp_buckets):
        super().__init__()
        self.query_model = QueryModel(layer_sizes, user_ids, timestamps, timestamp_buckets)
        self.candidate_model = CandidateModel(layer_sizes, movie_titles, movies)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model), ), )

    def compute_loss(self, features, training=False):
        """ Note from TF: We only pass the user id and timestamp features into the query model.
        This is to ensure that the training inputs would have the same keys as the
        query inputs. Otherwise the discrepancy in input structure would cause an
        error when loading the query model after saving it.
        """
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
        })
        movie_embeddings = self.candidate_model(features["movie_title"])

        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)
