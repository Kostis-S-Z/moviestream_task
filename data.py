import numpy as np
import pandas as pd
import tensorflow_datasets as tfds


def fetch_data_local(path='ml-100k'):
    """
    This dataset does not come with column names by default, so we are adding them manually.
    :param path: to the dataset
    :return: a dataframe for the movies and the ratings of the users
    """
    genre_cols = ['Unknown_Genre', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
    movies = pd.read_csv(f'{path}/u.item', sep='|', names=movies_cols,
                         encoding='latin-1')

    ratings = pd.read_csv(f'{path}/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    return movies, ratings


def fetch_data():
    """
    Use tensorflow datasets to download MovieLens.
    :return: the ratings of the users and the movies in the dataset
    """
    ratings = tfds.load('movielens/100k-ratings', split='train')
    movies = tfds.load('movielens/100k-movies', split='train')

    ratings = ratings.map(lambda x: {
        'movie_title': x['movie_title'],
        'user_id': x['user_id'],
        'timestamp': x['timestamp'],
    })
    movies = movies.map(lambda x: x['movie_title'])

    return movies, ratings


def preprocess_data(movies, ratings):
    """
    Get the user IDs, the movie titles, the timestamps of the users' ratings and an array of evenly distributed
    vector of 1000 timestamps from the first rating that happened to the last.
    """
    timestamps = np.concatenate(list(ratings.map(lambda x: x['timestamp']).batch(100)))

    max_timestamp = timestamps.max()
    min_timestamp = timestamps.min()

    timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=1000)

    movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
    user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(lambda x: x['user_id']))))

    return timestamps, timestamp_buckets, movie_titles, user_ids


def split_dataset(ratings):
    """
    Make an 80/20 split for train and test data and load them in memory in batches.
    """
    train = ratings.take(80_000)
    test = ratings.skip(80_000).take(20_000)

    cached_train = train.shuffle(100_000).batch(2048)
    cached_test = test.batch(4096).cache()
    return cached_train, cached_test
