import pandas as pd
import tensorflow_datasets as tfds


def fetch_data_local(path='ml-100k'):
    """
    This dataset does not come with column names by default, so we are adding them manually.
    :param path: to the dataset
    :return: a dataframe for the movies and the ratings of the users
    """
    genre_cols = [
        "Unknown_Genre", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

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
    ratings = tfds.load("movielens/100k-ratings", split="train")
    movies = tfds.load("movielens/100k-movies", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })

    return movies, ratings
