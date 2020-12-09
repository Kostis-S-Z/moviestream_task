import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from data import fetch_data_local

n_vectors = 50


def main():
    movies, ratings = fetch_data_local()
    user_ids = np.unique(list(ratings['user_id']))

    all_user_predictions = make_predictions(ratings, n_vectors)

    store_k_rec_per_user(user_ids, all_user_predictions, movies, ratings, 1)
    store_k_rec_per_user(user_ids, all_user_predictions, movies, ratings, 5)
    store_k_rec_per_user(user_ids, all_user_predictions, movies, ratings, 10)

    # Random sample
    sample_user_result(np.random.choice(user_ids), all_user_predictions, movies, ratings)
    # Sample by user input
    while True:
        user_id = int(input('Enter the ID of the user you want to suggest the next movie or 0 to exit:'))

        if user_id in user_ids:
            sample_user_result(user_id, all_user_predictions, movies, ratings)
        elif user_id == 0:
            break
        else:
            print('Sorry, user ID does not exist!')


def make_predictions(ratings, vectors=50):
    """
    Perform Singular Value Decomposition (SVD)
    """
    # Index ratings of movie_ids based on user_ids and fill missing values with 0
    ratings_mat = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    r_matrix = ratings_mat.to_numpy()
    mean_user_rating = np.mean(r_matrix, axis=1).reshape(-1, 1)
    r_matrix_demean = r_matrix - mean_user_rating

    u_mat, sigma, vt_mat = svds(r_matrix_demean, k=vectors)
    sigma = np.diag(sigma)

    predicted_ratings = np.dot(np.dot(u_mat, sigma), vt_mat) + mean_user_rating.reshape(-1, 1)

    predicted_ratings = pd.DataFrame(predicted_ratings, columns=ratings_mat.columns)

    return predicted_ratings


def recommend_movies(user_id, predicted_ratings, movies, ratings, n_movies=10):
    """
    Given a user ID recommend n movies they might like depending on the predicted ratings.
    :return: a dataframe of the top n movies
    """
    user_id = user_id - 1  # user IDs start from 1
    sorted_user_pred = predicted_ratings.iloc[user_id].sort_values(ascending=False)

    user_watched = ratings[ratings.user_id == user_id]
    # merge in one dataframe the movies they have watched with all the others
    user_total = user_watched.merge(movies, how='left', left_on='movie_id', right_on='movie_id')
    user_total = user_total.sort_values(['rating'], ascending=False)

    watched = movies['movie_id'].isin(user_total['movie_id'])
    not_watched = movies[~watched]

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (not_watched.merge(pd.DataFrame(sorted_user_pred).reset_index(),
                                         how='left',
                                         left_on='movie_id',
                                         right_on='movie_id').rename(columns={user_id: 'Predictions'}))

    recommendations = recommendations.sort_values('Predictions', ascending=False).iloc[:n_movies, :-1]

    return recommendations


def store_k_rec_per_user(user_ids, all_user_predictions, movies, ratings, k):
    """
    Create a dataframe sorted by user IDs with their top K recommendations
    """
    recommendations_per_user = []
    for i in user_ids:
        rec_i = recommend_movies(i, all_user_predictions, movies, ratings, n_movies=k)
        rec_i = rec_i.assign(user_id=i)
        recommendations_per_user.append(rec_i)

    all_recommendations_per_user = pd.concat(recommendations_per_user)

    all_recommendations_per_user = all_recommendations_per_user.filter(['user_id', 'title', 'release_date', 'imdb_url'])
    all_recommendations_per_user.to_csv(f'top_{k}.csv', index=False)


def sample_user_result(user_id, all_user_predictions, movies, ratings):
    """
    Print the next movie a user will want to see with a friendly message.
    """

    recommendation = recommend_movies(user_id, all_user_predictions, movies, ratings, n_movies=1)
    recommendation = recommendation['title'].values[0]

    prompts = [
        ['what about', '?'],
        ['I think you will also like', '!'],
        ['try', '!']
    ]
    msg = np.random.choice(len(prompts))

    print(f'User {user_id} {prompts[msg][0]} "{recommendation}" {prompts[msg][1]}')


if __name__ == '__main__':
    main()
