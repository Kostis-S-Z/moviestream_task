"""
This code has been adapted from a tensorflow tutorial at
https://github.com/tensorflow/recommenders
"""

import tensorflow as tf

import matplotlib.pyplot as plt

from data import fetch_data, preprocess_data, split_dataset
from models import MovielensModel

net = [64, 64]
lr = 0.1
n_epochs = 100
seed = 42


def main():
    tf.random.set_seed(seed)

    movies, ratings = fetch_data()

    timestamps, timestamp_buckets, movie_titles, user_ids = preprocess_data(movies, ratings)

    ratings = ratings.shuffle(100_000, seed=seed, reshuffle_each_iteration=False)

    train, test = split_dataset(ratings)

    model = MovielensModel(net, movie_titles, movies, user_ids, timestamps, timestamp_buckets)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(lr))

    history = model.fit(
        train,
        validation_data=test,
        validation_freq=5,
        epochs=n_epochs,
        verbose=0
    )

    train_history = model.evaluate(train, return_dict=True)
    test_history = model.evaluate(test, return_dict=True)

    print(f'Top-100 accuracy (train): {train_history["factorized_top_k/top_100_categorical_accuracy"]:.2f}.')
    print(f'Top-100 accuracy (test): {test_history["factorized_top_k/top_100_categorical_accuracy"]:.2f}.')

    plot_loss(history.history)


def plot_loss(history):
    plt.plot(history['loss'])
    plt.title("Training loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()


if __name__ == '__main__':
    main()
