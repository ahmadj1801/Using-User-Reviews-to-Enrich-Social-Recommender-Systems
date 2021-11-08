import math

import keras
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import gensim

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec
from keras.layers import Input, Embedding, Dot, Reshape, Dense, Multiply, Flatten, Dropout, Concatenate, Bidirectional, \
    LSTM, Conv1D, MaxPooling1D, Attention, GlobalAveragePooling1D, SimpleRNN, LeakyReLU
from keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from keras.callbacks import History
from keras.utils.vis_utils import plot_model

from DatasetLoader import DatasetLoader
from tensorflow.python.ops.metrics_impl import root_mean_squared_error

'''File Paths'''
USER_PATH = "../yelp_academic_dataset_user.json"
BUSINESS_PATH = "../yelp_academic_dataset_business.json"
REVIEW_PATH = "../yelp_academic_dataset_review.json"

'''Parameters for optimization'''
DATA_SIZE = 0
USERS = 1
REVIEWS = 500

'''Neural Model Parameters'''
EPOCHS = 10
VALIDATION_SPLIT = 0.05
TEST_SIZE = 0.2
BATCH_SIZE = 128
LEAKY_ALPHA = 0.1
LR = 0.001
DROPOUT = 0.2

'''DOC2VEC Model Parameters'''
D2V_EPOCHS = 1
VEC_SIZE = 64
DM = 1


def get_sentiment(star):
    """
    Given a star rating, calculate the sentiment
    star: rating  [1, 5]
    """
    if star == 3:
        return 0
    elif star < 3:
        return -1
    else:
        return 1


def get_recommendable(star):
    """
    Given a star rating, calculate if an item is recommendable
    star: rating [1, 5]
    """
    if star > 3:
        return 1
    else:
        return 0


def social_recommender(n_users, n_items):
    """
    Neural Model
    """
    business_input = business_vec = Input(shape=[1], name="Business-Input")
    business_embedding = Embedding(n_items + 1, 100, name="Business-Embedding")(business_input)
    business_vec = Flatten(name="Flatten-Business")(business_embedding)

    """User Embedding"""
    user_input = user_vec = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users + 1, 100, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    """Sentiment Input"""
    sentiment_input = sentiment_vec = Input(shape=[1], name="Sentiment-Input")
    # sentiment_embedding = Embedding(3, 100, name="Sentiment-Embedding")(sentiment_input)
    sentiment_vec = Flatten(name="Flatten-Sentiment")(sentiment_input)  # (sentiment_embedding)

    """Friendship Input"""
    friendship_input = Input(shape=(VEC_SIZE,), name="Friendship-Input")
    # friendship_embedding = Embedding(2, 100, name="Friendship-Embedding")(friendship_input)
    friendship_vec = Flatten(name="Flatten-Friendship")(friendship_input)

    """Embedding of average rating by user"""
    average_rating_input = Input(shape=[1], name="Average-Rating-Input")
    average_rating_embedding = Embedding(6, 100, name="Average-Rating-Embedding")(average_rating_input)
    average_vec = Flatten(name="Flatten-Average")(average_rating_embedding)

    """Recommendable Embedding"""
    recommendable_input = recommendable_vec = Input(shape=[1], name="Recommendable-Input")
    recommendable_embedding = Embedding(10, 256, name="Recommendable-Embedding")(recommendable_input)
    recommendable_vec = Flatten(name="Flatten-Recommendable")(recommendable_embedding)

    """Review Input - Doc2Vec"""
    review_input = review_vec = Input(shape=(VEC_SIZE,), name="Review-Input")
    # review_embedding = Embedding(input_dim=VEC_SIZE, output_dim=256)(review_input)
    review_vec = Flatten(name="Flatten-Review")(review_input)  # (review_embedding)

    """Initial Dense layer with Dropout applied"""
    business_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(business_vec)
    business_dense = Dropout(DROPOUT)(business_dense)
    user_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(user_vec)
    user_dense = Dropout(DROPOUT)(user_dense)
    sentiment_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(sentiment_vec)
    sentiment_dense = Dropout(DROPOUT)(sentiment_dense)
    friendship_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(friendship_vec)
    friendship_dense = Dropout(DROPOUT)(friendship_dense)
    average_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA))(average_vec)
    average_dense = Dropout(DROPOUT)(average_dense)
    recommendable_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(recommendable_vec)
    recommendable_dense = Dropout(DROPOUT)(recommendable_dense)
    review_dense = Dense(256, activation=LeakyReLU(alpha=LEAKY_ALPHA), use_bias=True)(review_vec)
    review_dense = Dropout(DROPOUT)(review_dense)

    """User-Item Space - Learn User latent"""
    conc1 = Concatenate()([business_dense, user_dense, sentiment_dense])

    fc1_1 = Dense(128, activation=LeakyReLU(alpha=LEAKY_ALPHA))(conc1)
    fc2_1 = Dense(64, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc1_1)
    fc3_1 = Dense(32, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc2_1)
    fc4_1 = Dense(16, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc3_1)
    fc5_1 = Dense(8, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc4_1)
    out_1 = Dense(4)(fc5_1)

    model1 = Model([business_input, user_input, sentiment_input], out_1)

    """Social Space: User-User - Learn user latent"""
    conc2 = Concatenate()([user_dense, friendship_dense, average_dense])
    fc1_2 = Dense(128, activation=LeakyReLU(alpha=LEAKY_ALPHA))(conc2)
    fc2_2 = Dense(64, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc1_2)
    fc3_2 = Dense(32, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc2_2)
    fc4_2 = Dense(16, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc3_2)
    fc5_2 = Dense(8, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc4_2)
    out_2 = Dense(4)(fc5_2)

    model2 = Model([user_input, friendship_input, average_rating_input], out_2)

    """User-Review Space - learn item latent"""
    conc3 = Concatenate()([user_dense, business_dense, recommendable_dense, review_dense])

    # add fully-connected-layers
    fc1_3 = Dense(128, activation=LeakyReLU(alpha=LEAKY_ALPHA))(conc3)
    fc2_3 = Dense(64, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc1_3)
    fc3_3 = Dense(32, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc2_3)
    fc4_3 = Dense(16, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc3_3)
    fc5_3 = Dense(8, activation=LeakyReLU(alpha=LEAKY_ALPHA))(fc4_3)
    out_3 = Dense(4)(fc5_3)

    model3 = Model([business_input, user_input, recommendable_input, review_input], out_3)

    """Concatenation of latent factors"""
    combined = Concatenate()([model1.output, model2.output, model3.output])

    """Final Layers using the latent representation"""
    z = Dense(4, activation=LeakyReLU(alpha=LEAKY_ALPHA))(combined)
    output = Dense(1, activation=LeakyReLU(alpha=LEAKY_ALPHA))(z)

    model = Model([user_input, business_input, sentiment_input, friendship_input, average_rating_input,
                   review_input, recommendable_input], output)

    """Adam Optimizer"""
    adam = Adam(learning_rate=LR)

    """Compiled model with MSE loss"""
    model.compile(optimizer=adam, loss='mean_squared_error',
                  metrics=['MAE', 'RootMeanSquaredError', 'accuracy'])

    plot_model(model, to_file="../Output/model.png")

    return model


def calculate_accuracy(predictions, truth):
    """
    Calculates accuracy of the model with modified ratings
    predictions: Rounded values from the model
    truth: Actual rating provided from ui for vj
    returns: % correct
    """
    correct = 0
    for i in range(len(predictions)):
        prediction = round(predictions[i][0])
        if prediction == truth[i]:
            correct = correct + 1
    return correct / len(predictions)


def get_doc2vec_vectors(doc2vec_model, ids):
    """
    Retrives the doc2vec vectors
    doc2vec_model: Doc2Vec Model
    ids: list of ID's to identify doc2vec vector
    retuns: List of doc2vec vectors
    """
    vectors = list()
    for i in range(len(ids)):
        id = str(ids[i])
        vector = doc2vec_model.docvecs[id]
        vector = np.array(vector)
        vectors.append(vector)
    return vectors


def get_friendships(friends, all_users):
    encoded_friends = to_categorical(friends, num_classes=len(all_users) + 1, dtype=np.int)
    vector = np.zeros(shape=len(all_users) + 1, dtype=np.int)
    for i in range(len(encoded_friends)):
        vector = np.add(vector, np.array(encoded_friends[i]))
    return vector


def graph_metric(model, metric):
    """
    Code to compute graphical output
    """
    plt.plot(model.history[metric])
    plt.plot(model.history['val_' + metric])
    plt.title('Model ' + metric)
    plt.xlabel('epoch')
    plt.xlabel(metric)
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('../Output/' + metric + '.png')
    plt.show()
    plt.clf()


def parameters_to_file():
    file = open('../Output/parameters.txt', 'w+')
    file.write('\n== Data Parameters ==')
    file.write('\nNumber of Users: ' + str(USERS))
    file.write('\nNumber of Reviews: ' + str(REVIEWS))
    file.write('\n== Training Splits ==')
    file.write('\nTest Split: ' + str(TEST_SIZE))
    file.write('\nValidation Split: ' + str(VALIDATION_SPLIT))
    file.write('\n== Neural Network Parameters ==')
    file.write('\nEpochs: ' + str(EPOCHS))
    file.write('\nDropout Rate: ' + str(DROPOUT))
    file.write('\nLearning Rate: ' + str(LR))
    file.write('\nLeaky ReLu Alpha: ' + str(LEAKY_ALPHA))
    file.write('\nBatch Size: ' + str(BATCH_SIZE))
    file.write('\n== Doc2Vec Parameters ==')
    file.write('\nEpochs: ' + str(D2V_EPOCHS))
    file.write('\nApproach: ' + str(DM))
    file.write('\nEmbedding Dimension: ' + str(VEC_SIZE))


def main():
    """
    Parameters for training via arg parse
    """
    parser = argparse.ArgumentParser(description="Social Recommendation System")
    parser.add_argument("--data", type=int, default=0, help="Read in the full dataset or a portion\n0 - portion\n1- "
                                                            "Full dataset")
    parser.add_argument("--users", type=int, default=1, help="Number of users to read in")
    parser.add_argument("--reviews", type=int, default=500, help="Number of reviews to read in")
    parser.add_argument("--batch_size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--neural_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--validation_split", type=float, default=0.05, help="The split ratio for the validation set")
    parser.add_argument("--test_split", type=float, default=0.2, help="The split ratio for the test set")
    parser.add_argument("--leaky_alpha", type=float, default=0.1, help="The alpha value for the activation function")
    parser.add_argument("--d2v_epochs", type=int, default=1, help="The number of epochs to train the D2V model")
    parser.add_argument("--d2v_vec_size", type=int, default=64, help="The output vector size from the D2V model")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the neural model')
    parser.add_argument('--drp', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--dm', type=int, default=1, help='Approach for Doc2Vec Training')
    parser.add_argument("-f")
    arguments = parser.parse_args()

    # Using the global variables
    global EPOCHS, VALIDATION_SPLIT, TEST_SIZE, BATCH_SIZE, LEAKY_ALPHA, D2V_EPOCHS, VEC_SIZE, LR, DROPOUT, USERS, \
        REVIEWS, DATA_SIZE, DM

    """
    Extract the parameters from the command line. 
    Alternatively, use the defaults.
    """
    DATA_SIZE = arguments.data
    USERS = arguments.users
    REVIEWS = arguments.reviews
    EPOCHS = arguments.neural_epochs
    VALIDATION_SPLIT = arguments.validation_split
    TEST_SIZE = arguments.test_split
    BATCH_SIZE = arguments.batch_size
    LEAKY_ALPHA = arguments.leaky_alpha
    LR = arguments.lr
    DROPOUT = arguments.drp

    D2V_EPOCHS = arguments.d2v_epochs
    VEC_SIZE = arguments.d2v_vec_size  # For DOC2VEC training
    DM = arguments.dm

    parameters_to_file()

    # Instance of data set
    dataset = DatasetLoader()

    if DATA_SIZE == 0:
        # Load a portion of the user dataset
        users = dataset.load_data(USER_PATH, USERS)

        # load a portion of the reviews dataset
        reviews = dataset.load_data(REVIEW_PATH, REVIEWS)

    else:
        # Load the full dataset
        users = dataset.load_data(USER_PATH)

        # load the full dataset
        reviews = dataset.load_data(REVIEW_PATH)

    print('Users Read')
    print('Reviews Read')

    users.drop(columns=['review_count', 'useful', 'funny',
                        'cool', 'elite', 'fans', 'average_stars', 'compliment_hot',
                        'compliment_more', 'compliment_profile', 'compliment_cute',
                        'compliment_list', 'compliment_note', 'compliment_plain',
                        'compliment_cool', 'compliment_funny', 'compliment_writer',
                        'compliment_photos'], axis=1, inplace=True)

    reviews.drop(columns=['useful', 'funny', 'cool'], axis=1, inplace=True)

    # load the full business dataset
    business = dataset.load_data(BUSINESS_PATH)  # 160 585
    business.drop(columns=['address', 'city', 'state', 'postal_code',
                           'latitude', 'longitude', 'review_count', 'is_open',
                           'attributes', 'categories', 'hours'], axis=1, inplace=True)
    print('Businesses Read')

    """
    Encode the business ID's into numbers
    """
    print('Encoding Business IDs')
    business_encoder = LabelEncoder()
    business['business_id'] = business_encoder.fit_transform(business.business_id)
    
    print('Encoding Review IDs')
    review_encoder = LabelEncoder()
    reviews['review_id'] = review_encoder.fit_transform(reviews.review_id)
    reviews['business_id'] = business_encoder.transform(reviews.business_id)

    # get all the users so they can be encoded uniformly
    user_list = users['user_id'].tolist()
    review_users = reviews['user_id'].tolist()
    friends = dataset.friends_list(users)

    # List of all users
    complete_username_list = user_list + friends + review_users

    # Encoder to encode users
    print('Encoding User IDs')
    user_encoder = LabelEncoder()
    complete_user_id_list = user_encoder.fit_transform(complete_username_list)
    unique_users = set(complete_user_id_list)
    list_unique_users = list(unique_users)

    # update the reviews data frame
    users['user_id'] = user_encoder.transform(users.user_id)
    reviews['user_id'] = user_encoder.transform(reviews.user_id)
    f = user_encoder.transform(friends)

    # Turn friends list into number ids
    users = dataset.update_user_friends(users, friends, f)

    averages = reviews.groupby('user_id').agg(['mean'])
    averages.drop(columns=['business_id', 'review_id'], axis=1, inplace=True)
    # print(averages.columns)
    # averages.columns = ['stars_mean']

    review_users = reviews['user_id'].tolist()
    data = list()
    for user in review_users:
        star = averages.loc[user].tolist()[0]
        data.append(star)
    data = np.asarray(data).astype('float32')
    reviews['average_star'] = data
    review_users = reviews['user_id'].unique().tolist()
    read_users = users['user_id'].tolist()
    user_friends = users['friends'].tolist()
    all_users_with_friends = list()
    review_users_with_friends = list()
    for user in review_users:
        if user in read_users:
            index = read_users.index(user)
            f = np.asarray(user_friends[index], dtype=str)
        else:
            f = np.asarray([user], dtype=str)
        f = ' '.join(map(str, f))
        all_users_with_friends.append(f)

    """
    We can let the ID's be list_unique_users
    We can let the data be all_users_with_friends
    """

    list_of_reviews = reviews['text'].tolist()
    clean_reviews = dataset.clean_reviews(list_of_reviews)

    """
    Obtain a sentiment for the review.
    1   - Good
    0   - Neutral
    -1  - Bad
    """
    reviews['sentiment'] = [get_sentiment(x) for x in reviews['stars']]

    """
    From the review obtain if the business is recommendable
    1   - Recommendable
    -1  - Not Recommendable
    """
    reviews['recommendable'] = [get_recommendable(x) for x in reviews['stars']]

    max_id = max(list(unique_users))

    '''DOC2VEC MODEL'''
    review_ids = reviews['review_id'].tolist()
    doc2vec_model = dataset.doc2vec_model(review_ids, clean_reviews, epochs=D2V_EPOCHS, vec_size=VEC_SIZE, dm=DM)
    doc2vec_model = Doc2Vec.load(doc2vec_model)

    #  DOC2VEC Vocabulary size
    vocab_size = len(doc2vec_model.wv)

    vectors = get_doc2vec_vectors(doc2vec_model, reviews['review_id'].tolist())

    """User 2 Vec Model - Removed -> list_unique_users"""
    user2vec_model = dataset.user2vec_model((reviews['user_id'].unique()).tolist(), all_users_with_friends, epochs=1, vec_size=VEC_SIZE, dm=1)
    # user2vec_model = "user2vec.model"

    user2vec_model = Doc2Vec.load(user2vec_model)
    friend_vectors = get_doc2vec_vectors(user2vec_model, reviews['user_id'].tolist())

    reviews['doc2vec'] = vectors
    reviews['friends'] = friend_vectors

    # Train and testing portions
    train, test = train_test_split(reviews, test_size=TEST_SIZE, random_state=42)
    train_reviews = pd.DataFrame(train["doc2vec"].tolist())
    test_reviews = pd.DataFrame(test["doc2vec"].tolist())
    train_friends = pd.DataFrame(train["friends"].tolist())
    test_friends = pd.DataFrame(test["friends"].tolist())
    train_all_friends = pd.DataFrame(reviews["friends"].tolist())

    model = social_recommender(n_users=len(unique_users), n_items=len(business))

    model.summary()

    early_stoppage = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.01)

    history = model.fit([train.user_id, train.business_id, train.sentiment, train_friends, train.average_star,
                         train_reviews, train.recommendable], train.stars, validation_split=VALIDATION_SPLIT,
                        batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[early_stoppage])

    graph_metric(history, 'accuracy')
    graph_metric(history, 'loss')
    graph_metric(history, 'MAE')
    graph_metric(history, 'root_mean_squared_error')

    predictions = model.predict([test.user_id, test.business_id, test.sentiment, test_friends, test.average_star,
                                 test_reviews, test.recommendable])

    evaluation = model.evaluate([test.user_id, test.business_id, test.sentiment, test_friends, test.average_star,
                                 test_reviews, test.recommendable], test.stars,
                                batch_size=BATCH_SIZE)

    file = open('../Output/results.txt', 'w+')
    file.write("Accuracy = " + str(calculate_accuracy(predictions, test.stars.tolist()) * 100))
    # print('Accuracy = {}%'.format(calculate_accuracy(predictions, test.stars.tolist()) * 100))
    for i in range(len(model.metrics_names)):
        file.write("\n" + model.metrics_names[i] + " = " + str(evaluation[i]))
        print("{} = {}".format(model.metrics_names[i], evaluation[i]))


if __name__ == "__main__":
    main()
