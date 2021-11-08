import numpy as np
import pandas as pd
import nltk
import re
import string
import gensim

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

nltk.download("stopwords")
nltk.download("wordnet")


class DatasetLoader:

    """
    Load the JSON data from the appropriate location
    path: the path to the file
    rows: default=False | if false, the full file will be read
     """
    def load_data(self, path, rows=False):
        if rows == False:
            data = pd.read_json(path, lines=True)
        else:
            data = pd.read_json(path, lines=True, nrows=rows)
        return data

    """
    Add each friends to a list
    Assumption: users and reviews are the same length
    :returns friends
    """
    def friends_list(self, users):
        friends = []
        for i in range(len(users)):
            users_friends = users.iloc[i]['friends']
            users_friends = self.split_friends(users_friends)
            friends = friends + users_friends
        return friends

    """
    Turn textual friend relationships into numeric ones
    :returns users
    """
    def update_user_friends(self, users, usernames, ids):
        for i in range(len(users)):
            users_friends = users.iloc[i]['friends']
            users_friends = self.split_friends(users_friends)
            f = []
            for friend in users_friends:
                pos = usernames.index(friend)
                f.append(ids[pos])
            users.at[i, 'friends'] = f
        return users

    """
        Create a matrix with ratings in each cell
        review users -> rows
        businesses -> columns
        ratings -> cell
    """
    def ratings_matrix(self, reviews):
        users = reviews['user_id'].tolist()
        tmp_users = list(set(users))
        businesses = reviews['business_id'].tolist()
        tmp_businesses = list(set(businesses))
        ratings = reviews['stars'].tolist()
        # null_matrix = [0 for i in range(len(tmp_businesses))]
        adj_matrix = [[0 for i in range(len(tmp_businesses))] for i in range(len(tmp_users))]
        for i in range(len(ratings)):
            user_id = users[i]
            user_pos = tmp_users.index(user_id)
            bus_id = businesses[i]
            bus_pos = tmp_businesses.index(bus_id)
            rating = ratings[i]
            adj_matrix[user_pos][bus_pos] = rating
            print("{} of {}".format(i, len(ratings)))
        return adj_matrix

    """
    Get the friendship relations in tuples
    :returns a list of tuples of node edges
    """
    def friend_tuples(self, users):
        user_ids = users['user_id'].tolist()
        friendships = users['friends'].tolist()
        edges = []
        for i in range(len(user_ids)):
            user = user_ids[i]
            friends = friendships[i]
            for friend in friends:
                edge = (user, friend)
                edges.append(edge)
        return edges

    """
    Get the user purchase relation in tuples
    :returns a list of tuples of node edges
    """
    def user_rating_tuples(self, reviews):
        user_ids = reviews['user_id'].tolist()
        bus_ids = reviews['business_id'].tolist()
        edges = []
        for i in range(len(user_ids)):
            edge = (user_ids[i], bus_ids[i])
            edges.append(edge)
        return edges

    """
    Clean the reviews
    :returns a list of cleaned reviews
    """
    def clean_reviews(self, reviews):
        cleaned_reviews = []
        for review in reviews:
            review = self.clean_text(review)
            cleaned_reviews.append(review)
        return cleaned_reviews

    """
    A function to clean text
    :returns a clean text
    """
    def clean_text(self, review):
        stop_words = set(stopwords.words("english"))
        porter_stemmer = PorterStemmer()
        lemma = WordNetLemmatizer()
        # Convert to lower case
        review = review.lower()
        # Remove stop words
        review = " ".join([word for word in review.split() if word not in stop_words])
        # Remove URLS
        review = re.sub("https?:\/\/.*[\r\n]*", "", review)
        # Remove hash tags
        review = re.sub("#", "", review)
        # Remove punctuation
        punct = set(string.punctuation)
        review = "".join([letter for letter in review if letter not in punct])
        review = " ".join([lemma.lemmatize(word) for word in review.split()])
        return review

    """
    Creates and returns a trained Doc2Vec model
    :returns doc2vec model
    :param ids -> a list of reviewids
    :param data -> a list of text
    :param epochs -> iterations to train on - default=10
    :param size -> size of the vector - deafult=32
    :param alpha ->default=0.025
    :param dm ->Defines the training algorithm - default=1
    dm = 1 (Distributed memory) | dm = 2(Bag of words)
    """
    def doc2vec_model(self, ids, data, epochs=20, vec_size=64, alpha=0.025, dm=1):
        model_name = "doc2vec.model"
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(id) for id in ids])
                       for i, _d in enumerate(data)]
        model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=alpha*0.001, min_count=1, dm=dm, workers=20)
        model.build_vocab(tagged_data)

        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= 0.0002
            model.min_alpha = model.alpha
        model.save(model_name)
        print("Model trained and saved")
        return model_name

    
    def user2vec_model(self, ids, data, epochs=20, vec_size=200, alpha=0.025, dm=2):
        model_name = "user2vec.model"
        tagged_data = [TaggedDocument(words=_d.split(), tags=[str(id) for id in ids])
                       for i, _d in enumerate(data)]
        model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=alpha * 0.001, min_count=1, dm=dm, workers=20)
        model.build_vocab(tagged_data)
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
            model.alpha -= 0.0002
            model.min_alpha = model.alpha
        model.save(model_name)
        print("Model trained and saved")
        return model_name

    """
    Split a comma separated friends list
    :returns array of friends
    """
    def split_friends(self, friends):
        f = friends.split(',')
        return f
