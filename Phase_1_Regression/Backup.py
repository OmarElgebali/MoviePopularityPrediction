import json
import nltk
import numpy as np
from matplotlib import pyplot as plt
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def train_dict_to_list1(dataset, column, key):
    """
    Fit Action => convert each row in (column) to Json then extract all possible values in that column
    :param dataset: the dataset that values will be extracted from
    :param column: the feature that will apply the extraction
    :param key: the key to get the specific value of dictionary
    :return: list of trained possible values
    """
    dataset[column] = dataset[column].apply(lambda c: json.loads(c))
    values_list = []
    for item_list in dataset[column]:
        for item in item_list:
            specific_key = item[key]
            if specific_key not in values_list:
                values_list.append(specific_key)

    return values_list

def train_dict_to_list2(dataset):
    def generate_possible_vales_list_based_target(df, feature_name, key):
        """
            1. Track all ratings associated with each feature in a dictionary
            2. Calculate average ratings for each feature
            3. Create and sort a list of tuples (dictionary value, key)
            4. Create a list of only the feature names, from lowest rating to highest rating
        :param df: the dataset that values will be extracted from
        :param key: the key to get the specific value of dictionary
        :param feature_name: the feature that will apply the extraction
        :return: a list of all unique possible feature values sorted based on target average
        """
        #
        feature_dict = {}  # (summation of all target values of same feature value, total number of feature values)
        for index, r in df.iterrows():
            feature_values_in_row = r[feature_name]
            for sub_value in feature_values_in_row:
                sub_value_name = sub_value[key]
                if sub_value[key] not in feature_dict:
                    feature_dict[sub_value_name] = (df.iloc[index]['vote_average'], 1)  #
                else:
                    feature_dict[sub_value_name] = (feature_dict[sub_value_name][0] + (df['vote_average'][index]),
                                                    feature_dict[sub_value_name][1] + 1)

        for possible_value in feature_dict:
            # converts the tuple into average of all vote_averages
            feature_dict[possible_value] = feature_dict[possible_value][0] / feature_dict[possible_value][1]
        lst_possible_vals_and_votesAVG = []
        for name in feature_dict:
            lst_possible_vals_and_votesAVG.append((feature_dict[name], name))
        lst_possible_vals_and_votesAVG = sorted(lst_possible_vals_and_votesAVG)
        feature_list = []
        ratings_list = []  # unuseful
        for element in lst_possible_vals_and_votesAVG:
            feature_list.append(element[1])
            ratings_list.append(element[0])

        # get the variance of the ratings. This is helpful for determining the usefulness of the information (to be displayed in below plot)
        var = round(np.var(ratings_list), 3)

        # before returning the list, do a quick visualization to show that generate_list works
        fig, ax = plt.subplots(figsize=(6, 5))
        if feature_name not in ['genres', 'spoken_languages', 'production_countries']:
            n = 70  # sample at intervals of n
        else:
            n = 1
        X = []  # sample for associated movie(s) rating average
        Y = []  # sample for feature names
        for i in range(0, len(feature_list) - 1, n):
            X.append(ratings_list[i])
            Y.append(feature_list[i])
        """
        # to show how feature values sorted based on target average
        y_pos = np.arange(len(Y))
        ax.barh(y_pos, X, align='center')
        # ax.set_yticklabels(Y)
        ax.invert_yaxis()  # labels read top-to-bottom

        ax.set_xlabel('Overall average movie ratings')
        ax.set_ylabel(feature_name + ' sample list index')
        ax.set_title(feature_name + ' to associated movie(s) performance (' + str(
            int(len(feature_list) / n)) + ' samples), variance: ' + str(var))

        plt.show()
        """
        return feature_list

def text_tokenize(dataset, feature):
    """
    made for long text to remove Stopwords and punctuations from each row in specific (dataset) of specific (feature)
    :param dataset: Dataset to replace its column (feature) with the tokenized values
    :param feature: the feature that tokenization will apply on
    """
    vals = []
    for index, r in enumerate(dataset[feature]):
        text = r.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.casefold() not in stop_words]
        vals.append(' '.join(filtered_tokens))
    dataset[feature] = vals

"""
Plot scatter plots for each feature against the target column
target_col = 'vote_average'
feature_cols = ['runtime', 'vote_count', 'cast', 'crew', 'genres', 'keywords', 'production_companies']
for feat in feature_cols:
    plt.scatter(preprocessed_movies[feat], preprocessed_movies[target_col])
    plt.title(f"{feat} vs {target_col}")
    plt.xlabel(feat)
    plt.ylabel(target_col)
    plt.show()
"""
