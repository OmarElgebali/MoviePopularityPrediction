import pandas as pd
import numpy as np
import nltk
import json
import os
import pickle
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

################################################################################################################################
# <GLOBALS>
features_dictionary_list = [
    {'F_name': 'genres', 'Key': 'id', 'Model': []},
    {'F_name': 'keywords', 'Key': 'id', 'Model': []},
    {'F_name': 'production_companies', 'Key': 'id', 'Model': []},
    {'F_name': 'production_countries', 'Key': 'name', 'Model': []},
    {'F_name': 'spoken_languages', 'Key': 'iso_639_1', 'Model': []},
    {'F_name': 'cast', 'Key': 'name', 'Model': []},
    {'F_name': 'crew', 'Key': 'name', 'Model': []}
]
features_raw_text = {
    "title": None, "overview": None,
    "tagline": None, "original_title": None,
}
features_to_scale = {
    # Numerical Features
    "budget": None, "vote_count": None,
    "id": None, "viewercount": None,
    "revenue": None, "runtime": None,

    # dictionary_list Features
    "genres": None, "keywords": None,
    "production_companies": None, "production_countries": None,
    "spoken_languages": None, "cast": None,
    "crew": None,

    # Encoded Features
    "original_language": None, "status": None,

    # Date Features
    "release_date": None,

    # Raw Text Features
    "title": None, "overview": None,
    "tagline": None, "original_title": None,
}
features_categorical = {
    "original_language": None, "status": None
}
features_means = {
    "budget": None, "vote_count": None,
    "id": None, "viewercount": None,
    "revenue": None, "runtime": None
}
regression_models = {
    "Ridge": None,
    "Elastic_net": None,
    "Random_Forest": None,
    "Poly_features": None
}


"""
strings
list of dict
categorical_labels
numerical
"""
selected_features = ['runtime', 'viewercount', 'vote_count', 'release_date', 'cast', 'crew', 'genres', 'keywords',
                     'production_companies', 'vote_average']

folder_name = "SavedModels_Regression"
JsonFiles_features_dictionary_list = "features_dictionary_list.json"
ModelsFiles_features_raw_text = {
    "title": "features_raw_text_title.pkl",
    "overview": "features_raw_text_overview.pkl",
    "tagline": "features_raw_text_tagline.pkl",
    "original_title": "features_raw_text_original_title.pkl",
}
ModelsFiles_features_to_scale = {
    # Numerical Features
    "budget": "features_to_scale_budget.pkl",
    "vote_count": "features_to_scale_vote_count.pkl",
    "id": "features_to_scale_id.pkl",
    "viewercount": "features_to_scale_viewercount.pkl",
    "revenue": "features_to_scale_revenue.pkl",
    "runtime": "features_to_scale_runtime.pkl",

    # dictionary_list Features
    "genres": "features_to_scale_genres.pkl",
    "keywords": "features_to_scale_keywords.pkl",
    "production_companies": "features_to_scale_production_companies.pkl",
    "production_countries": "features_to_scale_production_countries.pkl",
    "spoken_languages": "features_to_scale_spoken_languages.pkl",
    "cast": "features_to_scale_cast.pkl",
    "crew": "features_to_scale_crew.pkl",

    # Encoded Features
    "original_language": "features_to_scale_original_language.pkl",
    "status": "features_to_scale_status.pkl",

    # Date Features
    "release_date": "features_to_scale_release_date.pkl",

    # Raw Text Features
    "title": "features_to_scale_title.pkl",
    "overview": "features_to_scale_overview.pkl",
    "tagline": "features_to_scale_tagline.pkl",
    "original_title": "features_to_scale_original_title.pkl",
}
ModelsFiles_features_categorical = {
    "original_language": "features_categorical_original_language.pkl",
    "status": "features_categorical_status.pkl"
}
JsonFile_features_means = "features_means.json"
ModelsFiles_regression_models = {
    "Ridge": "regression_models_Ridge.pkl",
    "Elastic_net": "regression_models_Elastic_net.pkl",
    "Random_Forest": "regression_models_Random_Forest.pkl",
    "Poly_features": "regression_models_Poly_features.pkl"
}
results = pd.DataFrame()
################################################################################################################################
# <METHODS>
def check_numerical(dataset):
    for feature in dataset.columns:
        if feature in ['budget', 'revenue', 'id', 'runtime', 'vote_count', 'viewercount']:
            dataset[feature] = dataset[feature].astype(np.float64)


def feature_calculate_mens(dataset):
    """
    Calculate the mean of the non-zero values in each feature of the numerical values
    :param dataset: dataset to calculate the mean of each feature
    """
    for key in features_means:
        features_means[key] = dataset[key][dataset[key] != 0].mean()


def feature_handling_nulls(dataset):
    # Replace null vals with mean of non-zero values
    for key in features_means:
        dataset[key].fillna(features_means[key], inplace=True)
    # Replace null strings with 0 to be removed in TFIDF
    for key in features_raw_text:
        dataset[key].fillna(0, inplace=True)
    # Replace null lists to empty lists as the transform function will handle automatically
    for key in features_dictionary_list:
        dataset[key['F_name']].fillna('[]', inplace=True)
    # Replace null categories to 'Unknown' to be mapped to 'Unknown' label
    for key in features_categorical:
        dataset[key].fillna("Unknown", inplace=True)
    # Replace null release_date to 1/1/1895 as to be the oldest movie date
    dataset['release_date'].fillna('1/1/1895', inplace=True)


def feature_scaling_train(dataset):
    for key in features_to_scale:
        min_max_model = MinMaxScaler()
        data = np.array(dataset[key].values).reshape(-1, 1)
        min_max_model.fit(data)
        features_to_scale[key] = min_max_model


def feature_scaling_transform(dataset):
    for key in features_to_scale:
        data = np.array(dataset[key]).reshape(-1, 1)
        transformed_data = features_to_scale[key].transform(data)
        dataset[key] = transformed_data


def convert_date(dataset):
    """
    converts date format to numerical value using julian_day equation
    :param dataset:  Dataset to replace its column 'release_date' with the new numerical values
    """
    dates = []
    for index, r in enumerate(dataset['release_date']):
        datetime_obj = datetime.strptime(r, '%m/%d/%Y')
        a = (14 - datetime_obj.month) // 12
        y = datetime_obj.year + 4800 - a
        m = datetime_obj.month + 12 * a - 3
        julian_day = datetime_obj.day + ((153 * m + 2) // 5) + 365 * y + (y // 4) - (y // 100) + (y // 400) - 32045
        dates.append(julian_day)
    dataset['release_date'] = dates


def convert_homepage(dataset):
    dataset['homepage_exists'] = 0
    dataset.loc[~dataset['homepage'].isnull(), 'homepage_exists'] = 1
    dataset['homepage'] = dataset['homepage_exists']
    dataset.drop(columns='homepage_exists', inplace=True)


def organize_columns(dataset):
    return dataset[[
        'id', 'profitable', 'runtime', 'viewercount', 'vote_count',
        'homepage', 'release_date',
        'original_language', 'status',
        'cast', 'crew', 'genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages',
        'overview', 'tagline', 'original_title', 'title',
        'vote_average']]


def extract_profit(dataset):
    dataset['profit'] = dataset['revenue'] - dataset['budget']
    dataset['profitable'] = (dataset['profit'] > 0).astype(int)
    dataset.drop(columns='profit', inplace=True)


def feature_encoder_train(dataset):
    """
    Fit Action => train LabelEncoder models based on (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will train from
    """
    for feature_obj in features_categorical:
        lbl_model = LabelEncoder()
        list_vals = list(dataset[feature_obj].values)
        list_vals.append('Unknown')
        lbl_model.fit(list_vals)
        features_categorical[feature_obj] = lbl_model


def feature_encoder_transform(dataset):
    """
    Transform Action => apply the trained LabelEncoder models from (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will apply on
    """
    for feature_obj in features_categorical:
        dataset[feature_obj] = [label if label in features_categorical[feature_obj].classes_ else "Unknown" for label in
                                dataset[feature_obj]]
        dataset[feature_obj] = features_categorical[feature_obj].transform(list(dataset[feature_obj].values))


def train_dict_to_list(dataset):
    def generate_possible_vales_list_based_target(df, feature_name, key):
        """
            1. Track all ratings associated with each feature in a dictionary
            2. Calculate average ratings for each feature
            3. Create and sort a list of tuples (dictionary value, key)
            4. Create a list of only the feature names, from the lowest rating to the highest rating
        :param df: the dataset that values will be extracted from
        :param key: the key to get the specific value of dictionary
        :param feature_name: the feature that will apply the extraction
        :return: a list of all unique possible feature values sorted based on target average
        """
        # (summation of all target values of same feature value, total number of feature values)
        feature_dict = {}
        for index, r in df.iterrows():
            feature_values_in_row = r[feature_name]
            for sub_value in feature_values_in_row:
                sub_value_name = sub_value[key]
                if sub_value[key] not in feature_dict:
                    if index >= df.shape[0]:
                        break
                    feature_dict[sub_value_name] = (df.iloc[index]['vote_average'], 1)
                else:
                    feature_dict[sub_value_name] = (feature_dict[sub_value_name][0] + (df['vote_average'][index]),
                                                    feature_dict[sub_value_name][1] + 1)

        # converts the tuple into average of all vote_averages
        for possible_value in feature_dict:
            feature_dict[possible_value] = feature_dict[possible_value][0] / feature_dict[possible_value][1]

        lst_possible_vals_and_votesAVG = []
        for name in feature_dict:
            lst_possible_vals_and_votesAVG.append((feature_dict[name], name))
        lst_possible_vals_and_votesAVG = sorted(lst_possible_vals_and_votesAVG)

        feature_list = []
        # ratings_list = []  # useful for plotting
        for element in lst_possible_vals_and_votesAVG:
            feature_list.append(element[1])
            # ratings_list.append(element[0])
        return feature_list

    for feature_obj in features_dictionary_list:
        dataset[feature_obj['F_name']] = dataset[feature_obj['F_name']].apply(lambda c: json.loads(c))
        feature_obj['Model'] = generate_possible_vales_list_based_target(dataset, feature_obj['F_name'],
                                                                         feature_obj['Key'])


def apply_dictionary_list_transformation(passed_dataset):
    def transform_dict_to_list(dataset, column, key, trained_list):
        """
        Transform Acton => With all possible values in (trained_list) apply on specific (dataset) at (column) of (id) to generate
        new vectorized form of current row
        :param dataset: the dataset that (trained_list) will apply on
        :param column: the feature that (trained_list) trained on to generate new vectorized form
        :param key: the key that (trained_list) trained on
        :param trained_list: list of trained possible values in feature (column) with key in dictionary (Key)
        """

        def get_column_features(row_list_dictionaries, all_possible_values):
            value_dict = {value: 0 for value in all_possible_values}
            for dictionary_in_list in row_list_dictionaries:
                a_key = dictionary_in_list[key]
                if a_key in value_dict:
                    value_dict[a_key] = 1
            return list(value_dict.values())

        feature_vectors = []
        for i, row in dataset.iterrows():
            # num_values = len(row[column])
            column_features = get_column_features(row[column], trained_list)
            # feature_vector = column_features + [num_values]
            feature_vectors.append(column_features)

        dataset[column] = np.array(feature_vectors).tolist()

    def w_avg(arr):
        weight = 0  # weight
        s = 0  # position*weight
        for element in arr:
            s += (element[0] * element[1])  # s += index * weight
            weight += element[1]
        if weight == 0:
            return 0
        return s / weight + 1  # weighted average

    def split_arr(arr, n_splits):
        # looping till length l
        for i in range(0, len(arr), n_splits):
            yield arr[i:i + n_splits]

    def find_concentration(arr, n=5):  # n is the number of concentration points to find
        # separate array into batches
        batches = list(split_arr(arr, int(len(arr) / n)))
        concentrations = []
        for i in range(len(batches)):
            point = 0
            num_ones = 0
            for j in range(len(batches[i])):
                if batches[i][j] == 1:
                    point += j + (i * int(len(arr) / n))  # adding correction for batches
                    num_ones += 1
            if num_ones > 0:
                point = point / num_ones
                concentrations.append((point, num_ones))
        return concentrations

    for feature_obj in features_dictionary_list:
        transform_dict_to_list(passed_dataset, feature_obj['F_name'], feature_obj['Key'], feature_obj['Model'])
        passed_dataset[feature_obj['F_name']] = passed_dataset[feature_obj['F_name']].apply(
            lambda x: w_avg(find_concentration(x)))


def text_to_csr_matrix_train(dataset):
    for key in features_raw_text:
        words_tokenization = []
        for index, r in enumerate(dataset[key]):
            t = nltk.word_tokenize(str(r))
            lemmatized_text = ' '.join(t)
            words_tokenization.append(lemmatized_text)
        dataset[key] = words_tokenization
        tfidf_model = TfidfVectorizer()
        tfidf_model.fit(words_tokenization)
        features_raw_text[key] = tfidf_model


def text_to_csr_matrix_transform(dataset):
    def w_avg_weighted_array(arr):
        indices = arr.indices
        data = arr.data
        product = np.dot(data, indices)
        weight = data.sum()
        if weight == 0:
            return 0
        return product / weight + 1  # weighted average

    for key in features_raw_text:
        tfidf_data = features_raw_text[key].transform(dataset[key])
        list_of_weights = []
        for row in tfidf_data:
            list_of_weights.append(w_avg_weighted_array(row))
        dataset[key] = list_of_weights


def plot_correlation(dataset):
    correlation_fit = dataset.corr()  # Get the correlation between the features
    # top_feature = corr.index[abs(corr['vote_average']) > 0.2] # Top 50% Correlation training features with the Value
    plt.subplots(figsize=(12, 8))  # Correlation plot
    # top_corr = data[corr].corr()
    sns.heatmap(correlation_fit, annot=True)
    plt.show()


def plot_correlation_with_target(dataset):
    correlation_fit = dataset.corr()[['vote_average']]  # Get the correlation between the features
    # top_feature = corr.index[abs(corr['vote_average']) > 0.2] # Top 50% Correlation training features with the Value
    # plt.subplots(figsize=(12, 8))   # Correlation plot
    # top_corr = data[corr].corr()
    sns.heatmap(correlation_fit, annot=True)
    plt.show()


def feature_selection(dataset):
    correlation_fit = dataset.corr()
    # globals()[top_feature] = correlation_fit.index[(0.22 < abs(correlation_fit['vote_average']))]
    # return dataset[globals()[top_feature]]
    # return dataset.loc[:, ['crew', 'keywords', 'genres', 'production_companies', 'runtime', 'vote_average']]


def preprocessing_train(dataset_to_train):
    """
    fit all preprocessing algorithms and models to a parameter (dataset_to_train)
    :param dataset_to_train: the dataset to fit all preprocessing models from
    """
    # (1) Convert Date to Numerical and Homepage to Boolean
    # dataset_to_train = outlier_detection(dataset_to_train)
    check_numerical(dataset_to_train)
    # (5) Filling nulls with mean
    feature_calculate_mens(dataset_to_train)
    feature_handling_nulls(dataset_to_train)
    # handling_outliers(dataset_to_train)
    convert_date(dataset_to_train)
    convert_homepage(dataset_to_train)
    # (2) Extract Profit feature
    extract_profit(dataset_to_train)
    # (3) LabelEncoding: Convert Categorical to Numerical
    feature_encoder_train(dataset_to_train)
    feature_encoder_transform(dataset_to_train)
    # (4) Transform Dictionary_List Features to Vectors
    train_dict_to_list(dataset_to_train)
    apply_dictionary_list_transformation(dataset_to_train)
    # (6) Converting Raw Text
    text_to_csr_matrix_train(dataset_to_train)
    text_to_csr_matrix_transform(dataset_to_train)
    # (7) Apply Feature Scaling
    dataset_to_train.to_csv('train2.csv', index=False)
    feature_scaling_train(dataset_to_train)
    feature_scaling_transform(dataset_to_train)
    # dset = organize_columns(dataset_to_train)
    # plot_correlation(dset)
    # plot_correlation_with_target(dset)
    # # return dset
    return organize_columns(dataset_to_train)


def preprocessing_transform(dataset_to_transform):
    """
    apply all preprocessing trained algorithms and models on a parameter (dataset_to_transform)
    :param dataset_to_transform: the dataset to apply all preprocessing trained models on
    """
    # (1) Convert Date to Numerical and Homepage to Boolean
    check_numerical(dataset_to_transform)
    # (2) Filling nulls with mean
    feature_handling_nulls(dataset_to_transform)
    # (3) Convert Date and HomePage
    convert_date(dataset_to_transform)
    convert_homepage(dataset_to_transform)
    # (4) Extract Profit feature, Reorder columns, and drop extra columns added
    extract_profit(dataset_to_transform)
    # (5) LabelEncoding: Convert Categorical to Numerical
    feature_encoder_transform(dataset_to_transform)
    # (6) Transform Dictionary_List Features to Vectors
    for feature_obj in features_dictionary_list:
        dataset_to_transform[feature_obj['F_name']] = dataset_to_transform[feature_obj['F_name']].apply(
            lambda c: json.loads(c))
    apply_dictionary_list_transformation(dataset_to_transform)
    # (7) Converting Raw Text
    for key in features_raw_text:
        words_tokenization = []
        for index, r in enumerate(dataset_to_transform[key]):
            t = nltk.word_tokenize(str(r))
            lemmatized_text = ' '.join(t)
            words_tokenization.append(lemmatized_text)
        dataset_to_transform[key] = words_tokenization
    text_to_csr_matrix_transform(dataset_to_transform)
    # (8) Apply Feature Scaling
    feature_scaling_transform(dataset_to_transform)
    return organize_columns(dataset_to_transform)


# def select_best_n_features():
#     from sklearn.feature_selection import SelectFromModel
#
#     Y = preprocessed_movies['vote_average'].values
#     X = preprocessed_movies.drop(['vote_average'], axis=1)
#     regressor = RandomForestRegressor()
#     regressor.fit(X, Y)
#
#     # select the top 5 features based on feature importance scores
#     sfm = SelectFromModel(regressor, threshold=-np.inf, max_features=13)
#     sfm.fit(X, Y)
#
#     # get the indices of the selected features
#     feature_indices = sfm.get_support(indices=True)
#
#     # get the names of the selected features
#     selected_features = [list(X.columns)[i] for i in feature_indices]
#
#     print('Selected features:', selected_features)


################################################################################################################################
# Regression Methods
def elastic_net_regression_train(dataframe):
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    # create polynomial features of degree 2
    regression_models['Poly_features'] = PolynomialFeatures(degree=2)
    X_poly = regression_models['Poly_features'].fit_transform(X)

    # perform Grid Search on Elastic Net Regression hyperparameters
    elastic_net_params = {'alpha': [0.1, 0.01, 0.001], 'l1_ratio': [0.25, 0.5, 0.75]}
    grid_search = GridSearchCV(ElasticNet(), elastic_net_params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_poly, y)

    # get best hyperparameters from Grid Search
    alpha = grid_search.best_params_['alpha']
    l1_ratio = grid_search.best_params_['l1_ratio']

    # fit the polynomial regression model with Elastic Net regularization
    regression_models['Elastic_net'] = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    regression_models['Elastic_net'].fit(X_poly, y)

    # evaluate the model using cross-validation
    scores = cross_val_score(regression_models['Elastic_net'], X_poly, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print('<Training> Elastic_net Regression MODEL score: {:.4f}'.format(
        regression_models['Elastic_net'].score(X_poly, y)))
    print('<Training> Elastic_net Regression RMSE scores: {}'.format(rmse_scores))
    print(
        '<Training> Elastic_net Regression Mean MSE: {:.4f}'.format(rmse_scores.mean() ** 2))


def elastic_net_regression_test(dataframe):
    # assign X and y variables
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    # create polynomial features of degree 2
    X_poly = regression_models['Poly_features'].transform(X)

    # evaluate the model using root mean squared error (using train set for demonstration purposes)
    y_train_pred = regression_models['Elastic_net'].predict(X_poly)
    results['Elastic_net_Y'] = y_train_pred
    print('<Testing>  Elastic_net Regression MODEL score: {:.4f}'.format(
        regression_models['Elastic_net'].score(X_poly, y)))
    print("<Testing>  Elastic_net Regression R2 score:", r2_score(y, y_train_pred))
    print('<Testing>  Elastic_net Regression MSE: {:.4f}'.format(mean_squared_error(y, y_train_pred, squared=True)))


def ridge_regression_train(dataframe):
    # assign X and y variables
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    """
        regression_models['Poly_features'] = PolynomialFeatures(degree=2) this line must be in the first model to train
    """

    # create polynomial features of degree 2
    X_poly = regression_models['Poly_features'].transform(X)

    # perform grid search to find the best hyper-parameters
    param_grid = {'alpha': [0.1, 1, 10, 0.01, 0.001, 0.0001]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_poly, y)

    # fit the Ridge regression model with best hyper-parameters
    regression_models['Ridge'] = grid_search.best_estimator_
    regression_models['Ridge'].fit(X_poly, y)

    # evaluate the model using root mean squared error (using train set for demonstration purposes)
    y_train_pred = regression_models['Ridge'].predict(X_poly)

    # Set the axis labels

    # Show the plot

    print('<Training> Ridge Regression MODEL score: {:.4f}'.format(regression_models['Ridge'].score(X_poly, y)))
    print("<Training> Ridge Regression R2 score:", r2_score(y, y_train_pred))
    print('<Training> Ridge Regression MSE: {:.4f}'.format(mean_squared_error(y, y_train_pred, squared=True)))


def ridge_regression_test(dataframe):
    # assign X and y variables
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    # create polynomial features of degree 2
    X_poly = regression_models['Poly_features'].transform(X)

    # evaluate the model using root mean squared error (using train set for demonstration purposes)
    y_train_pred = regression_models['Ridge'].predict(X_poly)

    results['Ridge_Y'] = y_train_pred
    print('<Testing>  Ridge Regression MODEL score: {:.4f}'.format(regression_models['Ridge'].score(X_poly, y)))
    print("<Testing>  Ridge Regression R2 score:", r2_score(y, y_train_pred))
    print('<Testing>  Ridge Regression MSE: {:.4f}'.format(
        mean_squared_error(y, y_train_pred, squared=True)))


def random_forest_regression_train(dataframe):
    # assign X and y variables
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    # create polynomial features of degree 2
    X_poly = regression_models['Poly_features'].transform(X)

    # Create a Random Forest Regressor instance
    regression_models['Random_Forest'] = RandomForestRegressor(random_state=123, max_depth=10, n_estimators=100)
    regression_models['Random_Forest'].fit(X_poly, y)

    y_train_pred = regression_models['Random_Forest'].predict(X_poly)
    print('<Training> Random_Forest Regression MODEL score: {:.4f}'.format(
        regression_models['Random_Forest'].score(X_poly, y)))
    print("<Training> Random_Forest Regression R2 score:", r2_score(y, y_train_pred))
    print('<Training> Random_Forest Regression Mean MSE: {:.4f}'.format(
        mean_squared_error(y, y_train_pred, squared=True)))


def random_forest_regression_test(dataframe):
    # assign X and y variables
    y = dataframe['vote_average'].values
    X = dataframe.drop(['vote_average'], axis=1).values

    # create polynomial features of degree 2
    X_poly = regression_models['Poly_features'].transform(X)

    y_train_pred = regression_models['Random_Forest'].predict(X_poly)

    results['Random_Forest_Y'] = y_train_pred
    print('<Testing>  Random_Forest Regression MODEL score: {:.4f}'.format(
        regression_models['Random_Forest'].score(X_poly, y)))
    print("<Testing>  Random_Forest Regression R2 score:", r2_score(y, y_train_pred))
    print('<Testing>  Random_Forest Regression MSE: {:.4f}'.format(mean_squared_error(y, y_train_pred, squared=True)))


################################################################################################################################
# <SAVE>
def save_model(dictionary_models, dictionary_file):
    for feature in dictionary_models.keys():
        with open(os.path.join(folder_name, dictionary_file[feature]), 'wb') as f:
            pickle.dump(dictionary_models[feature], f)


def save_train():
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # JSON files
    with open(os.path.join(folder_name, JsonFile_features_means), 'w') as f:
        json.dump(features_means, f)
    with open(os.path.join(folder_name, JsonFiles_features_dictionary_list), 'w') as f:
        json.dump(features_dictionary_list, f)
    # Pickle files
    save_model(features_categorical, ModelsFiles_features_categorical)
    save_model(regression_models, ModelsFiles_regression_models)
    save_model(features_to_scale, ModelsFiles_features_to_scale)
    save_model(features_raw_text, ModelsFiles_features_raw_text)


def load_model(dictionary_models, dictionary_file):
    for feature in dictionary_models.keys():
        with open(os.path.join(folder_name, dictionary_file[feature]), 'rb') as f:
            dictionary_models[feature] = pickle.load(f)


def load_test():
    global features_means, features_dictionary_list
    # JSON files
    with open(os.path.join(folder_name, JsonFile_features_means), 'r') as f:
        features_means = json.load(f)
    with open(os.path.join(folder_name, JsonFiles_features_dictionary_list), 'r') as f:
        features_dictionary_list = json.load(f)
    # Pickle files
    load_model(features_categorical, ModelsFiles_features_categorical)
    load_model(regression_models, ModelsFiles_regression_models)
    load_model(features_to_scale, ModelsFiles_features_to_scale)
    load_model(features_raw_text, ModelsFiles_features_raw_text)


################################################################################################################################
# <Scripts>
def train_script(save=0):
    # Read Original 2 Datasets with merging on id
    movies_dataset = pd.read_csv('movies-regression-dataset.csv')
    bonus_dataset = pd.read_csv('movies-credit-students-train.csv')
    movies = bonus_dataset.merge(movies_dataset, left_on='movie_id', right_on='id', how='left')
    movies = movies.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
    movies = movies.rename(columns={'title_y': 'title'})
    # Splitting Train (80%) & Test (20%)
    movies_train, movies_test = train_test_split(movies, test_size=0.20, shuffle=True, random_state=42)
    # Read Nulls 2 Datasets with merging on id for training Scalar on Minimum values to avoid -ve
    null_movies_dataset = pd.read_csv('null_main.csv')
    null_bonus_dataset = pd.read_csv('null_bonus.csv')
    null_movies = null_bonus_dataset.merge(null_movies_dataset, left_on='movie_id', right_on='id', how='left')
    null_movies = null_movies.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
    null_movies = null_movies.rename(columns={'title_y': 'title'})
    movies_train = pd.concat([movies_train, null_movies])
    preprocessed_movies = preprocessing_train(movies_train)
    preprocessed_movies_test = preprocessing_transform(movies_test)
    train_set = preprocessed_movies.loc[:, selected_features]
    test_set = preprocessed_movies_test.loc[:, selected_features]
    elastic_net_regression_train(train_set)
    elastic_net_regression_test(test_set)
    print('-' * 80)
    ridge_regression_train(train_set)
    ridge_regression_test(test_set)
    print('-' * 80)
    random_forest_regression_train(train_set)
    random_forest_regression_test(test_set)
    print('=' * 150)
    if save != 0:
        save_train()


def test_script(csv_name, bonus_name, load=1, df_to_csv=1, df_out_csv_name='output_classification.csv'):
    dataset_csv = pd.read_csv(csv_name)
    print(dataset_csv.shape)
    bonus_csv = pd.read_csv(bonus_name)
    dataset = bonus_csv.merge(dataset_csv, left_on='movie_id', right_on='id', how='inner')
    dataset = dataset.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
    dataset = dataset.rename(columns={'title_y': 'title'})
    print(dataset.shape)
    output = dataset.copy()
    if load != 0:
        load_test()
    preprocessed_dataset = preprocessing_transform(dataset)
    test_set = preprocessed_dataset.loc[:, selected_features]
    preprocessed_dataset.to_csv('test.csv',index=False)
    elastic_net_regression_test(test_set)
    print('-' * 80)
    ridge_regression_test(test_set)
    print('-' * 80)
    random_forest_regression_test(test_set)
    print('=' * 150)
    output['Elastic_net_Y'] = results['Elastic_net_Y']
    output['Ridge_Y'] = results['Ridge_Y']
    output['Random_Forest_Y'] = results['Random_Forest_Y']
    if df_to_csv != 0:
        output.to_csv(df_out_csv_name, index=False)



################################################################################################################################
# <MAIN>
# train_script(save=1) #train & test with saving models
# test_script('test_main.csv', 'test_bonus.csv', load=1)
test_script('movies-tas-test day 2.csv', 'credit-tas-test.csv', load=1)

################################################################################################################################
# <EXTRA (For Analysis)>
# (Extra.1) Calculate the number of null values
"""
null_counts = movies.isnull().sum()
print(null_counts)
"""
# (Extra.2) Calculate the number of empty lists in a column
"""
for feature in ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew']:
    print(feature, " #(0's): ", movies[feature].apply(lambda l: len(l) == 2).sum())
"""
# (Extra.3) Correlation Plotting
"""
plot_correlation(preprocessed_movies)
plot_correlation_with_target(preprocessed_movies)
"""
# (Extra.4) Relation between profit, Profitable and vote_average
"""
movies['profit'] = movies['revenue'] - movies['budget']
movies['profitable'] = (movies['profit'] > 0).astype(int)
money = movies[['revenue', 'budget', 'profit', 'profitable', 'vote_average']]
corr = money.corr()  # Get the correlation between the features
plt.subplots(figsize=(12, 8))  # Correlation plot
sns.heatmap(corr, annot=True)
plt.show()
"""
