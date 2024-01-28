import os
import pickle
import pandas as pd
import numpy as np
import nltk
import json
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from seaborn import heatmap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import time
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

################################################################################################################################
# <GLOBALS>
features_dictionary_list = [
    {'F_name': 'genres', 'Key': 'id', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'keywords', 'Key': 'id', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'production_companies', 'Key': 'id', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'production_countries', 'Key': 'name', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'spoken_languages', 'Key': 'iso_639_1', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'cast', 'Key': 'name', 'Low': {}, 'Intermediate': {}, 'High': {}},
    {'F_name': 'crew', 'Key': 'name', 'Low': {}, 'Intermediate': {}, 'High': {}}
]
features_dictionary_list_gen_names = []
features_raw_text = {
    "title": None, "overview": None,
    "tagline": None, "original_title": None,
}
features_to_scale = {
    # Numerical Features
    "budget": None, "vote_count": None,
    "id": None, "viewercount": None,
    "revenue": None, "runtime": None,

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
features_selection = {
    "AdaBoost_DT": 29,
    "SVM": 16,
    "Random_Forest": 16,
    "KNN": 14
}
classes = ['High', 'Intermediate', 'Low']
Classifier_models = {}
features_selection_df = pd.DataFrame()

bar_class_models = ['AdaBoost_DT', 'SVM', 'Random_Forest', 'KNN']
bar_accuracy_scores = [time.time(), time.time(), time.time(), time.time()]
bar_training_times = [time.time(), time.time(), time.time(), time.time()]
bar_testing_times = [time.time(), time.time(), time.time(), time.time()]

folder_name = "SavedModels_Classification"
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
JsonFile_features_selection = "features_selection.json"
CSV_features_selection_df = "features_selection.csv"
ModelsFiles_Classifier_models = {
    "AdaBoost_DT": "classifier_models_AdaBoost_DT.pkl",
    "SVM": "classifier_models_SVM.pkl",
    "Random_Forest": "classifier_models_Random_Forest.pkl",
    "KNN": "classifier_models_KNN.pkl"
}

results = pd.DataFrame()

################################################################################################################################
# <METHODS>
def check_numerical(dataset):
    for feature in dataset.columns:
        if feature in ['budget', 'revenue', 'id', 'runtime', 'vote_count', 'viewercount']:
            dataset[feature] = dataset[feature].astype(np.float64)


def feature_calculate_means(dataset):
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
    global features_to_scale
    for key in features_to_scale:
        min_max_model = MinMaxScaler()
        data = np.array(dataset[key].values).reshape(-1, 1)
        min_max_model.fit(data)
        features_to_scale[key] = min_max_model


def feature_scaling_transform(dataset):
    global features_to_scale
    for key in features_to_scale:
        data = np.array(dataset[key]).reshape(-1, 1)
        transformed_data = features_to_scale[key].transform(data)
        dataset[key] = transformed_data


def feature_encoder_train(dataset):
    """
    Fit Action => train LabelEncoder models based on (features_categorical) dictionary on specific (dataset)
    :param dataset: the dataset that LabelEncoder model will train from
    """
    from sklearn.preprocessing import LabelEncoder
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


def convert_date(dataset):
    """
    converts date format to numerical value using julian_day equation
    :param dataset:  Dataset to replace its column 'release_date' with the new numerical values
    """
    from datetime import datetime
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
                       'overview', 'tagline', 'original_title', 'title'] +
                   features_dictionary_list_gen_names +
                   ['Rate']
                   ]


def extract_profit(dataset):
    dataset['profit'] = dataset['revenue'] - dataset['budget']
    dataset['profitable'] = (dataset['profit'] > 0).astype(int)
    dataset.drop(columns='profit', inplace=True)


def dict_to_three_classes_train(dataset):
    def generate_possible_vales_list_based_target(df, feature_name, key):
        """
        1. Track all target values associated with each feature in a dictionary
        2. Calculate frequency of each target value for each feature
        3. Create and sort a list of tuples (frequency, feature value)
        4. Create a list of only the feature names, from the highest frequency to the lowest frequency
        :param df: the dataset that values will be extracted from
        :param key: the key to get the specific value of dictionary
        :param feature_name: the feature that will apply the extraction
        :return: a list of all unique possible feature values sorted based on target frequency
        """
        feature_dict = {}
        for index, r in df.iterrows():
            feature_values_in_row = r[feature_name]  # list of dicts
            target_value = r['Rate']  # current class label (H, I, L)
            for sub_value in feature_values_in_row:  # each dict in list
                if index >= df.shape[0]:
                    break
                sub_value_name = sub_value[key]  # value of the passed key in current dict
                if sub_value_name not in feature_dict:
                    feature_dict[sub_value_name] = {'High': 0, 'Intermediate': 0,
                                                    'Low': 0}  # creates new value dict (H:0, I:0, L:0)
                feature_dict[sub_value_name][target_value] += 1  # increament class val (freq)

        list_classes = {
            'High': [],
            'Intermediate': [],
            'Low': []
        }
        for name, freq_dict in feature_dict.items():
            for label in classes:
                list_classes[label].append((freq_dict[label], name))

        for label in classes:
            list_classes[label] = sorted(list_classes[label])
        sorted_list_classes = {
            'High': [],
            'Intermediate': [],
            'Low': []
        }
        value_dict = {
            'High': {},
            'Intermediate': {},
            'Low': {}
        }
        for label in classes:
            sorted_list_classes[label] = [element[1] for element in list_classes[label]]
            count = 1
            for value in sorted_list_classes[label]:
                value_dict[label][value] = count
                count += 1

        return value_dict['High'], value_dict['Intermediate'], value_dict['Low']

    for feature_obj in features_dictionary_list:
        dataset[feature_obj['F_name']] = dataset[feature_obj['F_name']].apply(lambda c: json.loads(c))
        feature_obj['High'], feature_obj['Intermediate'], feature_obj[
            'Low'] = generate_possible_vales_list_based_target(dataset, feature_obj['F_name'], feature_obj['Key'])
        new_feature_high = feature_obj['F_name'] + "_high"
        new_feature_intermediate = feature_obj['F_name'] + "_intermediate"
        new_feature_low = feature_obj['F_name'] + "_low"
        for label in [new_feature_low, new_feature_intermediate, new_feature_high]:
            features_to_scale[label] = None
            ModelsFiles_features_to_scale[label] = f"features_to_scale_{label}.pkl"
            features_dictionary_list_gen_names.append(label)


def dict_to_three_classes_transform(passed_dataset):
    def transform_dict_to_three_classes(dataset, column, key, high_trained_list, intermediate_trained_list,
                                        low_trained_list):
        """
        Transform Acton => With all possible values in (trained_list) apply on specific (dataset) at (column) of (id) to generate
        new vectorized form of current row
        :param dataset: the dataset that (trained_list) will apply on
        :param column: the feature that (trained_list) trained on to generate new vectorized form
        :param key: the key that (trained_list) trained on
        :param high_trained_list: list of trained possible values in feature (column) with key in dictionary (Key) of label High
        :param intermediate_trained_list: list of trained possible values in feature (column) with key in dictionary (Key) of label Intermediate
        :param low_trained_list: list of trained possible values in feature (column) with key in dictionary (Key) of label Low
        """
        trained_dict = {
            'High': high_trained_list,
            'Intermediate': intermediate_trained_list,
            'Low': low_trained_list
        }
        """
        trained_dict = {
            'High':         {'Horror': 1,   'Drama': 2,     'Thriller': 3,  'Action': 4,    'Comedy': 5},
            'Intermediate': {'Drama': 1,    'Horror': 2,    'Action': 3,    'Comedy': 4,    'Thriller': 5},
            'Low':          {'Horror': 1,   'Thriller': 2,  'Comedy': 3,    'Drama': 4,     'Action': 5}
        }
        """

        def get_confidence_classes(row_list_dictionaries):
            confidence = {
                'High': 0,
                'Intermediate': 0,
                'Low': 0
            }
            """
            Row has ("Horror", "Action")
            confidence = {
                'High':         1 + 4,
                'Intermediate': 2 + 3,
                'Low':          1 + 5
            }
            """
            for label in classes:
                for dictionary_in_list in row_list_dictionaries:
                    value_in_key = dictionary_in_list[key]  # value_in_key ='DHorrorrama'
                    if value_in_key in trained_dict[label]:
                        confidence[label] += trained_dict[label][value_in_key]
                        #                    trained_dict['High']['Horror']

            return confidence['High'], confidence['Intermediate'], confidence['Low']

        feature_vectors_h = []
        feature_vectors_i = []
        feature_vectors_l = []
        for i, row in dataset.iterrows():
            high_val, intermediate_val, low_val = get_confidence_classes(row[column])
            feature_vectors_h.append(high_val)
            feature_vectors_i.append(intermediate_val)
            feature_vectors_l.append(low_val)

        dataset[column + "_high"] = feature_vectors_h
        dataset[column + "_intermediate"] = feature_vectors_i
        dataset[column + "_low"] = feature_vectors_l

    for feature_obj in features_dictionary_list:
        transform_dict_to_three_classes(passed_dataset, feature_obj['F_name'], feature_obj['Key'], feature_obj['High'],
                                        feature_obj['Intermediate'], feature_obj['Low'])


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


def preprocessing_train(dataset_to_train):
    """
    fit all preprocessing algorithms and models to a parameter (dataset_to_train)
    :param dataset_to_train: the dataset to fit all preprocessing models from
    """
    # (0) Convert Date to Numerical and Homepage to Boolean
    check_numerical(dataset_to_train)
    # (1) Filling nulls with mean
    feature_calculate_means(dataset_to_train)
    feature_handling_nulls(dataset_to_train)
    # (2) Convert Date to Numerical
    convert_date(dataset_to_train)
    # (3) Convert Homepage to Boolean
    convert_homepage(dataset_to_train)
    # (4) Extract Profit feature
    extract_profit(dataset_to_train)
    # (5) LabelEncoding: Convert Categorical to Numerical
    feature_encoder_train(dataset_to_train)
    feature_encoder_transform(dataset_to_train)
    # (6) Transform Dictionary_List Features to Vectors
    dict_to_three_classes_train(dataset_to_train)
    dict_to_three_classes_transform(dataset_to_train)
    # (7) Converting Raw Text to weighted TF-IDF
    text_to_csr_matrix_train(dataset_to_train)
    text_to_csr_matrix_transform(dataset_to_train)
    # (8) Apply Feature Scaling
    feature_scaling_train(dataset_to_train)
    feature_scaling_transform(dataset_to_train)
    # (9) Organize the important columns
    return organize_columns(dataset_to_train)


def preprocessing_transform(dataset_to_transform):
    """
    apply all preprocessing trained algorithms and models on a parameter (dataset_to_transform)
    :param dataset_to_transform: the dataset to apply all preprocessing trained models on
    """
    # (0) Convert Date to Numerical and Homepage to Boolean
    check_numerical(dataset_to_transform)
    # (1) Filling nulls with mean
    feature_handling_nulls(dataset_to_transform)
    # (2) Convert Date to Numerical
    convert_date(dataset_to_transform)
    # (3) Convert Homepage to Boolean
    convert_homepage(dataset_to_transform)
    # (4) Extract Profit feature
    extract_profit(dataset_to_transform)
    # (5) LabelEncoding: Convert Categorical to Numerical
    feature_encoder_transform(dataset_to_transform)
    # (6) Transform Dictionary_List Features to 3 Features : {feature_high, feature_intermediate, feature_low}
    for feature_obj in features_dictionary_list:
        dataset_to_transform[feature_obj['F_name']] = dataset_to_transform[feature_obj['F_name']].apply(
            lambda c: json.loads(c))

    dict_to_three_classes_transform(dataset_to_transform)
    # (7) Converting Raw Text to weighted TF-IDF
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
    # (9) Organize the important columns
    return organize_columns(dataset_to_transform)


def anova_train(dataset):
    from sklearn.feature_selection import f_classif
    global features_selection_df
    # Separate the target variable from the input features
    x = dataset.drop("Rate", axis=1)
    y = dataset["Rate"]
    # Compute the ANOVA F-value and p-value for each feature
    f_values, p_values = f_classif(x, y)
    # Create a dataframe to store the results
    features_selection_df = pd.DataFrame({"Feature": x.columns, "F-Value": f_values, "p-value": p_values})
    # Sort the dataframe by descending F-values
    features_selection_df.sort_values(by="F-Value", ascending=False, inplace=True)


def anova_test(n):
    # Select the top n features based on F-values
    selected_features = features_selection_df.head(n)["Feature"].tolist()
    return selected_features


################################################################################################################################
# Classifiers
def adaboost_dt_train(dataframe):
    global bar_training_times
    # Select the features for this iteration
    selected_features = anova_test(features_selection['AdaBoost_DT']) + ['Rate']
    X = dataframe[selected_features].drop(['Rate'], axis=1).values
    y = dataframe['Rate'].values
    start_time = time.time()
    param_list_dt = {'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 2,
                     'min_samples_split': 5}
    Classifier_models['AdaBoost_DT'] = AdaBoostClassifier(DecisionTreeClassifier(**param_list_dt),
                                                          algorithm="SAMME",
                                                          n_estimators=100)
    Classifier_models['AdaBoost_DT'].fit(X, y)
    bar_training_times[0] = time.time() - start_time
    # 2400x1200 px
    # plt.figure(figsize=(60, 30))
    # plot_tree(dt_classifier, filled=True)
    # plt.savefig("decision_tree.png", dpi=600, bbox_inches='tight')
    # plt.show()


def bagging_svm_train(dataframe):
    # Select the features for this iteration
    selected_features = anova_test(features_selection['SVM']) + ['Rate']
    X = dataframe[selected_features].drop(['Rate'], axis=1).values
    y = dataframe['Rate'].values

    start_time = time.time()
    param_list_svm = {'C': 100, 'gamma': 0.1, 'kernel': 'linear', 'decision_function_shape': 'ovo'}
    svm_ovo = svm.SVC(**param_list_svm)
    # Create a bagging classifier with 5 SVM models
    Classifier_models['SVM'] = BaggingClassifier(estimator=svm_ovo, n_estimators=1000)
    # Fit the bagged SVM model on the dataset
    Classifier_models['SVM'].fit(X, y)
    bar_training_times[1] = time.time() - start_time


def bagging_random_forest_train(dataframe):
    # Select the features for this iteration
    selected_features = anova_test(features_selection['Random_Forest']) + ['Rate']
    X = dataframe[selected_features].drop(['Rate'], axis=1).values
    y = dataframe['Rate'].values

    start_time = time.time()
    param_list_random_forest = {'n_estimators': 200, 'min_samples_split': 2, 'random_state': 42, 'max_depth': None}
    # Create a bagging classifier with 5 SVM models
    Classifier_models['Random_Forest'] = BaggingClassifier(estimator=RandomForestClassifier(**param_list_random_forest),
                                                           n_estimators=100,
                                                           max_samples=1.0, max_features=1.0, bootstrap=True,
                                                           bootstrap_features=False, n_jobs=-1, random_state=42)
    # Fit the bagged SVM model on the dataset
    Classifier_models['Random_Forest'].fit(X, y)
    bar_training_times[2] = time.time() - start_time


def knn_train(dataframe):
    # Select the features for this iteration
    selected_features = anova_test(features_selection['KNN']) + ['Rate']
    X = dataframe[selected_features].drop(['Rate'], axis=1).values
    y = dataframe['Rate'].values

    start_time = time.time()
    param_list_knn = {'n_neighbors': 6, 'weights': 'uniform', 'p': 1}
    Classifier_models['KNN'] = KNeighborsClassifier(**param_list_knn)
    # Fit the bagged SVM model on the dataset
    Classifier_models['KNN'].fit(X, y)
    bar_training_times[3] = time.time() - start_time


def model_test(dataframe, model, calc_time=0):
    global bar_testing_times, bar_class_models, bar_accuracy_scores
    selected_features = anova_test(features_selection[model]) + ['Rate']
    X_Test = dataframe[selected_features].drop(['Rate'], axis=1).values
    Y_Test = dataframe['Rate'].values
    start_time = time.time()
    y_pred = Classifier_models[model].predict(X_Test)
    estimated_time = time.time() - start_time
    results[model] = y_pred
    # Evaluate the performance of the classifier
    accuracy = accuracy_score(Y_Test, y_pred)
    if calc_time != 0:
        bar_testing_times[bar_class_models.index(model)] = estimated_time
        bar_accuracy_scores[bar_class_models.index(model)] = accuracy
    precision = precision_score(Y_Test, y_pred, average='macro')
    recall = recall_score(Y_Test, y_pred, average='macro')
    f1 = f1_score(Y_Test, y_pred, average='macro')
    print(f"{model} - Accuracy: {accuracy}")
    print(f"{model} - Precision : {precision}")
    print(f"{model} - Recall : {recall}")
    print(f"{model} - F1-score : {f1}")
    print("=" * 100)


def plot_time():
    # Generate bar graphs to show the results
    plt.bar(bar_class_models, bar_accuracy_scores)
    plt.title('Classification Accuracy')
    plt.show()

    plt.bar(bar_class_models, bar_training_times)
    plt.title('Total Training Time')
    plt.show()

    plt.bar(bar_class_models, bar_testing_times)
    plt.title('Total Test Time')
    plt.show()


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
    with open(os.path.join(folder_name, JsonFile_features_selection), 'w') as f:
        json.dump(features_selection, f)
    features_selection_df.to_csv(os.path.join(folder_name, CSV_features_selection_df), index=False)
    # Pickle files
    save_model(features_categorical, ModelsFiles_features_categorical)
    save_model(Classifier_models, ModelsFiles_Classifier_models)
    save_model(features_to_scale, ModelsFiles_features_to_scale)
    save_model(features_raw_text, ModelsFiles_features_raw_text)


def load_model(dictionary_models, dictionary_file):
    for feature in dictionary_models.keys():
        with open(os.path.join(folder_name, dictionary_file[feature]), 'rb') as f:
            dictionary_models[feature] = pickle.load(f)


def load_test():
    global features_means, features_dictionary_list, features_selection, features_selection_df
    # JSON files
    with open(os.path.join(folder_name, JsonFile_features_means), 'r') as f:
        features_means = json.load(f)
    with open(os.path.join(folder_name, JsonFiles_features_dictionary_list), 'r') as f:
        features_dictionary_list = json.load(f)
    with open(os.path.join(folder_name, JsonFile_features_selection), 'r') as f:
        features_selection = json.load(f)
    features_selection_df = pd.read_csv(os.path.join(folder_name, CSV_features_selection_df))
    # Pickle files
    load_model(features_categorical, ModelsFiles_features_categorical)
    load_model(Classifier_models, ModelsFiles_Classifier_models)
    load_model(features_to_scale, ModelsFiles_features_to_scale)
    load_model(features_raw_text, ModelsFiles_features_raw_text)


################################################################################################################################
# <Scripts>
def train_script(save=0, display_time=0):
    # Read Original 2 Datasets with merging on id
    movies_dataset = pd.read_csv('movies-classification-dataset.csv')
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

    anova_train(preprocessed_movies)

    adaboost_dt_train(preprocessed_movies)
    model_test(preprocessed_movies_test, 'AdaBoost_DT', calc_time=1)
    print('-' * 80)
    bagging_svm_train(preprocessed_movies)
    model_test(preprocessed_movies_test, 'SVM', calc_time=1)
    print('-' * 80)
    bagging_random_forest_train(preprocessed_movies)
    model_test(preprocessed_movies_test, 'Random_Forest', calc_time=1)
    print('-' * 80)
    knn_train(preprocessed_movies)
    model_test(preprocessed_movies_test, 'KNN', calc_time=1)
    print('=' * 150)
    if display_time != 0:
        plot_time()
        print('*' * 150)
        print(bar_class_models)
        print(bar_training_times)
        print(bar_testing_times)
        print('*' * 150)
    if save != 0:
        save_train()


def test_script_bonus(csv_name, bonus_name, load=1, df_to_csv=1, df_out_csv_name='output_classification.csv'):
    global results
    results = pd.DataFrame()
    dataset_csv = pd.read_csv(csv_name)
    bonus_csv = pd.read_csv(bonus_name)
    dataset = bonus_csv.merge(dataset_csv, left_on='movie_id', right_on='id', how='inner')
    dataset = dataset.drop(['movie_id', 'title_x'], axis=1)  # drop extra redundant columns
    dataset = dataset.rename(columns={'title_y': 'title'})
    output = dataset.copy()
    if load != 0:
        load_test()
    preprocessed_dataset = preprocessing_transform(dataset)
    for model in ['AdaBoost_DT', 'SVM', 'Random_Forest', 'KNN']:
        model_test(preprocessed_dataset, model, calc_time=0)
        output[model] = results[model]
        print('-' * 80)
    if df_to_csv != 0:
        output.to_csv(df_out_csv_name, index=False)
    print('=' * 150)


################################################################################################################################
# <MAIN>
train_script(save=1, display_time=0)

test_script_bonus('movies-tas-test day 2.csv', 'credit-tas-test.csv', load=1)
# test_script_bonus('test_main.csv', 'test_bonus.csv', load=1)

""""
# Results in Discussion

<Testing>  Elastic_net Regression MODEL score: 0.3824
<Testing>  Elastic_net Regression R2 score: 0.38240091574788415
<Testing>  Elastic_net Regression MSE: 0.4973
--------------------------------------------------------------------------------
<Testing>  Ridge Regression MODEL score: 0.3906
<Testing>  Ridge Regression R2 score: 0.3906106075050859
<Testing>  Ridge Regression MSE: 0.4907
--------------------------------------------------------------------------------
<Testing>  Random_Forest Regression MODEL score: 0.4246
<Testing>  Random_Forest Regression R2 score: 0.42462945809391384
<Testing>  Random_Forest Regression MSE: 0.4633

======================================================================================================================================================
AdaBoost_DT - Accuracy: 0.6640211640211641
AdaBoost_DT - Precision : 0.5285362153962837
AdaBoost_DT - Recall : 0.4473359973359974
AdaBoost_DT - F1-score : 0.4461602982794884
====================================================================================================
--------------------------------------------------------------------------------
SVM - Accuracy: 0.6904761904761905
SVM - Precision : 0.45643063426921154
SVM - Recall : 0.4783216783216783
SVM - F1-score : 0.46534814292035653
====================================================================================================
--------------------------------------------------------------------------------
Random_Forest - Accuracy: 0.6666666666666666
Random_Forest - Precision : 0.5586134492307128
Random_Forest - Recall : 0.450949050949051
Random_Forest - F1-score : 0.4506854917587965
====================================================================================================
--------------------------------------------------------------------------------
KNN - Accuracy: 0.6296296296296297
KNN - Precision : 0.41643518518518513
KNN - Recall : 0.41410256410256413
KNN - F1-score : 0.4035463590307025
====================================================================================================
======================================================================================================================================================
"""
