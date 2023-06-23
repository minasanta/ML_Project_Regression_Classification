import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def preprossing_the_test(data: pd.DataFrame):
    # Fill null values
    data = fullna(data)
    data['year'] = pd.to_datetime(data['release_date']).dt.year
    data['month'] = pd.to_datetime(data['release_date']).dt.month
    data['day'] = pd.to_datetime(data['release_date']).dt.day
    data.drop(['release_date'], axis=1, inplace=True)
    # convert it to dicts
    data = convert_the_dict_to_values_test(
        data, ("genres", "keywords", "production_companies", "production_countries", "spoken_languages"))
    # one hot encoding
    data = one_hot_encoding(data, "genres")
    data = one_hot_encoding(data, "production_countries")
    data = one_hot_encoding(data, "production_companies")
    data = one_hot_encoding(data, "keywords")
    data = one_hot_encoding(data, "spoken_languages")
    # label encoder
    data = Feature_Encoder_on_test(
        data, ("original_language", "original_title", "status"))
    # TF-IDF
    data = TF_IDF_encoder_test(data)

    # scalling the test numeric
    data = scalling(data, ["budget", "viewercount",
                    "revenue", "runtime", "vote_count", "year", "month", "day"])

    data = equal_features(data)
    return data


def fullna(data):
    data["budget"].fillna(35503348.905030556, inplace=True)
    data["viewercount"].fillna(24.70429396897039, inplace=True)
    data["revenue"].fillna(98787598.4377057, inplace=True)
    data["runtime"].fillna(109.80903104421449, inplace=True)
    data["vote_count"].fillna(805.9985895627644, inplace=True)
    data["release_date"].fillna("15/8/2002", inplace=True)

    data["genres"].fillna('[{"id": 18, "name": "Drama"}]', inplace=True)
    data["keywords"].fillna("[]", inplace=True)
    data["original_language"].fillna("en", inplace=True)
    data["original_title"].fillna("Nothing", inplace=True)
    data["overview"].fillna(" ", inplace=True)
    data["tagline"].fillna(" ", inplace=True)
    data["production_companies"].fillna("[]", inplace=True)
    data["production_countries"].fillna(
        '[{"iso_3166_1": "US", "name": "United States of America"}]', inplace=True)
    data["spoken_languages"].fillna(
        '[{"iso_639_1": "en", "name": "English"}]', inplace=True)
    data["status"].fillna("Released", inplace=True)
    return data


def convert_the_dict_to_values_test(data, cols):
    for X in cols:
        data[X] = data[X].apply(lambda x: ast.literal_eval(x))
        data[X] = [", ".join([d["name"] for d in lst]) for lst in data[X]]
    return data


def one_hot_encoding(data, c):
    mlb = MultiLabelBinarizer()
    new_feat = data[c].astype(str).str.split(", ")
    file = open(f"{c}.obj", 'rb')
    mlb = pickle.load(file)
    file.close()
    new_feat = pd.DataFrame(mlb.transform(new_feat), columns=mlb.classes_)
    data = pd.concat([data.reset_index(drop=True),
                      new_feat.reset_index(drop=True)], axis=1)
    data.drop(c, axis=1, inplace=True)
    return data


def Feature_Encoder_on_test(X, cols):
    enc = OrdinalEncoder()
    for c in cols:
        file = open(f"{c}.obj", 'rb')
        enc = pickle.load(file)
        enc.handle_unknown = "use_encoded_value"
        enc.unknown_value = 0
        file.close()
        X[c] = enc.transform(X[c].astype(str).values.reshape(-1, 1))
    return X


def TF_IDF_encoder_test(X):
    tfidf_vect = TfidfVectorizer()
    file = open(f"overview.obj", 'rb')
    tfidf_vect = pickle.load(file)
    file.close()
    try:
        xtrain_tfidf = tfidf_vect.transform(X["overview"])
        new_feat = pd.DataFrame(
            xtrain_tfidf.toarray(), columns=tfidf_vect.get_feature_names())
        X = pd.concat([X.reset_index(drop=True),
                       new_feat.reset_index(drop=True)], axis=1)
        X.drop("overview", inplace=True, axis=1)
    except:
        X.drop("overview", inplace=True, axis=1)
    return X


def scalling(data, cols):
    scaller = MinMaxScaler()
    filehandler = open(f"scaling.obj", "rb")
    scaller = pickle.load(filehandler)
    filehandler.close()
    data.loc[:, cols] = scaller.transform(data.loc[:, cols])
    return data


def equal_features(data):
    with open('columns_of_train.txt', 'r', encoding='utf-8') as f:
        contents = f.read()
    columns_of_training = list(contents.split(", "))

    columns_of_test = list(data.columns)
    data.drop(list(set(columns_of_test) - set(columns_of_training)),
              inplace=True, axis=1)

    if (len(data.columns) > len(columns_of_training)):
        i = 0
        while ((i < len(columns_of_training) and i < len(data.columns)) and len(columns_of_training) != len(data.columns)):
            if columns_of_training[i] != data.columns[i]:
                data.drop(data.columns[i], axis=1, inplace=True)
            else:
                i += 1
    if (len(columns_of_training) > len(data.columns)):
        add_to_test = list(set(columns_of_training) - set(columns_of_test))
        for f in add_to_test:
            data[f] = 0
    return data
