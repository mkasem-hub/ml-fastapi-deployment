# model/process.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

def preprocess_data(data, categorical_features, label, training=True, encoder=None, lb=None):
    X = data.drop(columns=[label])
    y = data[label]

    if training:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_encoded = encoder.fit_transform(X[categorical_features])
        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y.values).ravel()
    else:
        X_encoded = encoder.transform(X[categorical_features])
        y_encoded = lb.transform(y.values).ravel()

    X_processed = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))
    return X_processed, y_encoded, encoder, lb
