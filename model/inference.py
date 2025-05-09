import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, X):
    return model.predict(X)

def inference(model, X):
    return model.predict(X)