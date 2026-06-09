import joblib
from sklearn.preprocessing import StandardScaler

def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_scaled

def transform_scaler(X):
    scaler = joblib.load("models/scaler.pkl")
    return scaler.transform(X)