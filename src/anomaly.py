from sklearn.ensemble import IsolationForest

def train_anomaly(X):
    model = IsolationForest(contamination=0.1)
    model.fit(X)
    return model

def predict_anomaly(model, X):
    return model.predict(X)  # -1 = anomaly