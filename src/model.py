from xgboost import XGBClassifier
import joblib

def train_model(X, y):
    model = XGBClassifier(n_estimators=200, max_depth=6)
    model.fit(X, y)
    
    joblib.dump(model, "models/xgb_model.pkl")
    return model