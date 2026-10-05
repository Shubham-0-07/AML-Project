from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_supervised_model(X_train, y_train, model_path="model/supervised_model.pkl"):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    return report
