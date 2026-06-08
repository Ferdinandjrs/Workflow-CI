import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_experiment("Latihan Credit Scoring Lokal")


mlflow.sklearn.autolog()

def train_model():
    data_path = 'namadataset_preprocessing/Credit_Risk_Clean.csv'
    df = pd.read_csv(data_path)

    X = df.drop(columns=['dlq_2yrs'])
    y = df['dlq_2yrs']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="baseline_random_forest_local"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"[+] Training Lokal Sukses! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
