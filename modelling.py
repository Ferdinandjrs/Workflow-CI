import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_experiment("Latihan Credit Scoring Lokal")

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Credit_Risk_Clean.csv')
    
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, 'namadataset_preprocessing', 'Credit_Risk_Clean.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(current_dir, '..', 'Credit_Risk_Clean.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan!")

    print(f"[+] Membaca dataset dari: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop(columns=['dlq_2yrs'])
    y = df['dlq_2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    with mlflow.start_run(run_name="PROYEK_LOKAL_FIX_TOTAL"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        
        mlflow.log_metric("accuracy", acc)
        
        
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        
        print(f"[+] Kriteria 2 & 3 Sukses Total! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model()
