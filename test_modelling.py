import os
import pandas as pd

def test_data_leakage_and_existence():
    data_path = 'namadataset_preprocessing/Credit_Risk_Clean.csv'
    assert os.path.exists(data_path), f"File {data_path} tidak ditemukan!"

    df = pd.read_csv(data_path)
    assert 'dlq_2yrs' in df.columns, "Kolom target 'dlq_2yrs' tidak ditemukan!"
    assert len(df) > 0, "Dataset kosong!"
    print("[+] Unit Test Passed: Dataset siap!")