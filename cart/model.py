import joblib
import pandas as pd

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")

columns = [
    'JK', 'Jurusan_SMA', 'Pendidikan_Agama', 'Pkn', 'Bahasa_Indonesia', 'Matematika_Wajib',
    'Sejarah_Indonesia', 'Bahasa_Inggris', 'Seni_Budaya', 'Penjaskes', 'PKWu', 'Mulok',
    'Matematika_Peminatan', 'Biologi', 'Fisika', 'Kimia', 'Lintas_Minat',
    'Geografi', 'Sejarah_Minat', 'Sosiologi', 'Ekonomi', 'Hobi'
]

categorical_columns = ['JK', 'Jurusan_SMA', 'Hobi']

def predict_from_input(data_dict):
    df = pd.DataFrame([data_dict], columns=columns)
    X_encoded = encoder.transform(df)
    prediction = model.predict(X_encoded)
    return prediction[0]
