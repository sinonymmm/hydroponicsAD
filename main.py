import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    return df

def detect_anomalies(df, TDS_upper_limit, TDS_lower_limit):
    if 'TDS' not in df.columns or 'pH' not in df.columns:
        raise ValueError("Dataset harus memiliki kolom 'TDS' dan 'pH'.")

    df['anomali_ground_truth'] = (df['TDS'] < TDS_lower_limit) | (df['TDS'] > TDS_upper_limit)
    X = df.drop(columns=['anomali_ground_truth'])
    y = df['anomali_ground_truth']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_train_imputed)
    
    train_anomalies = model.predict(X_train_imputed) == -1
    test_anomalies = model.predict(X_test_imputed) == -1
    
    df_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    df_train['anomali'] = train_anomalies
    df_train['anomali_ground_truth'] = y_train.values
    
    df_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
    df_test['anomali'] = test_anomalies
    df_test['anomali_ground_truth'] = y_test.values
    
    accuracy = accuracy_score(y_test, test_anomalies) * 100
    precision = precision_score(y_test, test_anomalies) * 100
    recall = recall_score(y_test, test_anomalies) * 100
    f1 = f1_score(y_test, test_anomalies) * 100
    
    return df_train, df_test, model, accuracy, precision, recall, f1

def main_page():
    st.title("Deteksi Anomali Nutrisi Air Hidroponik")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    TDS_upper_limit = st.number_input("Batas atas TDS:", value=1200)
    TDS_lower_limit = st.number_input("Batas bawah TDS:", value=400)
    
    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        
        if 'TDS' not in df.columns:
            st.error("Dataset harus memiliki kolom 'TDS'.")
        else:
            st.write("Data Awal:")
            st.dataframe(df)
            
            try:
                df_train, df_test, model, accuracy, precision, recall, f1 = detect_anomalies(df, TDS_upper_limit, TDS_lower_limit)
            except ValueError as e:
                st.error(str(e))
                return
            
            st.write("Scatter Plot Data Train (Prediksi vs Ground Truth)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_train['pH'], df_train['TDS'], color='blue', label='Normal')
            ax.scatter(df_train[df_train['anomali']]['pH'], df_train[df_train['anomali']]['TDS'], color='red', label='Anomali (Prediksi)')
            ax.scatter(df_train[df_train['anomali_ground_truth']]['pH'], df_train[df_train['anomali_ground_truth']]['TDS'], color='orange', marker='x', label='Anomali (Ground Truth)')
            ax.legend()
            st.pyplot(fig)
            
            st.write("Scatter Plot Data Test (Prediksi vs Ground Truth)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_test['pH'], df_test['TDS'], color='blue', label='Normal')
            ax.scatter(df_test[df_test['anomali']]['pH'], df_test[df_test['anomali']]['TDS'], color='red', label='Anomali (Prediksi)')
            ax.scatter(df_test[df_test['anomali_ground_truth']]['pH'], df_test[df_test['anomali_ground_truth']]['TDS'], color='orange', marker='x', label='Anomali (Ground Truth)')
            ax.legend()
            st.pyplot(fig)
            
            st.write(f"Akurasi: {accuracy:.2f}%")
            st.write(f"Precision: {precision:.2f}%")
            st.write(f"Recall: {recall:.2f}%")
            st.write(f"F1 Score: {f1:.2f}%")

main_page()
