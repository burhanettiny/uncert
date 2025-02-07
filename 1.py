import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision(ms_within, ms_between):
    return np.sqrt(ms_between - ms_within) if ms_between > ms_within else float('nan')

def calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty):
    return np.sqrt(repeatability**2 + intermediate_precision**2 + extra_uncertainty**2)

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    texts = {
        "Türkçe": {"title": "Belirsizlik Hesaplama Uygulaması", "subtitle": "B. Yalçınkaya tarafından geliştirildi", "upload": "Excel dosyanızı yükleyin", "paste": "Verileri buraya yapıştırın", "extra_uncertainty": "Ekstra Belirsizlik Bütçesi", "results": "Sonuçlar", "error_bar": "Hata Bar Grafiği", "daily_measurements": "Günlük Ölçüm Sonuçları"},
        "English": {"title": "Uncertainty Calculation Application", "subtitle": "Developed by B. Yalçınkaya", "upload": "Upload your Excel file", "paste": "Paste data here", "extra_uncertainty": "Extra Uncertainty Budget", "results": "Results", "error_bar": "Error Bar Graph", "daily_measurements": "Daily Measurement Results"}
    }
    
    st.title(texts[language]["title"])
    st.caption(texts[language]["subtitle"])
    uploaded_file = st.file_uploader(texts[language]["upload"], type=["xlsx", "xls"])
    pasted_data = st.text_area(texts[language]["paste"])
    extra_uncertainty = st.number_input(texts[language]["extra_uncertainty"], min_value=0.0, value=0.0, step=0.01)
    
    measurements = []
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
    elif pasted_data:
        try:
            df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
        except Exception as e:
            st.error(f"Error! Please paste data in the correct format. ({str(e)})")
            return
    else:
        return
    
    df.columns = ["1. Gün", "2. Gün", "3. Gün"]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    
    st.write("Yapıştırılan Veri:")
    st.dataframe(df, use_container_width=True)
    
    if len(measurements) > 1:
        total_values = sum(len(m) for m in measurements)
        num_groups = len(measurements)
        grand_mean = np.mean([val for group in measurements for val in group])
        
        ss_between = sum(le
