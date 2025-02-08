import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return float('nan')

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması",
            "subtitle": "B. Yalçınkaya tarafından geliştirildi",
            "upload": "Excel dosyanızı yükleyin",
            "paste": "Verileri buraya yapıştırın",
            "extra_uncertainty": "Ek Belirsizlik Bütçesi",
            "results": "Sonuçlar",
            "error_bar": "Hata Bar Grafiği",
            "daily_measurements": "Günlük Ölçüm Sonuçları"
        },
        "English": {
            "title": "Uncertainty Calculation Application",
            "subtitle": "Developed by B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "paste": "Paste data here",
            "extra_uncertainty": "Extra Uncertainty Budget",
            "results": "Results",
            "error_bar": "Error Bar Graph",
            "daily_measurements": "Daily Measurement Results"
        }
    }
    
    st.title(texts[language]["title"])
   
