import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def calculate_average(measurements):
    return np.mean(measurements) if len(measurements) > 0 else float('nan')

def calculate_standard_uncertainty(measurements):
    return (np.std(measurements, ddof=1) / np.sqrt(len(measurements))) if len(measurements) > 1 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması, B. Yalçınkaya",
            "upload": "Excel dosyanızı yükleyin",
            "or": "Veya",
            "manual": "Verileri manuel girin",
            "enter_data": "Ölçümleri girin (virgülle ayırın)",
            "calculate": "Sonuçları Hesapla",
            "results": "Sonuçlar",
            "average": "Ortalama",
            "uncertainty": "Belirsizlik",
            "repeatability": "Tekrarlanabilirlik"
        },
        "English": {
            "title": "Uncertainty Calculation Application, B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "or": "Or",
            "manual": "Enter data manually",
            "enter_data": "Enter measurements (separated by commas)",
            "calculate": "Calculate Results",
            "results": "Results",
            "average": "Average",
            "uncertainty": "Uncertainty",
            "repeatability": "Repeatability"
        }
    }

    st.title(texts[language]["title"])
    
    # Kullanıcıdan veri alma yöntemi seçimi
    data_source = st.radio("", [texts[language]["upload"], texts[language]["manual"]])
    
    measurements = []
    
    if data_source == texts[language]["upload"]:
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write(df)
            measurements = df.iloc[:, 0].dropna().tolist()
    else:
        manual_input = st.text_area(texts[language]["enter_data"], "")
        if manual_input:
            try:
                measurements = [float(x.strip()) for x in manual_input.split(",")]
            except ValueError:
                st.error("Geçersiz giriş! Lütfen sayıları virgülle ayırarak girin.")
    
    if measurements:
        avg = calculate_average(measurements)
        uncertainty = calculate_standard_uncertainty(measurements)
        repeatability = calculate_repeatability(measurements)
        
        st.subheader(texts[language]["results"])
        st.write(f"{texts[language]['average']}: {avg}")
        st.write(f"{texts[language]['uncertainty']}: {uncertainty}")
        st.write(f"{texts[language]['repeatability']}: {repeatability}")
        
        # Veri görselleştirme
        plt.figure(figsize=(8, 6))
        plt.plot(measurements, 'o-', label="Measurements")
        plt.title("Measurement Data")
        plt.xlabel("Index")
        plt.ylabel("Measurement Value")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
