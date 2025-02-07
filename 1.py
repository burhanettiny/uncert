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
            "enter_data": "Ölçümleri girin (Her satır bir gün, her hücre bir analiz sonucu olacak şekilde girin)",
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
            "enter_data": "Enter measurements (Each row represents a day, each cell represents an analysis result)",
            "calculate": "Calculate Results",
            "results": "Results",
            "average": "Average",
            "uncertainty": "Uncertainty",
            "repeatability": "Repeatability"
        }
    }

    st.title(texts[language]["title"])
    
    data_source = st.radio("", [texts[language]["upload"], texts[language]["manual"]])
    
    measurements = []
    
    if data_source == texts[language]["upload"]:
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, header=None)
            st.write(df)
            measurements = [df.iloc[i, :].dropna().tolist() for i in range(min(3, len(df)))]
    else:
        manual_input = st.text_area(texts[language]["enter_data"], "")
        if manual_input:
            try:
                rows = manual_input.strip().split("\n")
                measurements = [list(map(float, row.split(",")))[:5] for row in rows[:3]]
            except ValueError:
                st.error("Geçersiz giriş! Lütfen her satırda en fazla 5 sayı olacak şekilde virgülle ayırarak girin.")
    
    if measurements:
        all_measurements = [item for sublist in measurements for item in sublist]
        avg = calculate_average(all_measurements)
        uncertainty = calculate_standard_uncertainty(all_measurements)
        repeatability = calculate_repeatability(all_measurements)
        
        st.subheader(texts[language]["results"])
        st.write(f"{texts[language]['average']}: {avg}")
        st.write(f"{texts[language]['uncertainty']}: {uncertainty}")
        st.write(f"{texts[language]['repeatability']}: {repeatability}")
        
        plt.figure(figsize=(8, 6))
        for i, day_measurements in enumerate(measurements):
            plt.plot(day_measurements, 'o-', label=f"Day {i+1}")
        plt.title("Measurement Data")
        plt.xlabel("Index")
        plt.ylabel("Measurement Value")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
