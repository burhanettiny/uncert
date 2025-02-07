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
            "calculate": "Sonuçları Hesapla",
            "results": "Sonuçlar",
            "average": "Ortalama",
            "uncertainty": "Belirsizlik",
            "expanded_uncertainty": "Genişletilmiş Belirsizlik (k=2)",
            "repeatability": "Tekrarlanabilirlik",
            "error_bar": "Hata Barı Grafiği",
        },
        "English": {
            "title": "Uncertainty Calculation App, B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "calculate": "Calculate Results",
            "results": "Results",
            "average": "Average",
            "uncertainty": "Uncertainty",
            "expanded_uncertainty": "Expanded Uncertainty (k=2)",
            "repeatability": "Repeatability",
            "error_bar": "Error Bar Graph",
        }
    }
    
    t = texts[language]
    
    st.title(t["title"])
    uploaded_file = st.file_uploader(t["upload"], type=["xlsx", "xls"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("## Veriler", df)
        
        measurements = df.iloc[:, 1:].values  # İlk sütun gün isimleri olabilir, ikinci sütundan itibaren ölçümler alınır
        
        if st.button(t["calculate"]):
            avg_values = [calculate_average(day) for day in measurements]
            uncertainty_values = [calculate_standard_uncertainty(day) for day in measurements]
            repeatability_values = [calculate_repeatability(day) for day in measurements]
            
            st.write(f"## {t['results']}")
            for i, day in enumerate(df.iloc[:, 0]):
                st.write(f"### {day}")
                st.write(f"**{t['average']}:** {avg_values[i]:.4f}")
                st.write(f"**{t['uncertainty']}:** {uncertainty_values[i]:.4f}")
                st.write(f"**{t['expanded_uncertainty']}:** {uncertainty_values[i] * 2:.4f}")
                st.write(f"**{t['repeatability']}:** {repeatability_values[i]:.4f}")
            
            fig, ax = plt.subplots()
            ax.errorbar(df.iloc[:, 0], avg_values, yerr=uncertainty_values, fmt='o', capsize=5, capthick=2, ecolor='red')
            ax.set_xlabel("Days")
            ax.set_ylabel("Measurement Average")
            ax.set_title(t["error_bar"])
            st.pyplot(fig)

if __name__ == "__main__":
    main()
