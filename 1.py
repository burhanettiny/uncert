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
            "uncertainty": "Belirsizlik"
        },
        "English": {
            "title": "Uncertainty Calculation Application, B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "calculate": "Calculate Results",
            "results": "Results",
            "average": "Average",
            "uncertainty": "Uncertainty"
        }
    }

    # Displaying the title based on selected language
    st.title(texts[language]["title"])

    # File upload section
    uploaded_file = st.file_uploader(texts[language]["upload"], type=["xlsx", "xls"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df)

        # Assuming the data is in the first column and is numerical
        measurements = df.iloc[:, 0].dropna()

        # Calculating the results
        avg = calculate_average(measurements)
        uncertainty = calculate_standard_uncertainty(measurements)
        repeatability = calculate_repeatability(measurements)

        # Displaying the results
        st.subheader(texts[language]["results"])
        st.write(f"{texts[language]['average']}: {avg}")
        st.write(f"{texts[language]['uncertainty']}: {uncertainty}")
        st.write(f"Repeatability: {repeatability}")

        # Plotting the measurements
        plt.figure(figsize=(8, 6))
        plt.plot(measurements, 'o-', label="Measurements")
        plt.title("Measurement Data")
        plt.xlabel("Index")
        plt.ylabel("Measurement Value")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
