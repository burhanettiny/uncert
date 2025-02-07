import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import scipy.stats as stats

def calculate_average(measurements):
    return np.mean(measurements) if len(measurements) > 0 else float('nan')

def calculate_standard_uncertainty(measurements):
    return (np.std(measurements, ddof=1) / np.sqrt(len(measurements))) if len(measurements) > 1 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def anova_one_way(measurements):
    return stats.f_oneway(*measurements) if len(measurements) > 1 else None

def calculate_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    return np.sqrt((ms_between - ms_within) / num_measurements_per_day) if ms_within and ms_between else float('nan')

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması, B. Yalçınkaya",
            "upload": "Excel dosyanızı yükleyin",
            "or": "Veya",
            "manual": "Verileri manuel girin",
            "paste_data": "Verileri doğrudan Excel'den yapıştırın",
            "calculate": "Sonuçları Hesapla",
            "results": "Sonuçlar",
            "average": "Ortalama",
            "uncertainty": "Belirsizlik",
            "repeatability": "Tekrarlanabilirlik",
            "anova": "ANOVA Testi",
            "intermediate_precision": "Intermediate Precision",
            "add_uncertainty_budget": "Belirsizlik Bütçesi Ekle",
            "expanded_uncertainty": "Genişletilmiş Belirsizlik (U, %)",
            "relative_expanded_uncertainty": "Göreceli Genişletilmiş Belirsizlik (%)"
        },
        "English": {
            "title": "Uncertainty Calculation Application, B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "or": "Or",
            "manual": "Enter data manually",
            "paste_data": "Paste data directly from Excel",
            "calculate": "Calculate Results",
            "results": "Results",
            "average": "Average",
            "uncertainty": "Uncertainty",
            "repeatability": "Repeatability",
            "anova": "ANOVA Test",
            "intermediate_precision": "Intermediate Precision",
            "add_uncertainty_budget": "Add Uncertainty Budget",
            "expanded_uncertainty": "Expanded Uncertainty (U, %)",
            "relative_expanded_uncertainty": "Relative Expanded Uncertainty (%)"
        }
    }

    st.title(texts[language]["title"])
    
    data_source = st.radio("", [texts[language]["upload"], texts[language]["manual"], texts[language]["paste_data"]])
    
    measurements = []
    
    if data_source == texts[language]["upload"]:
        uploaded_file = st.file_uploader("", type=["xlsx", "xls"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, header=None)
            st.write(df)
            measurements = [df.iloc[i, :].dropna().tolist() for i in range(min(3, len(df)))]
    elif data_source == texts[language]["paste_data"]:
        pasted_data = st.text_area("", "")
        if pasted_data:
            try:
                df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
                measurements = [df.iloc[i, :].dropna().tolist() for i in range(min(3, len(df)))]
                st.write(df)
            except Exception as e:
                st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
    else:
        st.write(texts[language]["manual"])
        data_matrix = []
        for day in range(3):
            row = []
            cols = st.columns(5)
            for col in cols:
                value = col.text_input(f"Day {day+1}", key=f"{day}-{len(row)}")
                if value:
                    try:
                        row.append(float(value))
                    except ValueError:
                        st.error("Geçersiz giriş! Lütfen sayıları doğru formatta girin.")
            data_matrix.append(row[:5])
        measurements = [row for row in data_matrix if row]
    
    if measurements:
        all_measurements = [item for sublist in measurements for item in sublist]
        avg = calculate_average(all_measurements)
        uncertainty = calculate_standard_uncertainty(all_measurements)
        repeatability = calculate_repeatability(all_measurements)
        
        anova_result = anova_one_way(measurements)
        ms_within = anova_result.statistic if anova_result else float('nan')
        ms_between = np.var([np.mean(m) for m in measurements], ddof=1) if len(measurements) > 1 else float('nan')
        intermediate_precision = calculate_intermediate_precision(ms_within, ms_between, len(measurements[0]))
        
        st.subheader(texts[language]["results"])
        st.write(f"{texts[language]['average']}: {avg}")
        st.write(f"{texts[language]['uncertainty']}: {uncertainty}")
        st.write(f"{texts[language]['repeatability']}: {repeatability}")
        st.write(f"{texts[language]['anova']}: {anova_result}")
        st.write(f"{texts[language]['intermediate_precision']}: {intermediate_precision}")
        
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
