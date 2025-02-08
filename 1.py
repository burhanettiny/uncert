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

def calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty):
    return np.sqrt(repeatability**2 + intermediate_precision**2 + extra_uncertainty**2)

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

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
    num_measurements_per_day = len(df)
    
    st.write("Yapıştırılan Veri:")
    st.dataframe(df, use_container_width=True)
    
    if len(measurements) > 1:
        total_values = sum(len(m) for m in measurements)
        num_groups = len(measurements)
        grand_mean = np.mean([val for group in measurements for val in group])
        
        ss_between = sum(len(m) * (np.mean(m) - grand_mean) ** 2 for m in measurements)
        ss_within = sum(sum((x - np.mean(m)) ** 2 for x in m) for m in measurements)
        
        df_between = num_groups - 1
        df_within = total_values - num_groups
        
        ms_between = ss_between / df_between if df_between > 0 else float('nan')
        ms_within = ss_within / df_within if df_within > 0 else float('nan')
        
        repeatability = calculate_repeatability(ms_within)
        intermediate_precision = calculate_intermediate_precision(ms_within, ms_between, num_measurements_per_day)
        combined_uncertainty = calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty)
        
        average_value = grand_mean
        expanded_uncertainty = combined_uncertainty * 2
        relative_expanded_uncertainty = calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value)

        # Relative Repeatability ve Relative Intermediate Precision hesaplaması
        relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
        relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
        
        # Relative Extra Uncertainty yüzde hesaplaması
        relative_extra_uncertainty = (extra_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

        # Sonuçlar Veri Çerçevesi
        results_df = pd.DataFrame({
            "Parametre": ["Tekrarlanabilirlik", "Intermediate Precision", "Ekstra Belirsizlik Bütçesi", "Combined Relative Uncertainty", "Relative Repeatability", "Relative Intermediate Precision", "Relative Extra Uncertainty"],
            "Değer": [f"{repeatability:.1f}", f"{intermediate_precision:.1f}", f"{extra_uncertainty:.1f}", f"{combined_uncertainty:.1f}", f"{relative_repeatability:.1f}", f"{relative_intermediate_precision:.1f}", f"{relative_extra_uncertainty:.1f}"],
            "Formül": ["√(MS_within)", "√(MS_between - MS_within) / N", "Extra Uncertainty", "√(Repeatability² + Intermediate Precision² + Extra Uncertainty²)", "(Repeatability / Mean)", "(Intermediate Precision / Mean)", "(Extra Uncertainty / Mean) × 100"]
        })
        
        # Ortalama, Expanded Uncertainty ve Relative Expanded Uncertainty'yi eklemek
        additional_row = pd.DataFrame({
            "Parametre": ["Ortalama Değer", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
            "Değer": [f"{average_value:.1f}", f"{expanded_uncertainty:.1f}", f"{relative_expanded_uncertainty:.1f}"],
            "Formül": ["mean(X)", "Combined Uncertainty × 2", "(Expanded Uncertainty / Mean) × 100"]
        })
        
        # Yeni satırı sonuçlar veri çerçevesine ekleyelim
        results_df = pd.concat([results_df, additional_row], ignore_index=True)
        
        st.write("Sonuçlar Veri Çerçevesi:")
        st.dataframe(results_df)
        
        fig, ax = plt.subplots()
        x_labels = ["1. Gün", "2. Gün", "3. Gün", "Ortalama"]
        x_values = [np.mean(day) for day in measurements] + [average_value]
        y_errors = [np.std(day, ddof=1) for day in measurements] + [combined_uncertainty]
        ax.errorbar(x_labels, x_values, yerr=y_errors, fmt='o', capsize=5, ecolor='red', linestyle='None')
        ax.set_ylabel("Değer")
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_title(texts[language]["error_bar"])
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        for i, group in enumerate(measurements):
            ax.plot(range(1, len(group) + 1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
        ax.set_xlabel("Ölçüm Sayısı")
        ax.set_ylabel("Değer")
        ax.set_title(texts[language]["daily_measurements"])
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
