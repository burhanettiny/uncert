import numpy as np
import streamlit as st
import pandas as pd
import scipy.stats as stats

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within)

def calculate_intermediate_precision(ms_within, ms_between):
    return np.sqrt(ms_between - ms_within)

def calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty):
    return np.sqrt(repeatability**2 + intermediate_precision**2 + extra_uncertainty**2)

def main():
    st.title("Belirsizlik Hesaplama Uygulaması")
    
    uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx", "xls"])
    extra_uncertainty = st.number_input("Ekstra Belirsizlik Bütçesi", min_value=0.0, value=0.0, step=0.01)
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
        st.write("Yüklenen Veri:", df)
        measurements = [df.iloc[i, :].dropna().tolist() for i in range(min(3, len(df)))]
        
        if len(measurements) > 1:
            anova_result = stats.f_oneway(*measurements)
            ms_within = anova_result.statistic if anova_result else float('nan')
            ms_between = np.var([np.mean(m) for m in measurements], ddof=1) if len(measurements) > 1 else float('nan')
            
            repeatability = calculate_repeatability(ms_within)
            intermediate_precision = calculate_intermediate_precision(ms_within, ms_between)
            combined_uncertainty = calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty)
            
            average_value = np.mean([item for sublist in measurements for item in sublist])
            expanded_uncertainty = combined_uncertainty * 2
            relative_expanded_uncertainty = (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')
            
            results_df = pd.DataFrame({
                "Parametre": ["Ortalama Değer", "Tekrarlanabilirlik", "Intermediate Precision", "Combined Relative Uncertainty", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
                "Değer": [average_value, repeatability, intermediate_precision, combined_uncertainty, expanded_uncertainty, relative_expanded_uncertainty]
            })
            
            st.subheader("Sonuçlar")
            st.table(results_df)

if __name__ == "__main__":
    main()
