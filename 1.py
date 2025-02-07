import numpy as np
import streamlit as st
import pandas as pd
import scipy.stats as stats
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision(ms_within, ms_between):
    return np.sqrt(ms_between - ms_within) if ms_between > ms_within else float('nan')

def calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty):
    return np.sqrt(repeatability**2 + intermediate_precision**2 + extra_uncertainty**2)

def main():
    st.title("Belirsizlik Hesaplama Uygulaması")
    
    uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx", "xls"])
    pasted_data = st.text_area("Verileri buraya yapıştırın")
    extra_uncertainty = st.number_input("Ekstra Belirsizlik Bütçesi", min_value=0.0, value=0.0, step=0.01)
    
    measurements = []
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
    elif pasted_data:
        try:
            df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
        except Exception as e:
            st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
            return
    else:
        return
    
    df.columns = ["1. Gün", "2. Gün", "3. Gün"]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    
    st.write("Yapıştırılan Veri:")
    st.dataframe(df)
    
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
        intermediate_precision = calculate_intermediate_precision(ms_within, ms_between)
        combined_uncertainty = calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty)
        
        average_value = grand_mean
        expanded_uncertainty = combined_uncertainty * 2
        relative_expanded_uncertainty = (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')
        
        results_df = pd.DataFrame({
            "Parametre": ["Ortalama Değer", "Tekrarlanabilirlik", "Intermediate Precision", "Combined Relative Uncertainty", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
            "Değer": [f"{average_value:.1f}", f"{repeatability:.1f}", f"{intermediate_precision:.1f}", f"{combined_uncertainty:.1f}", f"{expanded_uncertainty:.1f}", f"{relative_expanded_uncertainty:.1f}"],
            "Formül": ["mean(X)", "√(MS_within)", "√(MS_between - MS_within)", "√(Repeatability² + Intermediate Precision² + Extra Uncertainty²)", "Combined Uncertainty × 2", "(Expanded Uncertainty / Mean) × 100"]
        })
        
        st.subheader("Sonuçlar")
        st.table(results_df)
        
        fig, ax = plt.subplots()
        x_labels = ["1. Gün", "2. Gün", "3. Gün", "Ortalama"]
        x_values = [np.mean(day) for day in measurements] + [average_value]
        y_errors = [np.std(day, ddof=1) for day in measurements] + [combined_uncertainty]
        ax.errorbar(x_labels, x_values, yerr=y_errors, fmt='o', capsize=5, ecolor='red', linestyle='None')
        ax.set_ylabel("Değer")
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_title("Hata Bar Grafiği")
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        for i, group in enumerate(measurements):
            ax.plot(group, marker='o', linestyle='-', label=f"Gün {i+1}")
        ax.set_xlabel("Ölçüm Numarası")
        ax.set_ylabel("Değer")
        ax.set_title("Günlük Ölçüm Sonuçları")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
