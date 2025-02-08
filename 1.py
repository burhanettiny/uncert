import numpy as np 
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision(ms_within, ms_between, measurements, num_measurements_per_day):
    if ms_within > ms_between:
        sum_weighted_variances = sum(
            (np.std(m, ddof=1) ** 2) * (len(m) - 1) for m in measurements
        )
        sum_degrees_freedom = sum(len(m) - 1 for m in measurements)
        
        if sum_degrees_freedom > 0:
            urepro = np.sqrt(sum_weighted_variances / sum_degrees_freedom) / np.sqrt(sum_degrees_freedom)
            return urepro, True
        else:
            return float('nan'), False
    else:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day), False

def main():
    st.title("Belirsizlik Hesaplama Uygulaması")
    st.caption("B. Yalçınkaya tarafından geliştirildi")
    
    uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx", "xls"])
    pasted_data = st.text_area("Verileri buraya yapıştırın")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
    elif pasted_data:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    else:
        st.stop()
    
    if df.shape[1] < 2:
        st.error("Yetersiz veri sütunu! Lütfen en az iki sütun içeren veri yükleyin.")
        st.stop()
    
    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    num_measurements_per_day = len(df)
    
    total_values = sum(len(m) for m in measurements)
    num_groups = len(measurements)
    average_value = np.mean([val for group in measurements for val in group])
    
    ss_between = sum(len(m) * (np.mean(m) - average_value) ** 2 for m in measurements)
    ss_within = sum(sum((x - np.mean(m)) ** 2 for x in m) for m in measurements)
    
    df_between = num_groups - 1
    df_within = total_values - num_groups
    
    ms_between = ss_between / df_between if df_between > 0 else float('nan')
    ms_within = ss_within / df_within if df_within > 0 else float('nan')
    
    repeatability = calculate_repeatability(ms_within)
    intermediate_precision, is_urepro = calculate_intermediate_precision(ms_within, ms_between, measurements, num_measurements_per_day)
    
    results_df = pd.DataFrame({
        "Parametre": ["Tekrarlanabilirlik", "Intermediate Precision"],
        "Değer": [f"{repeatability:.4f}", f"{intermediate_precision:.4f}{'*' if is_urepro else ''}"],
    })
    
    st.write("Sonuçlar:")
    st.dataframe(results_df)
    
    if is_urepro:
        st.write("* Grup için MS değeri, Gruplararası MS değerinden büyük olduğundan Intermediate Precision değeri \"urepro\" hesaplanarak belirlenmiştir.")

if __name__ == "__main__":
    main()
