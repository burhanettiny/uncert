import numpy as np 
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision(ms_within, ms_between, measurements):
    sum_weighted_variances = sum((np.std(m, ddof=1) ** 2) * (len(m) - 1) for m in measurements)
    sum_degrees_freedom = sum(len(m) - 1 for m in measurements)
    
    if ms_within > ms_between and sum_degrees_freedom > 0:
        urepro = np.sqrt(sum_weighted_variances / sum_degrees_freedom)
        return urepro, True
    else:
        return np.sqrt((ms_between - ms_within) / len(measurements[0])), False

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

def main():
    st.title("Belirsizlik Hesaplama Uygulaması")
    st.caption("B. Yalçınkaya tarafından geliştirildi")
    
    uploaded_file = st.file_uploader("Excel dosyanızı yükleyin", type=["xlsx", "xls"])
    pasted_data = st.text_area("Verileri buraya yapıştırın")
    
    custom_extra_uncertainty_label = st.text_input("Ek Belirsizlik Bütçesi Etiketi", value="Ek Belirsizlik Bütçesi")
    extra_uncertainty = st.number_input(custom_extra_uncertainty_label, min_value=0.0, value=0.0, step=0.01)
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
    elif pasted_data:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    else:
        st.stop()
    
    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    
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
    intermediate_precision, is_urepro = calculate_intermediate_precision(ms_within, ms_between, measurements)
    
    relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
    relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
    relative_extra_uncertainty = extra_uncertainty / 100
    
    combined_relative_uncertainty = np.sqrt(
        relative_repeatability**2 +
        relative_intermediate_precision**2 +
        relative_extra_uncertainty**2
    )
    
    expanded_uncertainty = 2 * combined_relative_uncertainty * average_value
    relative_expanded_uncertainty = calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value)
    
    results_df = pd.DataFrame({
        "Parametre": [
            "Repeatability",
            "Intermediate Precision",
            custom_extra_uncertainty_label,
            "Combined Relative Uncertainty",
            "Relative Repeatability uncertainty",
            "Relative Intermediate Precision uncertainty",
            "Relative Ek Belirsizlik uncertainty"
        ],
        "Değer": [
            f"{repeatability:.4f}",
            f"{intermediate_precision:.4f}{'*' if is_urepro else ''}",
            f"{extra_uncertainty:.4f}",
            f"{combined_relative_uncertainty:.4f}",
            f"{relative_repeatability:.4f}",
            f"{relative_intermediate_precision:.4f}",
            f"{relative_extra_uncertainty:.4f}"
        ]
    })
    
    additional_row = pd.DataFrame({
        "Parametre": ["Ortalama Değer", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
        "Değer": [
            f"{average_value:.4f}",
            f"{expanded_uncertainty:.4f}",
            f"{relative_expanded_uncertainty:.4f}"
        ]
    })
    
    results_df = pd.concat([results_df, additional_row], ignore_index=True)
    
    st.write("Sonuçlar:")
    st.dataframe(results_df)
    
    if is_urepro:
        st.write("* Grup için MS değeri, Gruplararası MS değerinden büyük olduğundan Intermediate Precision değeri \"urepro\" hesaplanarak belirlenmiştir.")
    
    fig, ax = plt.subplots()
    df.T.plot(kind='bar', ax=ax, legend=False)
    ax.set_title("Günlük Ölçümler")
    ax.set_ylabel("Ölçüm Değerleri")
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
