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

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması",
            "subtitle": "B. Yalçınkaya tarafından geliştirildi",
            "paste": "Verileri buraya yapıştırın",
            "extra_uncertainty": "Ek Belirsizlik Bütçesi",
            "results": "Sonuçlar",
            "error_bar": "Hata Bar Grafiği",
            "daily_measurements": "Günlük Ölçüm Sonuçları",
            "average_value": "Ortalama Değer",
            "expanded_uncertainty": "Expanded Uncertainty (k=2)",
            "relative_expanded_uncertainty": "Relative Expanded Uncertainty (%)",
            "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle"
        },
        "English": {
            "title": "Uncertainty Calculation Application",
            "subtitle": "Developed by B. Yalçınkaya",
            "paste": "Paste data here",
            "extra_uncertainty": "Extra Uncertainty Budget",
            "results": "Results",
            "error_bar": "Error Bar Graph",
            "daily_measurements": "Daily Measurement Results",
            "average_value": "Average Value",
            "expanded_uncertainty": "Expanded Uncertainty (k=2)",
            "relative_expanded_uncertainty": "Relative Expanded Uncertainty (%)",
            "add_uncertainty": "Add Extra Uncertainty Budget"
        }
    }
    
    st.title(texts[language]["title"])
    st.caption(texts[language]["subtitle"])
    
    pasted_data = st.text_area(texts[language]["paste"])
    
    # Ekstra belirsizlik türlerini kullanıcıdan alıyoruz:
    extra_uncertainties = []
    st.subheader(texts[language]["add_uncertainty"])
    uncertainty_types = [
        "Partition Volume Uncertainty",
        "Pipet Uncertainty",
        "Balance Uncertainty",
        "Homogeneity Uncertainty",
        "Stability Uncertainty"
    ]
    
    for uncertainty_type in uncertainty_types:
        label = st.text_input(f"{uncertainty_type} ({texts[language]['extra_uncertainty']})", value="")
        if label:
            value = st.number_input(f"{uncertainty_type} Değeri", min_value=0.0, value=0.0, step=0.01)
            extra_uncertainties.append((label, value))
    
    if pasted_data:
        try:
            pasted_data = pasted_data.replace(',', '.')
            df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
        except Exception as e:
            st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
            st.stop()
    else:
        st.error("Lütfen veri yapıştırın!")
        st.stop()
    
    # DataFrame düzenlemesi:
    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    num_measurements_per_day = len(df)
    
    st.write("Yapıştırılan Veri:")
    st.dataframe(df, use_container_width=True)
    
    if len(measurements) > 1:
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
        intermediate_precision = calculate_intermediate_precision(ms_within, ms_between, num_measurements_per_day)
        
        relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
        relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
        
        # Ekstra belirsizlikleri hesaba katıyoruz:
        combined_extra_uncertainty = np.sqrt(sum(value ** 2 for label, value in extra_uncertainties))
        relative_extra_uncertainty = combined_extra_uncertainty / average_value if average_value != 0 else float('nan')
        
        combined_relative_uncertainty = np.sqrt(
            relative_repeatability**2 +
            relative_intermediate_precision**2 +
            relative_extra_uncertainty**2
        )
        
        expanded_uncertainty = 2 * combined_relative_uncertainty * average_value
        relative_expanded_uncertainty = calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value)
        
        results_df = pd.DataFrame({
            "Parametre": [
                "Tekrarlanabilirlik",
                "Intermediate Precision",
                *[label for label, _ in extra_uncertainties],
                "Combined Relative Uncertainty",
                "Relative Repeatability",
                "Relative Intermediate Precision",
                "Relative Ek Belirsizlik"
            ],
            "Değer": [
                f"{repeatability:.4f}",
                f"{intermediate_precision:.4f}",
                *[f"{value:.4f}" for _, value in extra_uncertainties],
                f"{combined_relative_uncertainty:.4f}",
                f"{relative_repeatability:.4f}",
                f"{relative_intermediate_precision:.4f}",
                f"{relative_extra_uncertainty:.4f}"
            ],
            "Formül": [
                "√(MS_within)",
                "√((MS_between - MS_within) / N)",
                *["(" + label + " değeri)" for label, _ in extra_uncertainties],
                "√((Relative Repeatability)² + (Relative Intermediate Precision)² + (Relative Ek Belirsizlik)²)",
                "(Repeatability / Mean)",
                "(Intermediate Precision / Mean)",
                f"({custom_extra_uncertainty_label} / 100)"
            ]
        })
        
        additional_row = pd.DataFrame({
            "Parametre": [texts[language]["average_value"], texts[language]["expanded_uncertainty"], texts[language]["relative_expanded_uncertainty"]],
            "Değer": [
                f"{average_value:.4f}",
                f"{expanded_uncertainty:.4f}",
                f"{relative_expanded_uncertainty:.4f}"
            ],
            "Formül": [
                "mean(X)",
                "Combined Relative Uncertainty × Mean × 2",
                "(Expanded Uncertainty / Mean) × 100"
            ]
        })
        
        results_df = pd.concat([results_df, additional_row], ignore_index=True)
        
        st.write(texts[language]["results"] + " Veri Çerçevesi:")
        st.dataframe(results_df)
        
        # Günlük Ölçüm Grafiği:
        fig1, ax1 = plt.subplots()
        for i, group in enumerate(measurements):
            ax1.plot(range(1, len(group) + 1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
        ax1.set_xlabel("Ölçüm Sayısı")
        ax1.set_ylabel("Değer")
        ax1.set_title(texts[language]["daily_measurements"])
        ax1.legend()
        st.pyplot(fig1)
        
        # Hata Bar Grafiği:
        fig2, ax2 = plt.subplots()
        
        # Sütun isimlerini al ve "Genel Ortalama" etiketini ekle:
        x_labels = df.columns.tolist()
        x_labels.append("Genel Ortalama")
        
        # Günlük ortalama değerleri hesapla:
        x_values = [np.mean(day) for day in measurements]
        x_values.append(np.mean([val for group in measurements for val in group]))
    
        # Medyanları hesapla (kesikli kırmızı çizgi için):
        all_measurements = [val for group in measurements for val in group]
        overall_median = np.median(all_measurements)
        overall_mean = np.mean(all_measurements)

        # Standart sapmalar:
        y_errors = [np.std(day, ddof=1) for day in measurements]
        y_errors.append(0)
        
        ax2.errorbar(x_labels, x_values, yerr=y_errors, fmt='o', capsize=5, ecolor='red', linestyle='None')

        # Ortalama çizgisi (düz siyah çizgi)
        ax2.axhline(y=overall_mean, color='black', linestyle='-', linewidth=2, label="Ortalama")

        # Medyan çizgisi (düz kesikli kırmızı çizgi - her bir x noktası için)
        ax2.axhline(y=overall_median, color='red', linestyle='--', linewidth=2, label="Medyan")

        # Grafik açıklamaları:
        st.markdown(f"**Medyan:** {overall_median:.2f} (Medyan)  <span style='color:red;'>──</span>", unsafe_allow_html=True)
        st.markdown(f"**Ortalama:** {overall_mean:.2f} (Ortalama)  <span style='color:black;'>──</span>", unsafe_allow_html=True)


        ax2.set_ylabel("Değer")
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels, rotation=90)
        ax2.set_title(texts[language]["error_bar"])
        st.pyplot(fig2)

if __name__ == "__main__":
    main()
