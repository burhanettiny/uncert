import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calculate_average(measurements):
    return np.mean(measurements) if measurements else float('nan')

def calculate_standard_uncertainty(measurements):
    return (np.std(measurements, ddof=1) / np.sqrt(len(measurements))) if len(measurements) > 1 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return float('nan')

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

# ------------------------
# Elle Giriş Modu
# ------------------------
def run_manual_mode():
    st.header("Elle Veri Girişi Modu")
    days = ['1. Gün', '2. Gün', '3. Gün']
    total_measurements = []
    uncertainty_components = []

    for day in days:
        st.subheader(f"{day} İçin Ölçüm Sonucu Girin")
        measurements = []
        for i in range(5):
            value = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

        uncertainty_component = st.number_input(f"{day} İçin Ekstra Belirsizlik Bileşeni (Opsiyonel)", value=0.0, step=0.01, format="%.4f", key=f"unc_{day}")
        uncertainty_components.append(uncertainty_component)

    if st.button("Sonuçları Hesapla (Elle Giriş)"):
        repeatability_values = []
        for i, day in enumerate(days):
            avg = calculate_average(total_measurements[i])
            uncertainty = calculate_standard_uncertainty(total_measurements[i])
            expanded_uncertainty = uncertainty * 2
            repeatability = calculate_repeatability(total_measurements[i])
            total_uncertainty = np.sqrt(uncertainty**2 + uncertainty_components[i]**2)
            expanded_total_uncertainty = total_uncertainty * 2

            st.write(f"### {day} Sonuçları")
            st.write(f"**Ortalama:** {avg:.4f}")
            st.write(f"**Belirsizlik:** {total_uncertainty:.4f}")
            st.write(f"**Genişletilmiş Belirsizlik (k=2):** {expanded_total_uncertainty:.4f}")
            st.write(f"**Tekrarlanabilirlik:** {repeatability:.4f}")
            repeatability_values.extend(total_measurements[i])

        overall_measurements = [value for day in total_measurements for value in day]
        overall_avg = calculate_average(overall_measurements)
        overall_uncertainty = calculate_standard_uncertainty(overall_measurements)
        expanded_overall_uncertainty = overall_uncertainty * 2
        repeatability_within_days = calculate_repeatability(repeatability_values)
        repeatability_between_days = calculate_repeatability([calculate_average(day) for day in total_measurements])

        st.write("## Genel Sonuçlar")
        st.write(f"**Genel Ortalama:** {overall_avg:.4f}")
        st.write(f"**Günler Arası Tekrarlanabilirlik:** {repeatability_between_days:.4f}")
        st.write(f"**Güç İçi Tekrarlanabilirlik:** {repeatability_within_days:.4f}")
        st.write(f"**Genel Belirsizlik:** {overall_uncertainty:.4f}")
        st.write(f"**Genişletilmiş Genel Belirsizlik (k=2):** {expanded_overall_uncertainty:.4f}")

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode():
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

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=1, max_value=10, value=1, step=1)
    extra_uncertainties = []
    st.subheader(texts[language]["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı ({texts[language]['extra_uncertainty']})", value="")
        if label:
            value = st.number_input(f"Ekstra Belirsizlik {i+1} Değeri", min_value=0.0, value=0.0, step=0.01)
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

        repeatability = calc_repeatability_from_ms(ms_within)
        intermediate_precision = calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day)
        relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
        relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
        combined_extra_uncertainty = np.sqrt(sum(value ** 2 for label, value in extra_uncertainties))
        relative_extra_uncertainty = combined_extra_uncertainty / average_value if average_value != 0 else float('nan')
        combined_relative_uncertainty = np.sqrt(
            relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_uncertainty**2
        )
        expanded_uncertainty = 2 * combined_relative_uncertainty * average_value
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_uncertainty, average_value)

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
            ]
        })
        additional_row = pd.DataFrame({
            "Parametre": [texts[language]["average_value"], texts[language]["expanded_uncertainty"], texts[language]["relative_expanded_uncertainty"]],
            "Değer": [
                f"{average_value:.4f}",
                f"{expanded_uncertainty:.4f}",
                f"{relative_expanded_uncertainty:.4f}"
            ]
        })
        results_df = pd.concat([results_df, additional_row], ignore_index=True)
        st.write(texts[language]["results"])
        st.dataframe(results_df)

        # Günlük Grafik
        fig1, ax1 = plt.subplots()
        for i, group in enumerate(measurements):
            ax1.plot(range(1, len(group) + 1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
        ax1.set_xlabel("Ölçüm Sayısı")
        ax1.set_ylabel("Değer")
        ax1.set_title(texts[language]["daily_measurements"])
        ax1.legend()
        st.pyplot(fig1)

        # Hata Bar Grafiği
        fig2, ax2 = plt.subplots()
        x_labels = df.columns.tolist()
        x_labels.append("Genel Ortalama")
        x_values = [np.mean(day) for day in measurements]
        x_values.append(np.mean([val for group in measurements for val in group]))
        y_errors = [np.std(day, ddof=1) for day in measurements]
        y_errors.append(0)
        overall_median = np.median([val for group in measurements for val in group])
        overall_mean = np.mean([val for group in measurements for val in group])
        ax2.errorbar(x_labels, x_values, yerr=y_errors, fmt='o', capsize=5, ecolor='red', linestyle='None')
        ax2.axhline(y=overall_mean, color='black', linestyle='-', linewidth=2, label="Ortalama")
        ax2.axhline(y=overall_median, color='red', linestyle='--', linewidth=2, label="Medyan")
        ax2.set_ylabel("Değer")
        ax2.set_xticks(range(len(x_labels)))
        ax2.set_xticklabels(x_labels, rotation=90)
        ax2.set_title(texts[language]["error_bar"])
        st.pyplot(fig2)
        st.markdown(f"**Medyan:** {overall_median:.2f}  <span style='color:red;'>──</span>", unsafe_allow_html=True)
        st.markdown(f"**Ortalama:** {overall_mean:.2f}  <span style='color:black;'>──</span>", unsafe_allow_html=True)

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    mode = st.radio("Veri Giriş Yöntemi / Data Input Method", ["Elle Giriş", "Yapıştırarak Giriş"])
    if mode == "Elle Giriş":
        run_manual_mode()
    else:
        run_paste_mode()

if __name__ == "__main__":
    main()
