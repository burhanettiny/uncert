import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# ------------------------
# Dil Metinleri
# ------------------------
languages = {
    "Türkçe": {
        "manual_header": "Elle Veri Girişi Modu",
        "manual_subheader": "{} İçin Ölçüm Sonucu Girin",
        "extra_uncert_label": "Ekstra Belirsizlik Bütçesi",
        "extra_uncert_count": "Ekstra Belirsizlik Bütçesi Sayısı",
        "extra_uncert_type": "{} için tür seçin",
        "absolute": "Mutlak Değer",
        "percent": "Yüzde",
        "calculate_button": "Sonuçları Hesapla (Elle Giriş)",
        "overall_results": "Genel Sonuçlar",
        "average": "Genel Ortalama",
        "repeatability_within": "Güç İçi Tekrarlanabilirlik",
        "repeatability_between": "Günler Arası Tekrarlanabilirlik",
        "combined_relative_unc": "Combined Relative Ek Belirsizlik",
        "expanded_uncertainty": "Genişletilmiş Genel Belirsizlik (k=2)",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "average_value": "Ortalama Değer",
        "expanded_uncertainty_col": "Expanded Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle"
    },
    "English": {
        "manual_header": "Manual Input Mode",
        "manual_subheader": "Enter Measurements for {}",
        "extra_uncert_label": "Extra Uncertainty Budget",
        "extra_uncert_count": "Number of Extra Uncertainty Budgets",
        "extra_uncert_type": "Select type for {}",
        "absolute": "Absolute Value",
        "percent": "Percent",
        "calculate_button": "Calculate Results (Manual Input)",
        "overall_results": "Overall Results",
        "average": "Average Value",
        "repeatability_within": "Repeatability Within Days",
        "repeatability_between": "Repeatability Between Days",
        "combined_relative_unc": "Combined Relative Extra Uncertainty",
        "expanded_uncertainty": "Expanded Overall Uncertainty (k=2)",
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste data here (columns = days)",
        "results": "Results",
        "daily_measurements": "Daily Measurement Results",
        "average_value": "Average Value",
        "expanded_uncertainty_col": "Expanded Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "add_uncertainty": "Add Extra Uncertainty Budget"
    }
}

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
def run_manual_mode(lang_texts):
    st.header(lang_texts["manual_header"])
    days = ['1. Gün', '2. Gün', '3. Gün']
    total_measurements = []

    for day in days:
        st.subheader(lang_texts["manual_subheader"].format(day))
        measurements = []
        for i in range(5):
            value = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["extra_uncert_label"])
    overall_measurements = [val for day in total_measurements for val in day]
    overall_avg = calculate_average(overall_measurements) if overall_measurements else 1

    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Extra Uncertainty {i+1} Name", value="", key=f"manual_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"manual_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Value", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Percent (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value))

    if st.button(lang_texts["calculate_button"]):
        repeatability_values = []
        for i, day in enumerate(days):
            avg = calculate_average(total_measurements[i])
            uncertainty = calculate_standard_uncertainty(total_measurements[i])
            repeatability = calculate_repeatability(total_measurements[i])
            total_uncertainty = np.sqrt(uncertainty**2 + sum([rel[1]**2 for rel in extra_uncertainties]))
            st.write(f"### {day} Results")
            st.write(f"**Average:** {avg:.4f}")
            st.write(f"**Uncertainty (incl. extra):** {total_uncertainty:.4f}")
            st.write(f"**Repeatability:** {repeatability:.4f}")
            repeatability_values.extend(total_measurements[i])

        overall_measurements = [val for day in total_measurements for val in day]
        overall_avg = calculate_average(overall_measurements)
        repeatability_within_days = calculate_repeatability(repeatability_values)
        repeatability_between_days = calculate_repeatability([calculate_average(day) for day in total_measurements])
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
        combined_relative_unc = np.sqrt((repeatability_within_days/overall_avg)**2 + (repeatability_between_days/overall_avg)**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * overall_avg

        st.write(f"## {lang_texts['overall_results']}")
        st.write(f"**{lang_texts['average']}:** {overall_avg:.4f}")
        st.write(f"**{lang_texts['repeatability_between']}:** {repeatability_between_days:.4f}")
        st.write(f"**{lang_texts['repeatability_within']}:** {repeatability_within_days:.4f}")
        st.write(f"**{lang_texts['combined_relative_unc']}:** {relative_extra_unc:.4f}")
        st.write(f"**{lang_texts['expanded_uncertainty']}:** {expanded_overall_uncertainty:.4f}")

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])

    if not pasted_data:
        st.error("Please paste data!")
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Error! ({str(e)})")
        st.stop()

    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]

    measurements = []
    for col in df.columns:
        group = []
        for val in df[col]:
            try:
                group.append(float(val))
            except:
                continue
        measurements.append(group)

    all_values = [val for group in measurements for val in group]
    overall_avg = np.mean(all_values) if all_values else 1

    # Ekstra Belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Extra Uncertainty {i+1} Name", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Value", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Percent (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value))

    # Hesaplamalar
    total_values = sum(len(m) for m in measurements)
    num_groups = len(measurements)
    average_value = np.mean(all_values)
    ss_between = sum(len(m) * (np.mean(m) - average_value) ** 2 for m in measurements)
    ss_within = sum(sum((x - np.mean(m)) ** 2 for x in m) for m in measurements)
    df_between = num_groups - 1
    df_within = total_values - num_groups
    ms_between = ss_between / df_between if df_between > 0 else float('nan')
    ms_within = ss_within / df_within if df_within > 0 else float('nan')

    repeatability = calc_repeatability_from_ms(ms_within)
    intermediate_precision = calc_intermediate_precision(ms_within, ms_between, len(measurements[0]))
    relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
    relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
    relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
    combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
    expanded_uncertainty = 2 * combined_relative_unc * average_value
    relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_uncertainty, average_value)

    # Sonuç Tablosu
    results_df = pd.DataFrame({
        "Parametre": [
            "Repeatability",
            "Intermediate Precision",
            "Combined Relative Uncertainty",
            "Relative Repeatability",
            "Relative Intermediate Precision",
            "Relative Extra Uncertainty"
        ],
        "Değer": [
            f"{repeatability:.4f}",
            f"{intermediate_precision:.4f}",
            f"{combined_relative_unc:.4f}",
            f"{relative_repeatability:.4f}",
            f"{relative_intermediate_precision:.4f}",
            f"{relative_extra_unc:.4f}"
        ]
    })

    additional_row = pd.DataFrame({
        "Parametre": [lang_texts["average_value"], lang_texts["expanded_uncertainty_col"], lang_texts["relative_expanded_uncertainty_col"]],
        "Değer": [f"{average_value:.4f}", f"{expanded_uncertainty:.4f}", f"{relative_expanded_uncertainty:.4f}"]
    })

    results_df = pd.concat([results_df, additional_row], ignore_index=True)
    st.write(lang_texts["results"])
    st.dataframe(results_df)

    # Günlük Grafik
    fig1, ax1 = plt.subplots()
    for i, group in enumerate(measurements):
        ax1.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=f"Day {i+1}")
    ax1.set_xlabel("Measurement Number")
    ax1.set_ylabel("Value")
    ax1.set_title(lang_texts["daily_measurements"])
    ax1.legend()
    st.pyplot(fig1)

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[language]

    mode = st.radio(lang_texts["paste_area"], ["Elle Giriş", "Yapıştırarak Giriş"] if language=="Türkçe" else ["Manual Input", "Paste Input"])
    if mode in ["Elle Giriş", "Manual Input"]:
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
