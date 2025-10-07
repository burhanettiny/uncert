import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
        "calculate_button": "Sonuçları Hesapla",
        "overall_results": "Genel Sonuçlar",
        "average_value": "Ortalama Değer",
        "repeatability_within": "Gün İçi Tekrarlanabilirlik",
        "repeatability_between": "Günler Arası Tekrarlanabilirlik",
        "combined_relative_unc": "Bileşik Göreceli Belirsizlik",
        "expanded_uncertainty": "Genişletilmiş Genel Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle",
        "download_pdf": "PDF İndir",
        "measurement_count": "Ölçüm Sayısı",
        "repeat": "Tekrar",
        "day_label": "{}. Gün",
        "table_day_col": "{}. Gün",
        "table_measurement_col": "Ölçüm {}",
        "formula_header": "Formül",
        "formula_desc_header": "Ne Anlama Geliyor"
    },
    "English": {
        "manual_header": "Manual Input Mode",
        "manual_subheader": "Enter Measurements for {}",
        "extra_uncert_label": "Extra Uncertainty Budget",
        "extra_uncert_count": "Number of Extra Uncertainty Budgets",
        "extra_uncert_type": "Select type for {}",
        "absolute": "Absolute Value",
        "percent": "Percent",
        "calculate_button": "Calculate Results",
        "overall_results": "Overall Results",
        "average_value": "Average Value",
        "repeatability_within": "Repeatability Within Days",
        "repeatability_between": "Repeatability Between Days",
        "combined_relative_unc": "Combined Relative Uncertainty",
        "expanded_uncertainty": "Expanded Overall Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste data here (columns = days)",
        "results": "Results",
        "daily_measurements": "Daily Measurement Results",
        "add_uncertainty": "Add Extra Uncertainty Budget",
        "download_pdf": "Download PDF",
        "measurement_count": "Measurement Count",
        "repeat": "Repeat",
        "day_label": "Day {}",
        "table_day_col": "Day {}",
        "table_measurement_col": "Measurement {}",
        "formula_header": "Formula",
        "formula_desc_header": "Meaning"
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_std(arr):
    return np.std(arr, ddof=1) if len(arr) > 1 else 0.0

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else 0.0

# ------------------------
# PDF Oluşturma Fonksiyonu
# ------------------------
def create_pdf(results_list, lang_texts, filename="results.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 50
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, formula, desc in results_list:
        c.drawString(50, y, f"{param}: {value}   Formula: {formula} ({desc})")
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Sonuç Gösterimi (Tablo + Formül)
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.write(f"## {title}")
    df_values = pd.DataFrame([(p, v) for p, v, f, d in results_list], columns=[lang_texts["results"], "Value"])
    st.dataframe(df_values)
    st.write("### Formüller ve Açıklamaları")
    df_formulas = pd.DataFrame([(p, f"${f}$", d) for p, v, f, d in results_list],
                               columns=[lang_texts["results"], lang_texts["formula_header"], lang_texts["formula_desc_header"]])
    st.dataframe(df_formulas)
    return df_values

# ------------------------
# Günlük Grafik
# ------------------------
def plot_daily_measurements(measurements, col_names, lang_texts):
    fig, ax = plt.subplots()
    max_len = max((len(g) for g in measurements), default=1)
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = col_names[i] if i < len(col_names) else lang_texts["day_label"].format(i+1)
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)
    ax.set_xticks(range(1, max_len+1))
    ax.set_xlabel(lang_texts["measurement_count"])
    ax.set_ylabel("Value")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Elle Giriş Modu
# ------------------------
def run_manual_mode(lang_texts):
    st.header(lang_texts["manual_header"])
    days = [lang_texts["day_label"].format(i+1) for i in range(3)]
    total_measurements = []

    for day in days:
        st.subheader(lang_texts["manual_subheader"].format(day))
        measurements = []
        for i in range(5):
            value = st.number_input(f"{day} - {lang_texts['repeat']} {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

    df_manual = pd.DataFrame(total_measurements, columns=[lang_texts["table_measurement_col"].format(i+1) for i in range(5)], index=days)
    st.write("### Girilen Veriler")
    st.dataframe(df_manual)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["extra_uncert_label"])
    overall_measurements = [val for day in total_measurements for val in day]
    overall_avg = safe_mean(overall_measurements) if overall_measurements else 1.0

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
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        all_vals = [v for day in total_measurements for v in day]
        average_value = safe_mean(all_vals)
        repeatability_within_days = safe_std([v for day in total_measurements for v in day])
        day_means = [safe_mean(day) for day in total_measurements]
        repeatability_between_days = safe_std(day_means)
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))

        u_repeat_rel = repeatability_within_days / average_value if average_value != 0 else 0.0
        u_ip_rel = repeatability_between_days / average_value if average_value != 0 else 0.0
        combined_relative_unc = np.sqrt(u_repeat_rel**2 + u_ip_rel**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * average_value
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_overall_uncertainty, average_value)

        results_list = [
            (lang_texts["repeatability_within"], f"{repeatability_within_days:.4f}", r"s_r = \sqrt{MS_{within}}", "Gün içi tekrarlanabilirlik"),
            (lang_texts["repeatability_between"], f"{repeatability_between_days:.4f}", r"s_{IP} = \sqrt{\frac{MS_{between}-MS_{within}}{n}}", "Günler arası tekrarlanabilirlik")
        ]

        for label, value, rel_val, input_type in extra_uncertainties:
            results_list.append((label, f"{value:.4f}", r"u_{extra}", "Ekstra belirsizlik"))

        results_list.extend([
            (lang_texts["combined_relative_unc"], f"{combined_relative_unc:.4f}", r"u_c = \sqrt{u_r^2 + u_{IP}^2 + u_{extra}^2}", "Bileşik göreceli belirsizlik"),
            (lang_texts["average_value"], f"{average_value:.4f}", r"\bar{x} = \frac{\sum x}{n}", "Ortalama değer"),
            (lang_texts["expanded_uncertainty"], f"{expanded_overall_uncertainty:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}", "Genişletilmiş belirsizlik (k=2)"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{relative_expanded_uncertainty:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100", "Göreceli genişletilmiş belirsizlik")
        ])

        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        plot_daily_measurements(total_measurements, [f"{d}" for d in days], lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results_manual.pdf",
                           mime="application/pdf")

# ------------------------
# Yapıştır / Paste Modu
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    df.columns = [lang_texts["table_day_col"].format(i+1) for i in range(df.shape[1])]
    df.index = [lang_texts["table_measurement_col"].format(i+1) for i in range(len(df))]
    st.write("### Girilen Veriler")
    st.dataframe(df)

    measurements = [df[col].dropna().tolist() for col in df.columns]
    all_values = [v for g in measurements for v in g if not np.isnan(v)]
    if not all_values:
        st.error("Geçerli sayısal veri bulunamadı!")
        st.stop()

    overall_avg = np.mean(all_values)

    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        valid_groups = [g for g in measurements if len(g) > 0]
        if len(valid_groups) < 2:
            st.error("Analiz için en az iki dolu sütun (gün) gerekli!")
            st.stop()

        means = [np.mean(g) for g in valid_groups]
        ns = [len(g) for g in valid_groups]
        N = sum(ns)
        k = len(valid_groups)
        grand_mean = np.average(means, weights=ns)
        ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in valid_groups)
        ss_between = sum(ns[i] * (means[i] - grand_mean)**2 for i in range(k))
        df_within = N - k
        df_between = k - 1
        ms_within = ss_within / df_within if df_within > 0 else 0.0
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        repeatability = np.sqrt(ms_within)
        n_eff = np.mean(ns)
        intermediate_precision = np.sqrt((ms_between - ms_within) / n_eff) if ms_between > ms_within else 0.0
        relative_repeatability = repeatability / grand_mean if grand_mean != 0 else 0.0
        relative_intermediate_precision = intermediate_precision / grand_mean if grand_mean != 0 else 0.0
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties])) if extra_uncertainties else 0.0
        combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
        expanded_uncertainty = 2 * combined_relative_unc * grand_mean
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_uncertainty, grand_mean)

        results_list = [
            (lang_texts["repeatability_within"], f"{repeatability:.4f}", r"s_r = \sqrt{MS_{within}}", "Gün içi tekrarlanabilirlik"),
            (lang_texts["repeatability_between"], f"{intermediate_precision:.4f}", r"s_{IP} = \sqrt{\frac{MS_{between}-MS_{within}}{n_{eff}}}", "Günler arası tekrarlanabilirlik")
        ]

        for label, value, rel_val, input_type in extra_uncertainties:
            results_list.append((label, f"{value:.4f}", r"u_{extra}", "Ekstra belirsizlik"))

        results_list.extend([
            (lang_texts["combined_relative_unc"], f"{combined_relative_unc:.4f}", r"u_c = \sqrt{u_r^2 + u_{IP}^2 + u_{extra}^2}", "Bileşik göreceli belirsizlik"),
            (lang_texts["average_value"], f"{grand_mean:.4f}", r"\bar{x} = \frac{\sum x_i}{n}", "Ortalama değer"),
            (lang_texts["expanded_uncertainty"], f"{expanded_uncertainty:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}", "Genişletilmiş belirsizlik (k=2)"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{relative_expanded_uncertainty:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100", "Göreceli genişletilmiş belirsizlik")
        ])

        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        plot_daily_measurements(measurements, [col for col in df.columns], lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results_paste.pdf",
                           mime="application/pdf")

# ------------------------
# Main
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]

    mode = st.sidebar.radio("Giriş Modu / Input Mode", ["Elle / Manual", "Yapıştır / Paste"])
    if mode.startswith("Elle"):
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
