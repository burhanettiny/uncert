import numpy as np
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from decimal import Decimal, ROUND_HALF_UP
import math

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
        "combined_relative_unc": "Birleşik Göreceli Belirsizlik",
        "expanded_uncertainty": "Genişletilmiş Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük sütunlar halinde buraya yapıştırın (boş hücreler dışlanır)",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle",
        "download_pdf": "PDF İndir",
        "input_data_table": "Girilen Veriler Tablosu"
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
        "expanded_uncertainty": "Expanded Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste data here (columns = days)",
        "results": "Results",
        "daily_measurements": "Daily Measurement Results",
        "add_uncertainty": "Add Extra Uncertainty Budget",
        "download_pdf": "Download PDF",
        "input_data_table": "Input Data Table"
    }
}

# ------------------------
# Excel Uyumluluğu ile Hesaplama Fonksiyonu
# ------------------------
def calculate_results_excel_style_v2(measurements, extras, lang_texts):
    valid_groups = [np.array([x for x in g if pd.notna(x)], dtype=float) for g in measurements if len(g) > 0]
    if len(valid_groups) < 1:
        st.error("Analiz için en az bir dolu sütun (gün) gerekli!")
        st.stop()

    k = len(valid_groups)
    ns = [len(g) for g in valid_groups]
    N = sum(ns)

    means = [np.mean(g) for g in valid_groups]
    grand_mean = np.average(means, weights=ns)

    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in valid_groups)
    df_within = max(N - k, 1)
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    if k > 1:
        ss_between = sum(n_i * (m_i - grand_mean)**2 for n_i, m_i in zip(ns, means))
        df_between = k - 1
        ms_between = ss_between / df_between if df_between > 0 else 0.0
    else:
        ms_between = 0.0

    repeatability = math.sqrt(ms_within)
    inter_precision = math.sqrt(ms_between) if k > 1 else 0.0

    rel_r = repeatability / grand_mean if grand_mean != 0 else 0
    rel_ip = inter_precision / grand_mean if grand_mean != 0 else 0
    rel_extra = np.sqrt(sum([r[2]**2 for r in extras])) if extras else 0

    u_c = math.sqrt(rel_r**2 + rel_ip**2 + rel_extra**2)
    U = 2 * u_c * grand_mean
    U_rel = (U / grand_mean) * 100 if grand_mean != 0 else 0

    def excel_round(value, digits=4):
        return float(Decimal(value).quantize(Decimal(f"1.{'0'*digits}"), rounding=ROUND_HALF_UP))

    repeatability = excel_round(repeatability, 3)
    inter_precision = excel_round(inter_precision, 3)
    rel_r = excel_round(rel_r, 5)
    rel_ip = excel_round(rel_ip, 5)
    rel_extra = excel_round(rel_extra, 5)
    u_c = excel_round(u_c, 5)
    U = excel_round(U, 2)
    U_rel = excel_round(U_rel, 1)
    grand_mean = excel_round(grand_mean, 2)

    results_list = [
        ("Repeatability", f"{repeatability}", r"s_r = \sqrt{MS_{within}}"),
        ("Intermediate Precision", f"{inter_precision}", r"s_{IP} = \sqrt{MS_{between}}"),
        ("Relative Repeatability", f"{rel_r:.5f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
        ("Relative Intermediate Precision", f"{rel_ip:.5f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}}"),
        ("Relative Extra Uncertainty", f"{rel_extra:.5f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
        ("Combined Relative Uncertainty", f"{u_c:.5f}", r"u_c = \sqrt{u_{r,rel}^2 + u_{IP,rel}^2 + u_{extra,rel}^2}"),
        (lang_texts["average_value"], f"{grand_mean}", r"\bar{x} = \frac{\sum x_i}{n}"),
        (lang_texts["expanded_uncertainty"], f"{U}", r"U = 2 \cdot u_c \cdot \bar{x}"),
        (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
    ]

    return results_list, valid_groups

# ------------------------
# Diğer Fonksiyonlar (PDF, Grafik vs) aynen kalabilir
# ------------------------
# create_pdf, display_results_with_formulas, plot_daily_measurements aynı şekilde kullanılabilir

# ------------------------
# run_manual_mode
# ------------------------
def run_manual_mode(lang_texts):
    st.header(lang_texts["manual_header"])
    days = ['1. Gün', '2. Gün', '3. Gün']
    measurements = []

    for day in days:
        st.subheader(lang_texts["manual_subheader"].format(day))
        values = []
        for i in range(5):
            val = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            values.append(val)
        measurements.append(values)

    overall_avg = np.mean([v for g in measurements for v in g if v != 0]) or 1.0

    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extras = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", key=f"manual_label_{i}")
        if label:
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label),
                             [lang_texts["absolute"], lang_texts["percent"]], key=f"manual_type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                rel_val = value / overall_avg if overall_avg != 0 else 0
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                rel_val = perc / 100
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))

    df_manual = pd.DataFrame(measurements, columns=days)
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df_manual)

    if st.button(lang_texts["calculate_button"]):
        results_list, valid_groups = calculate_results_excel_style_v2(measurements, extras, lang_texts)
        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        plot_daily_measurements(valid_groups, df_manual.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results_manual.pdf",
                           mime="application/pdf")

# ------------------------
# run_paste_mode
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()

    try:
        lines = pasted_data.strip().splitlines()
        rows = []
        for line in lines:
            parts = [x.replace(',', '.') for x in line.split() if x != ""]
            rows.append(parts)

        max_cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < max_cols:
                r.append(np.nan)

        df = pd.DataFrame(rows, dtype=float)
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df)

    measurements = [df[col].dropna().tolist() for col in df.columns]

    overall_avg = np.mean([v for g in measurements for v in g if not np.isnan(v)]) or 1.0

    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extras = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", key=f"paste_label_{i}")
        if label:
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label),
                             [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                rel_val = value / overall_avg if overall_avg != 0 else 0
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                rel_val = perc / 100
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))

    if st.button(lang_texts["calculate_button"]):
        results_list, valid_groups = calculate_results_excel_style_v2(measurements, extras, lang_texts)
        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results.pdf",
                           mime="application/pdf")

# ------------------------
# Main
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]

    mode = st.sidebar.radio("Giriş Modu / Input Mode",
                            ["Yapıştır / Paste", "Elle / Manual"],
                            index=0)

    if mode.startswith("Elle"):
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
