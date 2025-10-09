import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re

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
        "input_data_table": "Girilen Veriler Tablosu",
        "anova_table_label": "ANOVA Tablosu"
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
        "input_data_table": "Input Data Table",
        "anova_table_label": "ANOVA Table"
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calculate_results(measurements, extras, lang_texts):
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

    anova_df = pd.DataFrame({
        "Varyans Kaynağı": ["Gruplar Arasında", "Gruplar İçinde", "Toplam"],
        "SS": [ss_between, ss_within, ss_between + ss_within],
        "df": [df_between, df_within, df_between + df_within],
        "MS": [ms_between, ms_within, np.nan]
    })

    repeatability = np.sqrt(ms_within)
    n_per_group = int(round(np.mean(ns))) if int(round(np.mean(ns))) > 0 else 1
    inter_precision = np.sqrt((ms_between - ms_within) / n_per_group) if ms_between > ms_within else 0.0

    rel_r = repeatability / grand_mean if grand_mean != 0 else 0.0
    rel_ip = inter_precision / grand_mean if grand_mean != 0 else 0.0
    rel_extra = np.sqrt(sum([r[2]**2 for r in extras])) if extras else 0.0

    u_c = np.sqrt(rel_r**2 + rel_ip**2 + rel_extra**2)
    U = 2 * u_c * grand_mean
    U_rel = (U / grand_mean) * 100 if grand_mean != 0 else 0.0

    results_list = [
        ("Repeatability", f"{repeatability:.4f}", r"s_r = \sqrt{MS_{within}}"),
        ("Intermediate Precision", f"{inter_precision:.4f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n}}"),
        ("Relative Repeatability", f"{rel_r:.4f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
        ("Relative Intermediate Precision", f"{rel_ip:.4f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
        ("Relative Extra Uncertainty", f"{rel_extra:.4f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
        ("Combined Relative Uncertainty", f"{u_c:.4f}", r"u_c = \sqrt{u_{r,rel}^2 + u_{IP,rel}^2 + u_{extra,rel}^2}"),
        (lang_texts["average_value"], f"{grand_mean:.4f}", r"\bar{x} = \frac{\sum x_i}{n}"),
        (lang_texts["expanded_uncertainty"], f"{U:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
        (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
    ]
    return results_list, valid_groups, anova_df

# ------------------------
# PDF Fonksiyonu
# ------------------------
def create_pdf(results_list, anova_df, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 50
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, _ in results_list:
        c.drawString(50, y, f"{param}: {value}")
        y -= 18
        if y < 80:
            c.showPage()
            y = height - 50
    y -= 10
    c.drawString(50, y, lang_texts["anova_table_label"])
    y -= 20
    for idx, row in anova_df.iterrows():
        txt = f"{row['Varyans Kaynağı']}: SS={row['SS']:.6f}, df={int(row['df'])}, MS={row['MS']:.6f}" if pd.notna(row['MS']) else f"{row['Varyans Kaynağı']}: SS={row['SS']:.6f}, df={int(row['df'])}"
        c.drawString(50, y, txt)
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Grafik Fonksiyonu
# ------------------------
def plot_daily_measurements(measurements, col_names, lang_texts):
    fig, ax = plt.subplots()
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = col_names[i] if i < len(col_names) else f"Gün {i+1}"
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)
    ax.set_xlabel("Ölçüm Sayısı")
    ax.set_ylabel("Değer")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# RUN MANUAL MODE
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
        values = [v for v in values if v != 0.0]
        measurements.append(values)

    df_manual = pd.DataFrame([g + [np.nan]*(max(len(x) for x in measurements)-len(g)) for g in measurements]).T
    df_manual.columns = days
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df_manual)

    overall_avg = np.mean([v for g in measurements for v in g if v != 0]) or 1.0
    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extras = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", key=f"manual_label_{i}")
        if label:
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"manual_type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                rel_val = value / overall_avg if overall_avg != 0 else 0
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                rel_val = perc / 100
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))

    if st.button(lang_texts["calculate_button"]):
        results_list, valid_groups, anova_df = calculate_results(measurements, extras, lang_texts)
        st.subheader(lang_texts["overall_results"])
        st.dataframe(anova_df)
        plot_daily_measurements(valid_groups, df_manual.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts)
        st.download_button(lang_texts["download_pdf"], pdf_buffer, "manual_results.pdf", "application/pdf")

# ------------------------
# RUN PASTE MODE
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()
    try:
        pasted_data = pasted_data.replace(',', '.')
        lines = [ln.rstrip() for ln in pasted_data.strip().splitlines() if ln.strip() != ""]
        use_tab = any('\t' in ln for ln in lines)
        use_multi_space = any(re.search(r'\s{2,}', ln) for ln in lines)
        rows = []
        for line in lines:
            if use_tab:
                parts = line.split('\t')
            elif use_multi_space:
                parts = re.split(r'\s{2,}', line)
            else:
                parts = line.split()
            rows.append([float(p) for p in parts])
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r += [np.nan] * (max_cols - len(r))
        df = pd.DataFrame(rows)
        df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

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
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                rel_val = value / overall_avg if overall_avg != 0 else 0
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                rel_val = perc / 100
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))

    if st.button(lang_texts["calculate_button"]):
        results_list, valid_groups, anova_df = calculate_results(measurements, extras, lang_texts)
        st.subheader(lang_texts["overall_results"])
        st.dataframe(anova_df)
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts)
        st.download_button(lang_texts["download_pdf"], pdf_buffer, "paste_results.pdf", "application/pdf")

# ------------------------
# RUN VALIDATION MODE
# ------------------------
def download_and_load_sample_csv():
    sample_data = {
        "1. Gün": [34644.38, 35909.45, 33255.74, 33498.69, 33632.45],
        "2. Gün": [34324.02, 37027.40, 31319.64, 34590.12, 33720.00],
        "3. Gün": [35447.87, 35285.81, 34387.56, 35724.35, 34500.00],
        "Reference": [34800]*5
    }
    df_sample = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    df_sample.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    st.download_button("📥 Örnek Validation CSV’sini İndir", csv_data, "example_validation.csv", "text/csv")
    if st.button("📂 Örnek Veriyi Uygulamaya Yükle"):
        st.session_state["uploaded_df"] = df_sample
        st.success("✅ Örnek veri yüklendi.")
        st.dataframe(df_sample)
    return st.session_state.get("uploaded_df", None)

def run_validation_mode(lang_texts):
    st.header("Validation / Doğrulama Modu")
    df_sample = download_and_load_sample_csv()

    uploaded_file = st.file_uploader("CSV veya Excel dosyası yükleyin", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    elif df_sample is not None:
        df = df_sample
    else:
        st.stop()

    st.dataframe(df)
    reference_col = df["Reference"] if "Reference" in df.columns else None
    measurements = [df[col].dropna().tolist() for col in df.columns if col != "Reference"]
    results_list, valid_groups, anova_df = calculate_results(measurements, [], lang_texts)

    st.subheader("Sonuçlar / Results")
    st.dataframe(anova_df)
    plot_daily_measurements(valid_groups, [col for col in df.columns if col != "Reference"], lang_texts)

    pdf_buffer = create_pdf(results_list, anova_df, lang_texts)
    st.download_button(lang_texts["download_pdf"], pdf_buffer, "uncertainty_results_validation.pdf", "application/pdf")

# ------------------------
# MAIN
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]

    mode = st.sidebar.radio(
        "Giriş Modu / Input Mode",
        ["Yapıştır / Paste", "Elle / Manual", "Validation / Doğrulama"],
        index=0
    )

    if "uploaded_df" not in st.session_state:
        st.session_state["uploaded_df"] = None

    if mode.startswith("Elle"):
        run_manual_mode(lang_texts)
    elif mode.startswith("Validation"):
        run_validation_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
