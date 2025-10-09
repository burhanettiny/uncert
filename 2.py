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
# Güvenli Hesaplama Fonksiyonları
# ------------------------
def safe_mean(arr):
    arr = [a for a in arr if not np.isnan(a)]
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_std(arr):
    arr = [a for a in arr if not np.isnan(a)]
    return np.std(arr, ddof=1) if len(arr) > 1 else 0.0

# ------------------------
# PDF Oluşturma
# ------------------------
def create_pdf(results_list, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height - 50
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, formula in results_list:
        c.drawString(50, y, f"{param}: {value}")
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Sonuç Gösterimi
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.write(f"## {title}")

    # Kalın ve renkli gösterilecek satırlar
    highlight_map = {
        lang_texts["average_value"]: "color: #007BFF; font-weight: bold;",   # Mavi
        lang_texts["expanded_uncertainty"]: "color: #007BFF; font-weight: bold;",  # Yeşil
        lang_texts["relative_expanded_uncertainty_col"]: "color: #007BFF; font-weight: bold;"  # Mor
    }

    # HTML tablo oluştur
    table_html = """
    <style>
    table {
        width: 85%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px 12px;
        text-align: left;
    }
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    tr:hover {
        background-color: #f9f9f9;
    }
    </style>
    <table>
        <tr>
            <th>Parametre</th>
            <th>Değer</th>
        </tr>
    """

    for param, value, formula in results_list:
        style = highlight_map.get(param, "")
        table_html += f"<tr><td style='{style}'>{param}</td><td style='{style}'>{value}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.write("### Formüller")
    for param, _, formula in results_list:
        st.latex(formula)


# ------------------------
# Günlük Grafik
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
# Ortak Hesaplama Fonksiyonu
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

    ms_within = ss_within / df_within if df_within > 0 else 0
    ms_between = ss_between / df_between if df_between > 0 else 0

    repeatability = np.sqrt(ms_within)
    n_eff = np.mean(ns)
    inter_precision = np.sqrt((ms_between - ms_within) / n_eff) if ms_between > ms_within else 0

    rel_r = repeatability / grand_mean if grand_mean != 0 else 0
    rel_ip = inter_precision / grand_mean if grand_mean != 0 else 0
    rel_extra = np.sqrt(sum([r[2]**2 for r in extras])) if extras else 0

    u_c = np.sqrt(rel_r**2 + rel_ip**2 + rel_extra**2)
    U = 2 * u_c * grand_mean
    U_rel = (U / grand_mean) * 100 if grand_mean != 0 else 0

    results_list = [
        ("Repeatability", f"{repeatability:.4f}", r"s_r = \sqrt{MS_{within}}"),
        ("Intermediate Precision", f"{inter_precision:.4f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n_{eff}}}"),
        ("Relative Repeatability", f"{rel_r:.4f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
        ("Relative Intermediate Precision", f"{rel_ip:.4f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
        ("Relative Extra Uncertainty", f"{rel_extra:.4f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
        ("Combined Relative Uncertainty", f"{u_c:.4f}", r"u_c = \sqrt{u_{r,rel}^2 + u_{IP,rel}^2 + u_{extra,rel}^2}"),
        (lang_texts["average_value"], f"{grand_mean:.4f}", r"\bar{x} = \frac{\sum x_i}{n}"),
        (lang_texts["expanded_uncertainty"], f"{U:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
        (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
    ]

    return results_list, valid_groups

# ------------------------
# Elle Giriş Modu
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
        results_list, valid_groups = calculate_results(measurements, extras, lang_texts)
        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        plot_daily_measurements(valid_groups, df_manual.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results_manual.pdf",
                           mime="application/pdf")

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode(lang_texts):
    import re
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()

    try:
        # Oncelikle ondalık virgul -> nokta
        pasted_data = pasted_data.replace(',', '.')

        # Satırları al (boş satırları at)
        lines = [ln.rstrip() for ln in pasted_data.strip().splitlines() if ln.strip() != ""]

        # Hangi ayırıcıyı kullanacağımıza karar ver:
        use_tab = any('\t' in ln for ln in lines)
        use_multi_space = any(re.search(r'\s{2,}', ln) for ln in lines)

        rows = []
        for line in lines:
            if use_tab:
                parts = line.split('\t')                     # tab varsa boş hücreleri korur
            elif use_multi_space:
                parts = re.split(r'\s{2,}', line)           # 2+ boşluklı hizalamaya göre ayır
            else:
                parts = line.split()                        # son çare: tek boşlukla ayır
            parts = [p.strip() for p in parts]
            rows.append(parts)

        # Satırlar aynı uzunlukta değilse sağa doğru NaN ile doldur
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r += [''] * (max_cols - len(r))

        # DataFrame oluştur ve sayısala çevir (hatalı/boş -> NaN)
        df = pd.DataFrame(rows)
        df = df.replace('', np.nan)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]

    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    # Göster: eksik hücreleri vurgula (Styler destekliyorsa)
    st.subheader(lang_texts["input_data_table"])
    try:
        styled = df.style.applymap(lambda v: 'background-color: #ffcccc' if pd.isna(v) else '')
        st.dataframe(styled)
    except Exception:
        st.dataframe(df)

    # Sonraki adımlar: ölçümleri hazırla ve hesaplama butonu (mevcut kodunla aynı)
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
        results_list, valid_groups = calculate_results(measurements, extras, lang_texts)
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

    # Varsayılan mod "Yapıştır / Paste" olacak
    mode = st.sidebar.radio("Giriş Modu / Input Mode",
                            ["Yapıştır / Paste", "Elle / Manual"],
                            index=0)  # index=0 → Paste modu varsayılan

    if mode.startswith("Elle"):
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
