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
        "extra_uncert_label": "Ek Belirsizlik Kaynağı",
        "extra_uncert_count": "Ek Belirsizlik Kaynağı Sayısı",
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
        "add_uncertainty": "Ek Belirsizlik Kaynağı Ekle",
        "download_pdf": "PDF İndir",
        "input_data_table": "Girilen Veriler Tablosu",
        "method_choice": "Metot Seçimi",
        "top_down": "Top-Down (İstatistiksel)",
        "bottom_up": "Bottom-Up (Bileşen Tabanlı)",
        "bottomup_header": "Bottom-Up Modu", 
        "bottomup_desc": "Bottom-Up yöntemi ile belirsizlik hesaplaması yapabilirsiniz.",
        "bottomup_add": "Eklenen Bileşen Sayısı",
        "bottomup_calc": "Hesapla",
        "bottomup_uc": "Birleşik Göreceli Belirsizlik",
        "bottomup_U": "Genişletilmiş Belirsizlik (U)",
        "anova_table_label": "ANOVA Tablosu"
    },
    "English": {
        "manual_header": "Manual Input Mode",
        "manual_subheader": "Enter Measurements for {}",
        "extra_uncert_label": "Extra Uncertainty sources",
        "extra_uncert_count": "Number of Extra sources",
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
        "add_uncertainty": "Add Extra Uncertainty Sources",
        "download_pdf": "Download PDF",
        "input_data_table": "Input Data Table",
        "method_choice": "Select Method",
        "top_down": "Top-Down (Statistical)",
        "bottom_up": "Bottom-Up (Component-Based)",
        "bottomup_header": "Bottom-Up Mode",
        "bottomup_desc": "You can calculate uncertainty using the Bottom-Up method.",
        "bottomup_add": "Number of Components",
        "bottomup_calc": "Calculate",
        "bottomup_uc": "Combined Relative Uncertainty",
        "bottomup_U": "Expanded Uncertainty (U)",
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
        ("Relative Extra Uncertainty sources", f"{rel_extra:.4f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
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
    for param, value, formula in results_list:
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
# Sonuç Gösterim Fonksiyonu
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.write(f"## {title}")
    highlight_map = {
        lang_texts["average_value"]: "color: #007BFF; font-weight: bold;",
        lang_texts["expanded_uncertainty"]: "color: #007BFF; font-weight: bold;",
        lang_texts["relative_expanded_uncertainty_col"]: "color: #007BFF; font-weight: bold;"
    }
    table_html = """
    <style>
    table {width: 85%; border-collapse: collapse; margin-top: 10px; margin-bottom: 15px;}
    th, td {border: 1px solid #ddd; padding: 8px 12px; text-align: left;}
    th {background-color: #f5f5f5; font-weight: bold;}
    tr:hover {background-color: #f9f9f9;}
    </style>
    <table>
        <tr><th>Parametre</th><th>Değer</th></tr>
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
# Manual Mod
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
        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        st.subheader(lang_texts["anova_table_label"])
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))
        plot_daily_measurements(valid_groups, df_manual.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts)
        st.download_button(label=lang_texts["download_pdf"], data=pdf_buffer, file_name="uncertainty_results_manual.pdf", mime="application/pdf")

# ------------------------
# Paste Mod
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
            parts = [p.strip() for p in parts]
            rows.append(parts)
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r += [''] * (max_cols - len(r))
        df = pd.DataFrame(rows)
        df = df.replace('', np.nan)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)
        st.subheader(lang_texts["anova_table_label"])
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts)
        st.download_button(label=lang_texts["download_pdf"], data=pdf_buffer, file_name="uncertainty_results.pdf", mime="application/pdf")

# ------------------------
def run_validation_mode(lang_texts):
    st.header("Validation / Doğrulama Modu")

    # ------------------------
    # Örnek veri butonu
    # ------------------------
    if st.button("📊 Örnek Verileri Yükle / Use Default Data"):
        default_data = {
            "1. Gün": [34644.38, 35909.45, 33255.74, 33498.69, 33632.45],
            "2. Gün": [34324.02, 37027.40, 31319.64, 34590.12, 34521.42],
            "3. Gün": [35447.87, 35285.81, 34387.56, 35724.35, 36236.50]
        }
        st.session_state["df"] = pd.DataFrame(default_data)
        st.success("Örnek veriler başarıyla yüklendi ✅")

    # ------------------------
    # Excel’den kopyala-yapıştır
    # ------------------------
    st.subheader("📋 Verilerinizi buraya yapıştırabilirsiniz (Excel’den kopyala-yapıştır)")
    pasted_data = st.text_area("Veri giriş alanı", height=200, placeholder="Örnek: 1,85\t1,99\t1,94\n1,99\t1,88\t1,91\n...")

    df = None
    if pasted_data.strip():
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
                parts = [p.strip() for p in parts]
                rows.append(parts)

            max_cols = max(len(r) for r in rows)
            for r in rows:
                if len(r) < max_cols:
                    r += [''] * (max_cols - len(r))

            df = pd.DataFrame(rows)
            df = df.replace('', np.nan)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
            st.session_state["df"] = df
            st.success("Yapıştırılan veriler başarıyla işlendi ✅")

        except Exception as e:
            st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
            st.stop()

    # ------------------------
    # Veri yoksa uyarı
    # ------------------------
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Lütfen Excel’den verilerinizi yapıştırın veya örnek verileri yükleyin.")
        st.stop()

    df = st.session_state["df"]
    st.subheader(lang_texts.get("input_data_table", "Girdi Verileri"))
    st.dataframe(df)

    # ------------------------
    # Beklenen değerler
    # ------------------------
    st.subheader("Beklenen Değerler (Parametre Bazlı)")
    parameters = [
        "Repeatability", "Intermediate Precision", "Relative Repeatability",
        "Relative Intermediate Precision", "Relative Extra Uncertainty",
        "Combined Relative Uncertainty", "Ortalama Değer",
        "Genişletilmiş Belirsizlik (k=2)",
        "Göreceli Genişletilmiş Belirsizlik (%)"
    ]
    expected_values = {}
    for p in parameters:
        expected_values[p] = st.number_input(f"{p}", min_value=0.0, value=0.0, step=0.01, format="%.5f")

    tolerance = st.slider("Tolerans (%)", 1, 20, 5, step=1)

    # ------------------------
    # Hesaplama
    # ------------------------
    if st.button(lang_texts.get("calculate_button", "Sonuçları Hesapla")):
        measurements = [df[col].dropna().tolist() for col in df.columns]

        if not measurements or df.empty:
            st.error("Veri bulunamadı. Lütfen geçerli veriler girin veya örnek verileri yükleyin.")
            st.stop()

        # --- Hesaplama ---
        results_list, valid_groups, anova_df = calculate_results(measurements, [], lang_texts)

        # --- Sonuç DataFrame ---
        df_results = pd.DataFrame({
            "Parametre": parameters,
            "Değer": [row[1] if len(row) > 1 else None for row in results_list]
        })
        df_results["Değer"] = pd.to_numeric(df_results["Değer"], errors="coerce")
        df_results["Beklenen Değer"] = df_results["Parametre"].apply(lambda p: expected_values.get(p, 0.0))

        # --- Sonuç (Geçti/Kaldı) ---
        df_results["Sonuç"] = df_results.apply(
            lambda row: "✅ Geçti" if pd.notna(row["Değer"]) and (
                (row["Beklenen Değer"] == 0 and row["Değer"] == 0) or
                (row["Beklenen Değer"] != 0 and abs((row["Değer"] - row["Beklenen Değer"]) / row["Beklenen Değer"] * 100) <= tolerance)
            ) else "❌ Kaldı",
            axis=1
        )

        # --- Sonuç tablosu ---
        st.subheader("Sonuçlar (Beklenen Değer Karşılaştırmalı)")
        st.dataframe(df_results.style.format({"Değer": "{:.5f}", "Beklenen Değer": "{:.5f}"}))

        # --- ANOVA tablosu ---
        st.subheader(lang_texts.get("anova_table_label", "ANOVA Tablosu"))
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))

        # --- Günlük ölçüm grafiği ---
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
# ------------------------
# Bottom-Up Modu
# ------------------------
def run_bottom_up_mode(lang_texts):
    st.header(lang_texts.get("bottomup_header", "Bottom-Up Modu"))
    st.write(lang_texts.get("bottomup_desc", "Ölçüm bileşenleri ve belirsizliklerini giriniz."))

    # Kaç bileşen girişi yapılacak
    num_comp = st.number_input(lang_texts.get("bottomup_add", "Bileşen Sayısı"), min_value=1, max_value=15, value=3, step=1)

    components = []
    st.subheader("Bileşen Girdileri")
    for i in range(int(num_comp)):
        st.markdown(f"**Bileşen {i+1}**")
        name = st.text_input(f"Bileşen {i+1} Adı", key=f"bu_name_{i}")
        value = st.number_input(f"{name} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"bu_val_{i}")
        u_type = st.radio(f"{name} Belirsizlik Tipi", [lang_texts.get("absolute", "Mutlak"), lang_texts.get("percent", "Yüzde")], key=f"bu_type_{i}")
        u_val = st.number_input(f"{name} Belirsizlik", min_value=0.0, value=0.0, step=0.01, key=f"bu_unc_{i}")
        components.append({
            "name": name,
            "value": value,
            "u_type": u_type,
            "u_val": u_val
        })

    if st.button(lang_texts.get("bottomup_calc", "Hesapla")):
        # Hesaplama
        u_squares = []
        for comp in components:
            if comp["u_type"] == lang_texts.get("absolute", "Mutlak"):
                u_rel = comp["u_val"] / comp["value"] if comp["value"] != 0 else 0
            else:  # yüzde
                u_rel = comp["u_val"] / 100
            u_squares.append(u_rel**2)

        u_c_rel = (sum(u_squares))**0.5
        u_c = u_c_rel * sum(comp["value"] for comp in components) / len(components)  # Ortalama değerle çarp
        U = 2 * u_c  # k=2

        # Sonuç gösterimi
        st.subheader("Sonuçlar")
        st.markdown(f"**{lang_texts.get('bottomup_uc','Birleşik Göreceli Belirsizlik (u_c)')}:** {u_c:.6f}")
        st.markdown(f"**{lang_texts.get('bottomup_U','Genişletilmiş Belirsizlik (U)')}:** {U:.6f}")


# ------------------------
# Main
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]

    # Ana seviye seçim
    method_choice = st.sidebar.radio(
        lang_texts["method_choice"],
        [lang_texts["top_down"], lang_texts["bottom_up"]],
        index=0
    )

    if method_choice == lang_texts["top_down"]:
        mode = st.sidebar.radio(
            "Giriş Modu / Input Mode",
            ["Yapıştır / Paste", "Elle / Manual", "Validation / Doğrulama"],
            index=0
        )
        if mode.startswith("Elle"):
            run_manual_mode(lang_texts)
        elif mode.startswith("Validation"):
            run_validation_mode(lang_texts)
        else:
            run_paste_mode(lang_texts)

    elif method_choice == lang_texts["bottom_up"]:
        run_bottom_up_mode(lang_texts)


if __name__ == "__main__":
    main()
