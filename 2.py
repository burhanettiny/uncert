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
        "absolute": "Mutlak",
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

        # --- Bottom-Up metinleri ---
        "bottomup_header": "Bottom-Up Modu",
        "bottomup_desc": "Bileşenleri ve belirsizlik türlerini girerek birleşik belirsizliği hesaplayabilirsiniz.",
        "bottomup_add": "Bileşen Sayısı",
        "bottomup_calc": "Hesapla",
        "bottomup_uc": "Birleşik Göreceli Belirsizlik",
        "bottomup_U": "Genişletilmiş Belirsizlik (U)",
        "bottomup_ref_value": "Referans Değeri (nominal ölçüm değeri)",
        "load_default": "📊 Örnek Verileri Yükle",
        "reset": "🧹 Sıfırla",

        "anova_table_label": "ANOVA Tablosu"
    },

    "English": {
        "manual_header": "Manual Input Mode",
        "manual_subheader": "Enter Measurements for {}",
        "extra_uncert_label": "Extra Uncertainty Sources",
        "extra_uncert_count": "Number of Extra Sources",
        "extra_uncert_type": "Select type for {}",
        "absolute": "Absolute",
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

        # --- Bottom-Up section ---
        "bottomup_header": "Bottom-Up Mode",
        "bottomup_desc": "Enter components and their uncertainty types to calculate combined uncertainty.",
        "bottomup_add": "Number of Components",
        "bottomup_calc": "Calculate",
        "bottomup_uc": "Combined Relative Uncertainty",
        "bottomup_U": "Expanded Uncertainty (U)",
        "bottomup_ref_value": "Reference Value (nominal measurement value)",
        "load_default": "📊 Load Default Data",
        "reset": "🧹 Reset",

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
# Bottom-Up Modu (DÜZELTİLMİŞ)
# ------------------------
def run_bottom_up_mode(lang_texts):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    st.header(lang_texts.get("bottomup_header", "Bottom-Up Modu"))
    st.write(lang_texts.get("bottomup_desc", "Ölçüm bileşenleri ve belirsizliklerini giriniz."))

    # --- Session State ile örnek veri kontrolü (benzersiz anahtarlar kullan) ---
    if "bu_use_default" not in st.session_state:
        st.session_state["bu_use_default"] = False
    if "bu_default_components" not in st.session_state:
        st.session_state["bu_default_components"] = None

    # Butonlar (benzersiz key'ler ile)
    if st.button("📊 Örnek Verileri Yükle / Use Default Data", key="bu_load_btn"):
        # Örnek bileşenleri session'a kaydet
        st.session_state["bu_use_default"] = True
        st.session_state["bu_default_components"] = [
            {"name": "Terazi", "value": 100.0, "u_type": lang_texts.get("absolute", "Mutlak"), "u_val": 0.5},
            {"name": "Pipet", "value": 100.0, "u_type": lang_texts.get("percent", "Yüzde"), "u_val": 1.0},
            {"name": "Cihaz", "value": 100.0, "u_type": lang_texts.get("absolute", "Mutlak"), "u_val": 0.2},
        ]
        st.experimental_rerun()

    if st.button("🧹 Sıfırla / Reset", key="bu_reset_btn"):
        st.session_state["bu_use_default"] = False
        st.session_state["bu_default_components"] = None
        st.experimental_rerun()

    # --- Bileşen sayısı ---
    num_comp = st.number_input(
        lang_texts.get("bottomup_add", "Bileşen Sayısı"),
        min_value=1, max_value=15,
        value=3,
        step=1,
        key="bu_num_comp"
    )

    components = []
    st.subheader("Bileşen Girdileri")

    # --- Eğer örnek veri varsa onu kullan; değilse boş defaultlarla devam et ---
    default_data = st.session_state.get("bu_default_components") or []

    for i in range(int(num_comp)):
        # Eğer yüklenen örnek veri varsa onu kullan, yoksa genel default
        if st.session_state.get("bu_use_default") and i < len(default_data):
            d = default_data[i]
            name_default, value_default, type_default, unc_default = d["name"], d["value"], d["u_type"], d["u_val"]
        else:
            name_default, value_default, type_default, unc_default = f"Bileşen {i+1}", 0.0, lang_texts.get("absolute", "Mutlak"), 0.0

        st.markdown(f"**Bileşen {i+1}**")
        # Her input için benzersiz key kullan
        name = st.text_input(f"Bileşen {i+1} Adı", value=name_default, key=f"bu_name_{i}")
        value = st.number_input(f"{name} Değeri", min_value=0.0, value=value_default, step=0.01, key=f"bu_val_{i}")
        u_type = st.radio(
            f"{name} Belirsizlik Tipi",
            [lang_texts.get("absolute", "Mutlak"), lang_texts.get("percent", "Yüzde")],
            index=0 if type_default == lang_texts.get("absolute", "Mutlak") else 1,
            key=f"bu_type_{i}"
        )
        u_val = st.number_input(f"{name} Belirsizlik", min_value=0.0, value=unc_default, step=0.01, key=f"bu_unc_{i}")
        components.append({"name": name, "value": value, "u_type": u_type, "u_val": u_val})

    # --- Referans değer (ölçümün nominal/reference değeri) ---
    st.subheader("Referans / Nominal Değer")
    reference_value = st.number_input("Referans Değeri (ölçümün nominal değeri, örn. sonuç ortalaması)", min_value=0.0, value=100.0, step=0.01, key="bu_ref_value")

    # --- k değeri manuel ---
    st.subheader("Genişletilmiş Belirsizlik Katsayısı (k)")
    k = st.number_input("k değerini giriniz", min_value=1.0, max_value=10.0, value=2.0, step=0.01, key="k_manual_bu")

    # --- Hesaplama ---
    if len(components) > 0:
        u_squares = []
        for comp in components:
            if comp["u_type"] == lang_texts.get("absolute", "Mutlak"):
                u_rel = comp["u_val"] / comp["value"] if comp["value"] != 0 else 0.0
            else:
                u_rel = comp["u_val"] / 100.0
            u_squares.append(u_rel**2)
            comp["u_rel"] = u_rel

        # Birleşik göreceli belirsizlik (u_c,relative)
        u_c_rel = (sum(u_squares))**0.5
        # Birleşik mutlak belirsizlik: relative * referans değer
        u_c = u_c_rel * reference_value
        U = k * u_c  # k kullanıcıdan alınır

        # --- Görsel tablo ---
        st.subheader("Bileşenler ve Göreceli Belirsizlikleri")
        comp_df = pd.DataFrame(components)
        comp_df_display = comp_df[["name", "value", "u_type", "u_val", "u_rel"]].rename(columns={
            "name": "Bileşen",
            "value": "Değer",
            "u_type": "Belirsizlik Türü",
            "u_val": "Belirsizlik",
            "u_rel": "Göreceli Belirsizlik"
        })
        st.dataframe(
            comp_df_display.style.format({
                "Değer": "{:.4f}",
                "Belirsizlik": "{:.4f}",
                "Göreceli Belirsizlik": "{:.6f}"
            })
        )

        # --- Sonuçlar ---
        st.subheader("Birleşik ve Genişletilmiş Belirsizlik")
        col1, col2 = st.columns(2)
        col1.metric("Birleşik Göreceli Belirsizlik (u_c,rel)", f"{u_c_rel:.6f}")
        col2.metric(f"Genişletilmiş Belirsizlik (U) [k={k}]", f"{U:.6f}")

        # --- Formüller ---
        st.markdown("### Formüller")
        st.latex(r"u_{c,rel} = \sqrt{\sum_{i=1}^{n} u_{i,rel}^2}")
        st.latex(r"u_c = u_{c,rel} \cdot x_{ref}")
        st.latex(fr"U = k \cdot u_c \quad (k = {k})")

        # --- Grafik ---
        st.subheader("Bileşenlerin Göreceli Belirsizlik Katkısı")
        fig, ax = plt.subplots()
        names = [c["name"] for c in components]
        rel_vals = [c["u_rel"] for c in components]
        ax.barh(names, rel_vals)
        ax.set_xlabel("Göreceli Belirsizlik")
        ax.set_ylabel("Bileşen")
        ax.set_title("Bileşen Katkıları")
        st.pyplot(fig)

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
