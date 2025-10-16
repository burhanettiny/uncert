import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import math
from reportlab.lib.pagesizes import A4


st.set_page_config(page_title="Uncertainty Tool", layout="wide")

# ------------------------
# Diller / language dictionary (güncellenmiş)
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

        "anova_table_label": "ANOVA Tablosu",
        "validation_header": "Validation / Doğrulama Modu",
        "validation_desc": "Girdi verileri beklenen değerlerle karşılaştırılarak değerlendirilir.",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "measurement_number": "Ölçüm No",
        "value": "Değer",
        "day": "Gün",
        "legend": "Seriler",
        "citation_note": """
Bu araç aşağıdaki şekilde kaynak gösterilebilir:
Yalçınkaya, B. (2025). *UncertCalc: Laboratuvar Ölçümleri için Belirsizlik Hesaplama Aracı*.
(Sürüm 1.0) [https://uncertcalc.streamlit.app/].
"""

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

        "anova_table_label": "ANOVA Table",
        "validation_header": "Validation Mode",
        "validation_desc": "Compare measured results against expected parameter values.",
        "daily_measurements": "Daily Measurement Results",
        "measurement_number": "Measurement #",
        "value": "Value",
        "day": "Day",
        "legend": "Series",
        "citation_note": """
This tool may be cited in academic or technical publications as:
Yalçınkaya, B. (2025). *UncertCalc: An Uncertainty Calculation Tool for Laboratory Measurements*.
(Version 1.0) [https://uncertcalc.streamlit.app/].
"""

    }
}

# ------------------------
# Hesaplama Fonksiyonları (Top-Down)
# ------------------------
def calculate_results(measurements, extras, lang_texts):
    valid_groups = [g for g in measurements if len(g) > 0]
    if len(valid_groups) < 2:
        # raise or return empty results - caller uses Streamlit UI, so return empty
        return [], valid_groups, pd.DataFrame({
            "Varyans Kaynağı": [],
            "SS": [],
            "df": [],
            "MS": []
        })
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
    repeatability = math.sqrt(ms_within) if ms_within >= 0 else 0.0
    n_per_group = int(round(np.mean(ns))) if int(round(np.mean(ns))) > 0 else 1
    inter_precision = math.sqrt(max(ms_between - ms_within, 0.0) / n_per_group) if ms_between > ms_within else 0.0
    rel_r = repeatability / grand_mean if grand_mean != 0 else 0.0
    rel_ip = inter_precision / grand_mean if grand_mean != 0 else 0.0
    rel_extra = math.sqrt(sum([r[2]**2 for r in extras])) if extras else 0.0
    u_c = math.sqrt(rel_r**2 + rel_ip**2 + rel_extra**2)
    U = 2 * u_c * grand_mean
    U_rel = (U / grand_mean) * 100 if grand_mean != 0 else 0.0
    results_list = [
        ("Repeatability", f"{repeatability:.6f}", r"s_r = \sqrt{MS_{within}}"),
        ("Intermediate Precision", f"{inter_precision:.6f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n}}"),
        ("Relative Repeatability", f"{rel_r:.6f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
        ("Relative Intermediate Precision", f"{rel_ip:.6f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
        ("Relative Extra Uncertainty sources", f"{rel_extra:.6f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
        ("Combined Relative Uncertainty", f"{u_c:.6f}", r"u_c = \sqrt{u_{r,rel}^2 + u_{IP,rel}^2 + u_{extra,rel}^2}"),
        (lang_texts["average_value"], f"{grand_mean:.6f}", r"\bar{x} = \frac{\sum x_i}{n}"),
        (lang_texts["expanded_uncertainty"], f"{U:.6f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
        (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.6f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
    ]
    return results_list, valid_groups, anova_df

# ------------------------
# PDF Fonksiyonu
# ------------------------
import io
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Türkçe uyumlu font kaydı
pdfmetrics.registerFont(TTFont("DejaVuSans", "DejaVuSans.ttf"))

def create_pdf(results_list, anova_df, lang_texts, title="Uncertainty Results"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("DejaVuSans", 12)
    y = height - 50

    # Başlık
    c.drawString(50, y, title)
    y -= 30

    # Sonuç listesi
    for param, value, formula in results_list:
        c.drawString(50, y, f"{param}: {value}")
        y -= 18
        if y < 80:
            c.showPage()
            c.setFont("DejaVuSans", 12)
            y = height - 50

    # ANOVA tablosu
    y -= 10
    c.drawString(50, y, lang_texts.get("anova_table_label", "ANOVA Tablosu"))
    y -= 20
    for idx, row in anova_df.iterrows():
        ms_val = row['MS'] if 'MS' in row and not pd.isna(row['MS']) else None
        txt = f"{row['Varyans Kaynağı']}: SS={row['SS']:.6f}, df={int(row['df'])}"
        if ms_val is not None:
            txt += f", MS={row['MS']:.6f}"
        c.drawString(50, y, txt)
        y -= 14
        if y < 60:
            c.showPage()
            c.setFont("DejaVuSans", 12)
            y = height - 50

    # Cite edilebilir footer
    citation_text = lang_texts.get("citation_note", "").strip()
    if citation_text:
        textobject = c.beginText()
        textobject.setTextOrigin(50, 30)  # Alt kenardan 30 pt yukarı
        textobject.setFont("DejaVuSans", 10)  # Footer font boyutu
        for line in citation_text.split("\n"):
            textobject.textLine(line)
            c.drawText(textobject)

    
    c.save()
    buffer.seek(0)
    return buffer


# ------------------------
# Grafik Fonksiyonu
# ------------------------
def plot_daily_measurements(measurements, col_names, lang_texts):
    if not measurements:
        return
    fig, ax = plt.subplots()

    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = col_names[i] if i < len(col_names) else f"{lang_texts.get('day', 'Day')} {i+1}"
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)

    # X eksenini sadece tam sayılarla sınırlama
    ax.set_xticks(range(1, max(len(group) for group in measurements if len(group) > 0) + 1))

    # Çoklu dil destekli eksen ve başlık metinleri
    ax.set_xlabel(lang_texts.get("measurement_number", "Measurement #"))
    ax.set_ylabel(lang_texts.get("value", "Value"))
    ax.set_title(lang_texts.get("daily_measurements", "Daily Measurements"))

    # Legend başlığı dil seçimine göre değişecek şekilde
    legend_title = lang_texts.get("legend", None)
    legend = ax.legend(title=legend_title if legend_title else None)

    # Eğer legend başlığı varsa font boyutunu ayarla
    if legend and legend_title:
        plt.setp(legend.get_title(), fontsize='10', fontweight='bold')

    st.pyplot(fig)
    
# ------------------------
# Sayfa altına cite notu ekleme
# ------------------------
def show_citation(lang_texts):
    citation_text = lang_texts.get("citation_note", "").strip()
    if citation_text:
        st.markdown("---")  # ayırıcı çizgi
        # Çok satırlı metin için <br> kullanıyoruz
        citation_html = citation_text.replace("\n", "<br>")
        st.markdown(
            f"<p style='font-size:10px; color:gray; text-align:center'>{citation_html}</p>",
            unsafe_allow_html=True
        )

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
        try:
            st.latex(formula)
        except Exception:
            st.write(formula)

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
    # dataframe gösterimi
    max_len = max((len(g) for g in measurements), default=0)
    df_manual = pd.DataFrame([g + [np.nan] * (max_len - len(g)) for g in measurements]).T
    df_manual.columns = days
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df_manual)
    overall_avg = np.mean([v for g in measurements for v in g if v != 0]) if any(len(g) for g in measurements) else 1.0
    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1, key="manual_num_extra")
    extras = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", key=f"manual_label_{i}")
        if label:
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label),
                             [lang_texts["absolute"], lang_texts["percent"]],
                             key=f"manual_type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                rel_val = value / overall_avg if overall_avg != 0 else 0
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                rel_val = perc / 100.0
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))
    if st.button(lang_texts["calculate_button"], key="manual_calc_btn"):
        results_list, valid_groups, anova_df = calculate_results(measurements, extras, lang_texts)
        if not results_list:
            st.error("En az iki dolu sütun (gün) gerekli.")
            return
        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        st.subheader(lang_texts["anova_table_label"])
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))
        plot_daily_measurements(valid_groups, df_manual.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts, title="Manual Mode Results")
        st.download_button(label=lang_texts["download_pdf"], data=pdf_buffer, file_name="uncertainty_results_manual.pdf", mime="application/pdf")
        # ✅ Altta citation notunu göster
        show_citation(lang_texts)

# ------------------------
# Paste Mod (Top-Down paste)
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"], height=200)
    if not pasted_data:
        st.info("Veri yapıştırıldıktan sonra hesaplama butonuna basın.")
        return
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
        return
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df)
    measurements = [df[col].dropna().tolist() for col in df.columns]
    overall_avg = np.mean([v for g in measurements for v in g if not np.isnan(v)]) if measurements else 1.0
    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1, key="paste_num_extra")
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
                rel_val = perc / 100.0
                value = rel_val * overall_avg
            extras.append((label, value, rel_val))
    if st.button(lang_texts["calculate_button"], key="paste_calc_btn"):
        results_list, valid_groups, anova_df = calculate_results(measurements, extras, lang_texts)
        if not results_list:
            st.error("En az iki dolu sütun (gün) gerekli.")
            return
        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)
        st.subheader(lang_texts["anova_table_label"])
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
        pdf_buffer = create_pdf(results_list, anova_df, lang_texts, title="Paste Mode Results")
        st.download_button(label=lang_texts["download_pdf"], data=pdf_buffer, file_name="uncertainty_results.pdf", mime="application/pdf")
        # ✅ Altta citation notunu göster
        show_citation(lang_texts)
# ------------------------
# Validation Mod
# ------------------------
def run_validation_mode(lang_texts):
    st.header(lang_texts.get("validation_header", "Validation"))
    st.write(lang_texts.get("validation_desc", ""))

    if "df" not in st.session_state:
        st.session_state["df"] = None

    # örnek veri butonu
    if st.button(lang_texts.get("load_default", "📊 Örnek Verileri Yükle"), key="val_load_default"):
        st.session_state["df"] = pd.DataFrame({
            "1. Gün": [34644.38, 35909.45, 33255.74, 33498.69, 33632.45],
            "2. Gün": [34324.02, 37027.40, 31319.64, 34590.12, 34521.42],
            "3. Gün": [35447.87, 35285.81, 34387.56, 35724.35, 36236.50]
        })
        st.rerun()

    pasted_data = st.text_area("Veri giriş alanı (Excel'den kopyala-yapıştır)", height=200, placeholder="Örnek: 1,85\t1,99\t1,94\n1,99\t1,88\t1,91\n...")
    if pasted_data.strip():
        try:
            tmp = pasted_data.replace(',', '.')
            lines = [ln.rstrip() for ln in tmp.strip().splitlines() if ln.strip() != ""]
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
            st.success("Yapıştırılan veriler işlendi.")
            st.rerun()
        except Exception as e:
            st.error(f"Hata: {e}")

    if st.session_state.get("df") is None:
        st.info("Lütfen verileri yapıştırın veya örnek verileri yükleyin.")
        return

    df = st.session_state["df"]
    st.subheader(lang_texts.get("input_data_table", "Input Data Table"))
    st.dataframe(df)

    parameters = [
        "Repeatability", "Intermediate Precision", "Relative Repeatability",
        "Relative Intermediate Precision", "Relative Extra Uncertainty",
        "Combined Relative Uncertainty", "Ortalama Değer",
        "Genişletilmiş Belirsizlik (k=2)",
        "Göreceli Genişletilmiş Belirsizlik (%)"
    ]
    st.subheader("Beklenen Değerler (Parametre Bazlı)")
    expected_values = {}
    for p in parameters:
        expected_values[p] = st.number_input(f"{p}", min_value=0.0, value=0.0, step=0.00001, format="%.6f", key=f"val_expected_{p}")

    tolerance = st.slider("Tolerans (%)", 1, 50, 5, step=1, key="val_tolerance")

    if st.button(lang_texts.get("calculate_button", "Sonuçları Hesapla"), key="val_calc_btn"):
        measurements = [df[col].dropna().tolist() for col in df.columns]
        if not measurements or df.empty:
            st.error("Veri bulunamadı.")
            return
        results_list, valid_groups, anova_df = calculate_results(measurements, [], lang_texts)
        if not results_list:
            st.error("En az iki dolu sütun (gün) gerekli.")
            return
        df_results = pd.DataFrame({
            "Parametre": [r[0] for r in results_list],
            "Değer": [float(r[1]) if isinstance(r[1], str) and r[1].replace('.','',1).isdigit() else r[1] for r in results_list]
        })
        df_results["Beklenen Değer"] = df_results["Parametre"].apply(lambda p: expected_values.get(p, 0.0))
        def pass_fail(row):
            try:
                val = float(row["Değer"])
                exp = float(row["Beklenen Değer"])
            except Exception:
                return "❌ Kaldı"
            if exp == 0:
                return "✅ Geçti" if val == 0 else "❌ Kaldı"
            return "✅ Geçti" if abs((val - exp) / exp * 100) <= tolerance else "❌ Kaldı"
        df_results["Sonuç"] = df_results.apply(pass_fail, axis=1)
        st.subheader("Sonuçlar (Beklenen Değer Karşılaştırmalı)")
        st.dataframe(df_results.style.format({"Değer": "{:.6f}", "Beklenen Değer": "{:.6f}"}))
        st.subheader(lang_texts.get("anova_table_label", "ANOVA Table"))
        st.dataframe(anova_df.style.format({"SS": "{:.9f}", "MS": "{:.9f}", "df": "{:.0f}"}))
        plot_daily_measurements(valid_groups, df.columns.tolist(), lang_texts)
        show_citation(lang_texts)

# ------------------------
# Bottom-Up Mod (güncel)
# ------------------------
def run_bottom_up_mode(lang_texts):
    st.header(lang_texts.get("bottomup_header", "Bottom-Up Mode"))
    st.write(lang_texts.get("bottomup_desc", ""))

    # session keys
    if "bu_use_default" not in st.session_state:
        st.session_state["bu_use_default"] = False
    if "bu_default_components" not in st.session_state:
        st.session_state["bu_default_components"] = None

    # butonlar (dil destekli)
    if st.button(lang_texts.get("load_default", "📊 Örnek Verileri Yükle"), key="bu_load_btn"):
        st.session_state["bu_use_default"] = True
        st.session_state["bu_default_components"] = [
            {"name": "Terazi", "value": 100.0, "u_type": lang_texts.get("absolute", "Mutlak"), "u_val": 0.5},
            {"name": "Pipet", "value": 100.0, "u_type": lang_texts.get("percent", "Yüzde"), "u_val": 1.0},
            {"name": "Cihaz", "value": 100.0, "u_type": lang_texts.get("absolute", "Mutlak"), "u_val": 0.2},
        ]
        st.rerun()

    if st.button(lang_texts.get("reset", "🧹 Sıfırla"), key="bu_reset_btn"):
        st.session_state["bu_use_default"] = False
        st.session_state["bu_default_components"] = None
        # temizle: bileşenlerle ilgili bütün inputların varsayılanlarını temizlemek zor olabilir; rerun yeterli
        st.rerun()

    num_comp = st.number_input(lang_texts.get("bottomup_add", "Number of Components"), min_value=1, max_value=20, value=3, step=1, key="bu_num_comp")
    components = []
    st.subheader("Bileşen Girdileri")
    default_data = st.session_state.get("bu_default_components") or []

    for i in range(int(num_comp)):
        if st.session_state.get("bu_use_default") and i < len(default_data):
            d = default_data[i]
            name_default, value_default, type_default, unc_default = d["name"], d["value"], d["u_type"], d["u_val"]
        else:
            name_default, value_default, type_default, unc_default = f"Bileşen {i+1}", 0.0, lang_texts.get("absolute", "Mutlak"), 0.0

        st.markdown(f"**Bileşen {i+1}**")
        name = st.text_input(f"Bileşen {i+1} Adı", value=name_default, key=f"bu_name_{i}")
        value = st.number_input(f"{name} Değeri", min_value=0.0, value=value_default, step=0.01, key=f"bu_val_{i}")
        u_type = st.radio(f"{name} Belirsizlik Tipi", [lang_texts.get("absolute", "Mutlak"), lang_texts.get("percent", "Yüzde")],
                          index=0 if type_default == lang_texts.get("absolute", "Mutlak") else 1, key=f"bu_type_{i}")
        u_val = st.number_input(f"{name} Belirsizlik", min_value=0.0, value=unc_default, step=0.01, key=f"bu_unc_{i}")
        components.append({"name": name, "value": value, "u_type": u_type, "u_val": u_val})

    # referans değer
    st.subheader(lang_texts.get("bottomup_ref_value", "Reference Value"))
    reference_value = st.number_input(lang_texts.get("bottomup_ref_value", "Reference Value"), min_value=0.0, value=100.0, step=0.01, key="bu_ref_value")

    # k değeri
    st.subheader("Genişletilmiş Belirsizlik Katsayısı (k)")
    k = st.number_input("k değerini giriniz", min_value=1.0, max_value=10.0, value=2.0, step=0.01, key="k_manual_bu")

    if len(components) > 0:
        u_squares = []
        for comp in components:
            if comp["u_type"] == lang_texts.get("absolute", "Mutlak"):
                u_rel = comp["u_val"] / comp["value"] if comp["value"] != 0 else 0.0
            else:
                u_rel = comp["u_val"] / 100.0
            u_squares.append(u_rel**2)
            comp["u_rel"] = u_rel

        u_c_rel = math.sqrt(sum(u_squares)) if u_squares else 0.0
        u_c = u_c_rel * reference_value
        U = k * u_c

        # Tablo gösterimi
        st.subheader("Bileşenler ve Göreceli Belirsizlikleri")
        comp_df = pd.DataFrame(components)
        comp_df_display = comp_df[["name", "value", "u_type", "u_val", "u_rel"]].rename(columns={
            "name": "Bileşen",
            "value": "Değer",
            "u_type": "Belirsizlik Türü",
            "u_val": "Belirsizlik",
            "u_rel": "Göreceli Belirsizlik"
        })
        st.dataframe(comp_df_display.style.format({
            "Değer": "{:.4f}",
            "Belirsizlik": "{:.4f}",
            "Göreceli Belirsizlik": "{:.6f}"
        }))

        # sonuçlar
        st.subheader("Birleşik ve Genişletilmiş Belirsizlik")
        col1, col2 = st.columns(2)
        col1.metric("Birleşik Göreceli Belirsizlik (u_c,rel)", f"{u_c_rel:.6f}")
        col2.metric(f"Genişletilmiş Belirsizlik (U) [k={k}]", f"{U:.6f}")

        # formüller
        st.markdown("### Formüller")
        st.latex(r"u_{c,rel} = \sqrt{\sum_{i=1}^{n} u_{i,rel}^2}")
        st.latex(r"u_c = u_{c,rel} \cdot x_{ref}")
        st.latex(fr"U = k \cdot u_c \quad (k = {k})")

        # grafik (bileşen katkıları)
        st.subheader("Bileşenlerin Göreceli Belirsizlik Katkısı")
        fig, ax = plt.subplots()
        names = [c["name"] for c in components]
        rel_vals = [c["u_rel"] for c in components]
        ax.barh(names, rel_vals)
        ax.set_xlabel("Göreceli Belirsizlik")
        ax.set_ylabel("Bileşen")
        ax.set_title("Bileşen Katkıları")
        st.pyplot(fig)
        show_citation(lang_texts)

# ------------------------
# Main
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"], index=0)
    lang_texts = languages[lang_choice]

    method_choice = st.sidebar.radio(lang_texts["method_choice"], [lang_texts["top_down"], lang_texts["bottom_up"]], index=0)

    if method_choice == lang_texts["top_down"]:
        mode = st.sidebar.radio("Giriş Modu / Input Mode", ["Yapıştır / Paste", "Elle / Manual", "Validation / Doğrulama"], index=0)
        if mode.startswith("Elle"):
            run_manual_mode(lang_texts)
        elif mode.startswith("Validation"):
            run_validation_mode(lang_texts)
        else:
            run_paste_mode(lang_texts)
    else:
        run_bottom_up_mode(lang_texts)

if __name__ == "__main__":
    main()
