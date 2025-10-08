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
        "title": "🔬 UncertCalc: Laboratuvar Ölçüm Belirsizliği Hesaplama Aracı",
        "select_mode": "Veri Girişi Modu Seçin",
        "manual": "Elle Veri Girişi",
        "paste": "Veri Yapıştırma (Excel'den)",
        "manual_header": "Elle Veri Girişi Modu",
        "manual_subheader": "{} İçin Ölçüm Sonucu Girin",
        "paste_title": "📋 Veri Yapıştırma Modu",
        "paste_subtitle": "Excel veya tablo biçimindeki ölçüm verilerini aşağıya yapıştırın.",
        "paste_area": "Veri Alanı",
        "input_data_table": "Girilen Veri Tablosu",
        "extra_uncert_count": "Ekstra belirsizlik sayısı (0-10):",
        "add_uncertainty": "Ekstra Belirsizlikler",
        "extra_uncert_type": "{} türünü seçin",
        "absolute": "Mutlak",
        "percent": "Yüzde (%)",
        "calculate_button": "🔎 Hesapla",
        "results": "Sonuçlar",
        "download_pdf": "📄 PDF Sonuçlarını İndir",
        "chart_title": "Ölçüm Günlerine Göre Değerler",
        "x_axis": "Ölçüm Günü",
        "y_axis": "Değer",
        "formulas_title": "Formüller",
    },
    "English": {
        "title": "🔬 UncertCalc: Laboratory Measurement Uncertainty Calculator",
        "select_mode": "Select Data Entry Mode",
        "manual": "Manual Entry",
        "paste": "Paste Data (from Excel)",
        "manual_header": "Manual Data Entry Mode",
        "manual_subheader": "Enter results for {}",
        "paste_title": "📋 Paste Mode",
        "paste_subtitle": "Paste measurement data copied from Excel or another table below.",
        "paste_area": "Data Input Area",
        "input_data_table": "Input Data Table",
        "extra_uncert_count": "Number of extra uncertainty components (0-10):",
        "add_uncertainty": "Extra Uncertainties",
        "extra_uncert_type": "Select type for {}",
        "absolute": "Absolute",
        "percent": "Percent (%)",
        "calculate_button": "🔎 Calculate",
        "results": "Results",
        "download_pdf": "📄 Download PDF Results",
        "chart_title": "Measurements by Day",
        "x_axis": "Measurement Day",
        "y_axis": "Value",
        "formulas_title": "Formulas",
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calc_repeatability(data):
    return np.std(data, ddof=1)

def calc_intermediate_precision(data_groups):
    group_means = [np.mean(g) for g in data_groups]
    return np.std(group_means, ddof=1)

def calc_combined_uncertainty(components):
    return np.sqrt(np.sum(np.array(components) ** 2))

def calc_expanded_uncertainty(combined_uncertainty):
    return combined_uncertainty * 2  # k=2

# ------------------------
# Sonuç Hesaplama
# ------------------------
def calculate_results(measurements, extras, lang_texts):
    results = []
    valid_groups = [g for g in measurements if len(g) > 1]
    if not valid_groups:
        st.warning("Yeterli veri yok." if lang_texts == languages["Türkçe"] else "Insufficient data.")
        return [], []

    repeatability = np.mean([calc_repeatability(g) for g in valid_groups])
    inter_precision = calc_intermediate_precision(valid_groups)

    combined = calc_combined_uncertainty([repeatability, inter_precision] + [x[1] for x in extras])
    expanded = calc_expanded_uncertainty(combined)
    rel_expanded = (expanded / np.mean([v for g in valid_groups for v in g])) * 100

    results.append({
        "repeatability": repeatability,
        "intermediate": inter_precision,
        "combined": combined,
        "expanded": expanded,
        "relative_expanded": rel_expanded
    })
    return results, valid_groups

# ------------------------
# Görsel Çizim
# ------------------------
def plot_daily_measurements(data_groups, labels, lang_texts):
    plt.figure()
    plt.boxplot(data_groups, labels=labels)
    plt.xlabel(lang_texts["x_axis"])
    plt.ylabel(lang_texts["y_axis"])
    plt.title(lang_texts["chart_title"])
    st.pyplot(plt)

# ------------------------
# PDF Çıktı
# ------------------------
def create_pdf(results_list, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 11)
    y = 740
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for r in results_list:
        for k, v in r.items():
            c.drawString(60, y, f"{k}: {v:.5f}")
            y -= 20
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Sonuçları Görüntüleme
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.subheader(title)
    for r in results_list:
        st.write(f"**Repeatability (Sr):** {r['repeatability']:.5f}")
        st.write(f"**Intermediate Precision (Si):** {r['intermediate']:.5f}")
        st.write(f"**Combined Uncertainty (uc):** {r['combined']:.5f}")
        st.write(f"**Expanded Uncertainty (U):** {r['expanded']:.5f}")
        st.write(f"**Relative Expanded Uncertainty (%):** {r['relative_expanded']:.2f}%")
    st.markdown("---")

# ------------------------
# Veri Yapıştırma Modu
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])

    # 📘 Rehber Bilgi
    if lang_texts == languages["Türkçe"]:
        st.info("""
        **📘 Veri Girişi Rehberi**
        - Verileri sütunlar halinde, **her sütun bir günü temsil edecek şekilde** yapıştırın.  
        - Sayılar **nokta (.)** ile ayrılmış ondalık biçimde olmalıdır (örnek: 34567.89).  
        - Boş hücreler otomatik olarak dışlanır.  
        - Aşağıda örnek bir tablo formatı gösterilmektedir:
        """)
        example_data = pd.DataFrame({
            "1. Gün": [34644.38, 35909.45, 33255.74],
            "2. Gün": [34324.02, 37027.40, 31319.64],
            "3. Gün": [35447.87, 35285.81, 34387.56]
        })
    else:
        st.info("""
        **📘 Data Input Guide**
        - Paste your measurements **in columns**, where each column represents a separate day.  
        - Use **dot (.)** for decimal separation (example: 34567.89).  
        - Empty cells will be automatically excluded.  
        - Below is an example data format:
        """)
        example_data = pd.DataFrame({
            "DAY-1": [34644.38, 35909.45, 33255.74],
            "DAY-2": [34324.02, 37027.40, 31319.64],
            "DAY-3": [35447.87, 35285.81, 34387.56]
        })
    st.dataframe(example_data, use_container_width=True)
    st.markdown("---")

    # Yapıştırma Alanı
    placeholder_text = (
        "Örnek:\nDAY-1\tDAY-2\tDAY-3\n34644.38\t34324.02\t35447.87\n..."
        if lang_texts == languages["Türkçe"]
        else "Example:\nDAY-1\tDAY-2\tDAY-3\n34644.38\t34324.02\t35447.87\n..."
    )
    pasted_data = st.text_area(lang_texts["paste_area"], height=180, placeholder=placeholder_text)

    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+", header=0, engine='python')
    except Exception as e:
        st.error("Veri formatı hatalı, lütfen kontrol edin." if lang_texts == languages["Türkçe"]
                 else f"Data format error: {str(e)}")
        st.stop()

    df = df.apply(pd.to_numeric, errors='coerce')
    st.subheader(lang_texts["input_data_table"])
    st.dataframe(df)

    measurements = [df[col].dropna().tolist() for col in df.columns]
    overall_avg = np.mean([v for g in measurements for v in g if not np.isnan(v)]) or 1.0

    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extras = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı" if lang_texts == languages["Türkçe"]
                              else f"Extra Uncertainty {i+1} Label", key=f"label_{i}")
        if label:
            type_ = st.radio(lang_texts["extra_uncert_type"].format(label),
                             [lang_texts["absolute"], lang_texts["percent"]], key=f"type_{i}")
            if type_ == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"val_{i}")
                rel_val = value / overall_avg
            else:
                perc = st.number_input(f"{label} (%)", min_value=0.0, value=0.0, step=0.01, key=f"perc_{i}")
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
# Ana Uygulama
# ------------------------
def main():
    st.set_page_config(page_title="UncertCalc", layout="wide")
    lang_choice = st.sidebar.selectbox("🌐 Dil / Language", list(languages.keys()))
    lang_texts = languages[lang_choice]
    st.title(lang_texts["title"])
    mode = st.sidebar.radio(lang_texts["select_mode"], [lang_texts["manual"], lang_texts["paste"]])

    if mode == lang_texts["paste"]:
        run_paste_mode(lang_texts)
    else:
        st.info("Elle veri girişi bölümü yakında eklenecek." if lang_choice == "Türkçe"
                else "Manual entry mode coming soon.")

if __name__ == "__main__":
    main()
