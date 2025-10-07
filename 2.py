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
        "calculate_button": "Sonuçları Hesapla",
        "results": "Sonuçlar",
        "download_pdf": "PDF İndir",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın"
    },
    "English": {
        "calculate_button": "Calculate Results",
        "results": "Results",
        "download_pdf": "Download PDF",
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste data here (columns = days)"
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calc_anova(measurements):
    all_values = [v for g in measurements for v in g]
    grand_mean = np.mean(all_values)
    k = len(measurements)
    n = len(measurements[0]) if k > 0 else 0

    ss_between = n * sum((np.mean(g) - grand_mean)**2 for g in measurements)
    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in measurements)

    df_between = k - 1 if k > 1 else 1
    df_within = k * (n - 1) if n > 1 else 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return ms_between, ms_within

def calc_repeatability(ms_within):
    return np.sqrt(ms_within)

def calc_intermediate_precision(ms_between, ms_within):
    return np.sqrt(ms_within + ms_between)

def calc_combined_uncertainty(intermediate_precision, mean_value):
    return intermediate_precision / mean_value if mean_value != 0 else float('nan')

def calc_expanded_uncertainty(u_c):
    return 2 * u_c

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
        c.drawString(50, y, f"{param}: {value} | {formula}")
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[language]

    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"], height=200)

    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"[\t\s;]+", engine='python', header=None)
    except Exception as e:
        st.error(f"Veri okunamadı: {e}")
        st.stop()

    measurements = []
    for col in df.columns:
        values = df[col].dropna().astype(float).tolist()
        if values:
            measurements.append(values)

    if not measurements:
        st.error("Geçerli sayısal veri bulunamadı!")
        st.stop()

    all_values = [v for g in measurements for v in g]
    mean_value = np.mean(all_values)

    ms_between, ms_within = calc_anova(measurements)
    repeatability = calc_repeatability(ms_within)
    intermediate_precision = calc_intermediate_precision(ms_between, ms_within)
    u_c = calc_combined_uncertainty(intermediate_precision, mean_value)
    U = calc_expanded_uncertainty(u_c)
    relative_U = (U * 100) if not np.isnan(U) else float('nan')

    results_list = [
        ("MS_within", f"{ms_within:.6f}", "MS_within = SS_within / df_within"),
        ("MS_between", f"{ms_between:.6f}", "MS_between = SS_between / df_between"),
        ("Repeatability (s)", f"{repeatability:.6f}", "s = √MS_within"),
        ("Intermediate Precision (s_IP)", f"{intermediate_precision:.6f}", "s_IP = √(MS_within + MS_between)"),
        ("Combined Relative Uncertainty (u_c)", f"{u_c:.6f}", "u_c = s_IP / mean"),
        ("Expanded Uncertainty (k=2)", f"{U:.6f}", "U = 2 × u_c"),
        ("Relative Expanded Uncertainty (%)", f"{relative_U:.3f}", "U_rel = U × 100")
    ]

    st.write("### 📊 Sonuçlar / Results")
    st.dataframe(pd.DataFrame(results_list, columns=["Parameter", "Value", "Formula"]))

    pdf_buffer = create_pdf(results_list, lang_texts)
    st.download_button(label=lang_texts["download_pdf"], data=pdf_buffer,
                       file_name="uncertainty_results.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
