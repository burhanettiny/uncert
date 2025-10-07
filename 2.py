import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------------------ #
# Dil Metinleri
# ------------------------ #
languages = {
    "Türkçe": {
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri (günler sütun olacak şekilde) buraya yapıştırın",
        "calculate_button": "Sonuçları Hesapla",
        "results": "Sonuçlar",
        "average_value": "Ortalama Değer",
        "expanded_uncertainty": "Genişletilmiş Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)",
        "download_pdf": "PDF İndir",
        "daily_measurements": "Günlük Ölçümler"
    },
    "English": {
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste your data here (columns = days)",
        "calculate_button": "Calculate Results",
        "results": "Results",
        "average_value": "Average Value",
        "expanded_uncertainty": "Expanded Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "download_pdf": "Download PDF",
        "daily_measurements": "Daily Measurements"
    }
}

# ------------------------ #
# PDF oluşturma
# ------------------------ #
def create_pdf(results, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 11)
    y = height - 50
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, formula in results:
        c.drawString(50, y, f"{param}: {value}   Formül: {formula}")
        y -= 18
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------ #
# Veri Görselleştirme
# ------------------------ #
def plot_daily_measurements(measurements, lang_texts):
    fig, ax = plt.subplots()
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
    ax.set_xlabel("Tekrar No")
    ax.set_ylabel("Değer")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------ #
# Hesaplama
# ------------------------ #
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])

    pasted_data = st.text_area(lang_texts["paste_area"], height=200)
    if not pasted_data.strip():
        st.stop()

    try:
        pasted_data = pasted_data.replace(",", ".")
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+", header=None, engine="python")
        df = df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        st.error(f"Veri hatası: {e}")
        st.stop()

    st.write("### 🔹 Önizleme")
    st.dataframe(df)

    measurements = [df[col].dropna().values for col in df.columns]
    valid_groups = [g for g in measurements if len(g) > 0]

    if len(valid_groups) < 2:
        st.error("Analiz için en az iki dolu sütun (gün) gerekli!")
        st.stop()

    if st.button(lang_texts["calculate_button"]):
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
        intermediate_precision = np.sqrt((ms_between - ms_within) / n_eff) if ms_between > ms_within else 0

        all_values = [x for g in valid_groups for x in g]
        average_value = np.mean(all_values)

        rel_repeatability = repeatability / average_value if average_value != 0 else 0
        rel_intermediate_precision = intermediate_precision / average_value if average_value != 0 else 0

        u_c = np.sqrt(rel_repeatability**2 + rel_intermediate_precision**2)
        U = 2 * u_c * average_value
        U_rel = (U / average_value) * 100 if average_value != 0 else 0

        results = [
            ("Repeatability", f"{repeatability:.3f}", r"$s_r = \sqrt{MS_{within}}$"),
            ("Intermediate Precision", f"{intermediate_precision:.3f}", r"$s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n_{eff}}}$"),
            ("Average Value", f"{average_value:.3f}", r"$\bar{x} = \frac{\sum x_i}{n}$"),
            ("Combined Relative Uncertainty", f"{u_c:.4f}", r"$u_c = \sqrt{u_{r}^2 + u_{IP}^2}$"),
            (lang_texts["expanded_uncertainty"], f"{U:.3f}", r"$U = 2 \times u_c \times \bar{x}$"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.3f}", r"$U_{rel} = \frac{U}{\bar{x}} \times 100$")
        ]

        st.write("## 🔹 Sonuçlar")
        for name, val, formula in results:
            st.latex(formula)
            st.write(f"**{name}:** {val}")

        plot_daily_measurements(valid_groups, lang_texts)

        pdf_buf = create_pdf(results, lang_texts)
        st.download_button("📄 " + lang_texts["download_pdf"], data=pdf_buf, file_name="uncertainty_results.pdf", mime="application/pdf")

# ------------------------ #
# Ana Program
# ------------------------ #
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]
    run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
