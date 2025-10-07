import streamlit as st
import numpy as np
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
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri (günler sütun olacak şekilde) buraya yapıştırın",
        "calculate_button": "Sonuçları Hesapla",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "average_value": "Ortalama Değer",
        "expanded_uncertainty": "Genişletilmiş Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle",
        "extra_uncert_count": "Ekstra Belirsizlik Sayısı",
        "extra_uncert_type": "{} için tür seçin",
        "absolute": "Mutlak",
        "percent": "Yüzde",
        "download_pdf": "PDF İndir"
    }
}

# ------------------------
# PDF oluşturma
# ------------------------
def create_pdf(results_list, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 11)
    y = 750
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, formula in results_list:
        c.drawString(50, y, f"{param}: {value}")
        y -= 16
        c.drawString(70, y, f"Formül: {formula}")
        y -= 20
        if y < 80:
            c.showPage()
            y = 750
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Formül + Sonuç Gösterimi
# ------------------------
def display_results_with_formulas(results_list, title):
    st.subheader(title)
    df = pd.DataFrame([(p, v) for p, v, _ in results_list], columns=["Parametre", "Değer"])
    st.dataframe(df, use_container_width=True)
    st.markdown("### Formüller")
    for param, _, formula in results_list:
        st.latex(formula)

# ------------------------
# Grafik
# ------------------------
def plot_daily_measurements(measurements, col_names):
    fig, ax = plt.subplots()
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = col_names[i]
        ax.plot(range(1, len(group)+1), group, marker='o', label=label)
    ax.set_xlabel("Tekrar No")
    ax.set_ylabel("Ölçüm Değeri")
    ax.set_title("Günlük Ölçümler")
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Asıl hesaplama (eksik verileri dışlayan)
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])

    pasted_data = st.text_area(lang_texts["paste_area"], height=200)
    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+", header=None, engine="python")
    except Exception as e:
        st.error(f"Veri okunamadı: {str(e)}")
        st.stop()

    df.columns = [f"Gün {i+1}" for i in range(df.shape[1])]
    df = df.apply(pd.to_numeric, errors="coerce")

    # Eksik verileri dışla
    measurements = [df[col].dropna().tolist() for col in df.columns]
    valid_groups = [g for g in measurements if len(g) > 0]
    if len(valid_groups) < 2:
        st.error("Analiz için en az iki dolu sütun (gün) gerekli!")
        st.stop()

    # Hesaplamalar
    means = [np.mean(g) for g in valid_groups]
    ns = [len(g) for g in valid_groups]
    N = sum(ns)
    k = len(valid_groups)
    grand_mean = np.average(means, weights=ns)

    # ANOVA benzeri hesap
    ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in valid_groups)
    ss_between = sum(ns[i] * (means[i] - grand_mean)**2 for i in range(k))
    df_within = N - k
    df_between = k - 1
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    ms_between = ss_between / df_between if df_between > 0 else 0.0

    repeatability = np.sqrt(ms_within)
    n_eff = np.mean(ns)
    intermediate_precision = np.sqrt((ms_between - ms_within) / n_eff) if ms_between > ms_within else 0.0

    all_values = [x for g in valid_groups for x in g]
    avg_value = np.mean(all_values)

    rel_repeat = repeatability / avg_value if avg_value != 0 else 0.0
    rel_ip = intermediate_precision / avg_value if avg_value != 0 else 0.0
    combined_rel_unc = np.sqrt(rel_repeat**2 + rel_ip**2)
    U = 2 * combined_rel_unc * avg_value
    U_rel = (U / avg_value) * 100 if avg_value != 0 else 0.0

    # Sonuç listesi (LaTeX formülleriyle)
    results_list = [
        ("Repeatability", f"{repeatability:.3f}", r"s_r = \sqrt{MS_{within}}"),
        ("Intermediate Precision", f"{intermediate_precision:.3f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n_{eff}}}"),
        ("Combined Relative Uncertainty", f"{combined_rel_unc:.5f}", r"u_c = \sqrt{u_{r}^2 + u_{IP}^2}"),
        ("Relative Repeatability", f"{rel_repeat:.5f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
        ("Relative Intermediate Precision", f"{rel_ip:.5f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
        (lang_texts["average_value"], f"{avg_value:.3f}", r"\bar{x} = \frac{\sum x_i}{n}"),
        (lang_texts["expanded_uncertainty"], f"{U:.3f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
        (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.3f}", r"U_{rel} = \frac{U}{\bar{x}} \times 100")
    ]

    display_results_with_formulas(results_list, lang_texts["results"])
    plot_daily_measurements(valid_groups, df.columns)

    pdf_buffer = create_pdf(results_list, lang_texts)
    st.download_button(lang_texts["download_pdf"], data=pdf_buffer, file_name="uncertainty_results.pdf", mime="application/pdf")

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    lang_texts = languages["Türkçe"]
    run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
