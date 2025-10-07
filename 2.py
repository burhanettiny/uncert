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
        "combined_relative_unc": "Combined Relative Ek Belirsizlik",
        "expanded_uncertainty": "Genişletilmiş Genel Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle",
        "download_pdf": "PDF İndir"
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
        "combined_relative_unc": "Combined Relative Extra Uncertainty",
        "expanded_uncertainty": "Expanded Overall Uncertainty (k=2)",
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "paste_title": "Uncertainty Calculation Application",
        "paste_subtitle": "Developed by B. Yalçınkaya",
        "paste_area": "Paste data here (columns = days)",
        "results": "Results",
        "daily_measurements": "Daily Measurement Results",
        "add_uncertainty": "Add Extra Uncertainty Budget",
        "download_pdf": "Download PDF"
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calculate_average(measurements):
    return np.mean(measurements) if len(measurements) > 0 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if num_measurements_per_day <= 0:
        return float('nan')
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return float('nan')

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

# ------------------------
# PDF Oluşturma Fonksiyonu
# ------------------------
def create_pdf(results_list, lang_texts, filename="results.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    
    y = height - 50
    c.drawString(50, y, lang_texts["results"])
    y -= 30

    for param, value, formula in results_list:
        c.drawString(50, y, f"{param}: {value}   Formula: {formula}")
        y -= 20
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
    df_values = pd.DataFrame([(p, v) for p, v, f in results_list], columns=[lang_texts["results"], "Değer"])
    st.dataframe(df_values)
    st.write(f"### Formüller")
    for param, _, formula in results_list:
        st.markdown(f"**{param}:** ${formula}$", unsafe_allow_html=True)
    return df_values

# ------------------------
# Günlük Grafik
# ------------------------
def plot_daily_measurements(measurements, lang_texts):
    fig, ax = plt.subplots()
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue  # boş günleri atla
        label = f"{'Gün' if lang_texts['manual_header']=='Elle Veri Girişi Modu' else 'Day'} {i+1}"
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)
    ax.set_xticks(range(1, max((len(g) for g in measurements if len(g)>0), default=1)+1))
    ax.set_xlabel("Ölçüm Sayısı" if lang_texts['manual_header']=='Elle Veri Girişi Modu' else "Measurement Number")
    ax.set_ylabel("Değer" if lang_texts['manual_header']=='Elle Veri Girişi Modu' else "Value")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]

    measurements = []
    for col in df.columns:
        group = []
        for val in df[col]:
            try:
                group.append(float(val))
            except:
                continue  # boş hücreleri at
        measurements.append(group)

    all_values = [val for group in measurements for val in group]
    if not all_values:
        st.error("Yapıştırılan veride geçerli sayısal veri bulunamadı!")
        st.stop()

    average_value = np.mean(all_values)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label),
                                  [lang_texts["absolute"], lang_texts["percent"]],
                                  key=f"paste_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / average_value if average_value != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * average_value
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        total_values = sum(len(m) for m in measurements)
        num_groups = len([m for m in measurements if len(m) > 0])
        ss_between = sum(len(m)*(np.mean(m)-average_value)**2 for m in measurements if len(m) > 0)
        ss_within = sum(sum((x-np.mean(m))**2 for x in m) for m in measurements if len(m) > 0)
        df_between = num_groups-1
        df_within = total_values - num_groups
        ms_between = ss_between / df_between if df_between>0 else float('nan')
        ms_within = ss_within / df_within if df_within>0 else float('nan')

        repeatability = calc_repeatability_from_ms(ms_within)
        num_measurements_per_day = np.mean([len(m) for m in measurements if len(m)>0])
        intermediate_precision = calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day)
        relative_repeatability = repeatability / average_value if average_value !=0 else float('nan')
        relative_intermediate_precision = intermediate_precision / average_value if average_value !=0 else float('nan')
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
        combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
        expanded_uncertainty = 2 * combined_relative_unc * average_value
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_uncertainty, average_value)

        results_list = [
            ("Repeatability", f"{repeatability:.4f}", r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}"),
            ("Intermediate Precision", f"{intermediate_precision:.4f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n}}")
        ]

        for label, value, rel_val, input_type in extra_uncertainties:
            if input_type == lang_texts["percent"]:
                results_list.append((label, f"{value:.4f}", r"u_{extra} = \frac{\text{Percent}}{100} \cdot \bar{x}"))

        results_list.extend([
            ("Combined Relative Uncertainty", f"{combined_relative_unc:.4f}", r"u_c = \sqrt{u_{repeat}^2 + u_{IP}^2 + u_{extra}^2}"),
            ("Relative Repeatability", f"{relative_repeatability:.4f}", r"u_{repeat,rel} = \frac{s}{\bar{x}}"),
            ("Relative Intermediate Precision", f"{relative_intermediate_precision:.4f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
            ("Relative Extra Uncertainty", f"{relative_extra_unc:.4f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
            (lang_texts["average_value"], f"{average_value:.4f}", r"\bar{x} = \frac{\sum x_i}{n}"),
            (lang_texts["expanded_uncertainty"], f"{expanded_uncertainty:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{relative_expanded_uncertainty:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
        ])

        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)
        plot_daily_measurements(measurements, lang_texts)

        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results.pdf",
                           mime="application/pdf")

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    language = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[language]

    mode = st.sidebar.radio("Giriş Modu / Input Mode", ["Elle / Manual", "Yapıştır / Paste"])
    if mode.startswith("Elle"):
        st.warning("Elle giriş modu henüz bu güncelleme ile tamamlanmadı. Yapıştırarak giriş kullanın.")
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
