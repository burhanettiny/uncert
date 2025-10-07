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
# Hesaplama Fonksiyonları (güvenli)
# ------------------------
def safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_std(arr):
    return np.std(arr, ddof=1) if len(arr) > 1 else 0.0

def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if ms_within > 0 else 0.0

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if num_measurements_per_day <= 0:
        return 0.0
    diff = ms_between - ms_within
    return np.sqrt(diff / num_measurements_per_day) if diff > 0 else 0.0

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else 0.0

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
    df_values = pd.DataFrame([(p, v) for p, v, f in results_list], columns=[lang_texts["results"], "Değer"])
    st.dataframe(df_values)
    st.write("### Formüller")
    for param, _, formula in results_list:
        st.markdown(f"**{param}:** `{formula}`")
    return df_values

# ------------------------
# Günlük Grafik (orijinal sütun indeksine göre)
# ------------------------
def plot_daily_measurements(measurements, col_names, lang_texts):
    fig, ax = plt.subplots()
    max_len = max((len(g) for g in measurements), default=1)
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = col_names[i] if i < len(col_names) else f"Gün {i+1}"
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)
    ax.set_xticks(range(1, max_len+1))
    ax.set_xlabel("Ölçüm Sayısı")
    ax.set_ylabel("Değer")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Elle Giriş Modu (korunmuş, basit)
# ------------------------
def run_manual_mode(lang_texts):
    st.header(lang_texts["manual_header"])
    days = ['1. Gün', '2. Gün', '3. Gün']
    total_measurements = []

    for day in days:
        st.subheader(lang_texts["manual_subheader"].format(day))
        measurements = []
        for i in range(5):
            value = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["extra_uncert_label"])
    overall_measurements = [val for day in total_measurements for val in day]
    overall_avg = safe_mean(overall_measurements) if overall_measurements else 1.0

    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Extra Uncertainty {i+1} Name", value="", key=f"manual_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"manual_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Value", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Percent (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        # Flatten all manual values (they are filled with zeros by default — user may want that)
        all_vals = [v for day in total_measurements for v in day]
        average_value = safe_mean(all_vals)
        # within-day repeatability (using pooled sd)
        repeatability_within_days = safe_std([v for day in total_measurements for v in day])
        # between-day repeatability: use day means but skip empty-like days
        day_means = [safe_mean(day) for day in total_measurements]
        repeatability_between_days = safe_std(day_means)
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
        # avoid division by zero
        u_repeat_rel = repeatability_within_days / average_value if average_value != 0 else 0.0
        u_ip_rel = repeatability_between_days / average_value if average_value != 0 else 0.0
        combined_relative_unc = np.sqrt(u_repeat_rel**2 + u_ip_rel**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * average_value
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_overall_uncertainty, average_value)

        results_list = [
            ("Repeatability", f"{repeatability_within_days:.4f}", r"s = sqrt(var_pooled)"),
            ("Intermediate Precision", f"{repeatability_between_days:.4f}", r"s_{IP} = std(day means)")
        ]

        for label, value, rel_val, input_type in extra_uncertainties:
            if input_type == lang_texts["percent"]:
                results_list.append((label, f"{value:.4f}", r"u_{extra} = percent/100 * mean"))

        results_list.extend([
            ("Combined Relative Uncertainty", f"{combined_relative_unc:.4f}", r"u_c = sqrt(u_repeat^2 + u_IP^2 + u_extra^2)"),
            ("Relative Repeatability", f"{u_repeat_rel:.4f}", r"u_{repeat,rel} = s / x̄"),
            ("Relative Intermediate Precision", f"{u_ip_rel:.4f}", r"u_{IP,rel} = s_{IP} / x̄"),
            ("Relative Extra Uncertainty", f"{relative_extra_unc:.4f}", r"u_{extra,rel} = sqrt(sum u_extra^2)"),
            (lang_texts["average_value"], f"{average_value:.4f}", r"\bar{x} = sum(x)/n"),
            (lang_texts["expanded_uncertainty"], f"{expanded_overall_uncertainty:.4f}", r"U = 2 * u_c * x̄"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{relative_expanded_uncertainty:.4f}", r"U_rel = U / x̄ * 100")
        ])

        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)
        plot_daily_measurements(total_measurements, [f"{d}" for d in days], lang_texts)
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results_manual.pdf",
                           mime="application/pdf")

# ------------------------
# Yapıştırarak Giriş Modu (güncellendi: eksikleri dışla, değişken tekrar sayısı destekli)
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    df = df.apply(pd.to_numeric, errors='coerce')

    measurements = [df[col].dropna().tolist() for col in df.columns]
    all_values = [v for g in measurements for v in g if not np.isnan(v)]
    if not all_values:
        st.error("Yapıştırılan veride geçerli sayısal veri bulunamadı!")
        st.stop()

    overall_avg = np.mean(all_values)

    # --- Ek belirsizlik bütçesi ---
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        k = len(measurements)
        n_list = [len(m) for m in measurements if len(m) > 0]
        N = sum(n_list)

        means = [np.mean(m) for m in measurements if len(m) > 0]
        grand_mean = np.average(means, weights=n_list)

        # --- SS hesapları ---
        ss_within = sum(sum((x - np.mean(m))**2 for x in m) for m in measurements if len(m) > 1)
        ss_between = sum(n_list[i] * (means[i] - grand_mean)**2 for i in range(len(means)))

        df_within = N - k
        df_between = k - 1
        ms_within = ss_within / df_within if df_within > 0 else float('nan')
        ms_between = ss_between / df_between if df_between > 0 else float('nan')

        # Repeatability
        repeatability = np.sqrt(ms_within)

        # n_eff: efektif tekrar sayısı
        n_eff = np.mean(n_list)

        # Intermediate Precision
        if ms_between > ms_within:
            intermediate_precision = np.sqrt((ms_between - ms_within) / n_eff)
        else:
            intermediate_precision = 0.0

        # --- Göreceli değerler ---
        rel_r = repeatability / grand_mean if grand_mean != 0 else float('nan')
        rel_ip = intermediate_precision / grand_mean if grand_mean != 0 else float('nan')
        rel_extra = np.sqrt(sum([e[2]**2 for e in extra_uncertainties]))
        u_c = np.sqrt(rel_r**2 + rel_ip**2 + rel_extra**2)
        U = 2 * u_c * grand_mean
        U_rel = (U / grand_mean) * 100 if grand_mean != 0 else float('nan')

        results_list = [
            ("Repeatability", f"{repeatability:.3f}", r"s_r = \sqrt{MS_{within}}"),
            ("Intermediate Precision", f"{intermediate_precision:.3f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n_{eff}}}"),
            ("Combined Relative Uncertainty", f"{u_c:.4f}", r"u_c = \sqrt{u_{r}^2 + u_{IP}^2 + u_{extra}^2}"),
            ("Relative Repeatability", f"{rel_r:.4f}", r"u_{r,rel} = \frac{s_r}{\bar{x}}"),
            ("Relative Intermediate Precision", f"{rel_ip:.4f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
            ("Relative Extra Uncertainty", f"{rel_extra:.4f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
            (lang_texts["average_value"], f"{grand_mean:.4f}", r"\bar{x} = \frac{\sum x_i}{n}"),
            (lang_texts["expanded_uncertainty"], f"{U:.4f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{U_rel:.4f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
        ]

        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)
        plot_daily_measurements(measurements, lang_texts)

        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(
            label=lang_texts["download_pdf"],
            data=pdf_buffer,
            file_name="uncertainty_results.pdf",
            mime="application/pdf"
        )

# ------------------------
# Main
# ------------------------
def main():
    st.sidebar.title("Ayarlar / Settings")
    lang_choice = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[lang_choice]

    mode = st.sidebar.radio("Giriş Modu / Input Mode", ["Elle / Manual", "Yapıştır / Paste"])
    if mode.startswith("Elle"):
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
