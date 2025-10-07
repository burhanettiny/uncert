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
        "relative_expanded_uncertainty_col": "Relative Expanded Uncertainty (%)",
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın (her sütun = bir gün)",
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

def calculate_standard_uncertainty(measurements):
    return (np.std(measurements, ddof=1) / np.sqrt(len(measurements))) if len(measurements) > 1 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else 0.0

def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if (ms_within is not None and not np.isnan(ms_within) and ms_within >= 0) else 0.0

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    """
    Geleneksel yaklaşıma göre:
    Eğer MS_between > MS_within ise s_IP = sqrt((MS_between - MS_within) / n_bar)
    aksi halde s_IP = 0 (aralarında gerçek bir 'between' varyans farkı yok)
    """
    try:
        if num_measurements_per_day is None or num_measurements_per_day <= 0:
            return 0.0
        if ms_between is None or ms_within is None:
            return 0.0
        if ms_between > ms_within:
            diff = ms_between - ms_within
            return np.sqrt(diff / num_measurements_per_day)
        else:
            return 0.0
    except Exception:
        return 0.0

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if (average_value is not None and average_value != 0) else float('nan')

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
        # value'yi string'e güvenli şekilde çevir
        try:
            val_str = f"{value}"
        except:
            val_str = "N/A"
        c.drawString(50, y, f"{param}: {val_str}   Formula: {formula}")
        y -= 20
        if y < 50:  # yeni sayfa
            c.showPage()
            y = height - 50
    
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Formüller ve Tablo Gösterimi
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.write(f"## {title}")
    # results_list: [(param, value, formula), ...]
    df_values = pd.DataFrame([(p, v) for p, v, f in results_list], columns=[lang_texts["results"], "Değer"])
    st.dataframe(df_values)
    st.write(f"### Formüller")
    for param, _, formula in results_list:
        st.markdown(f"**{param}:** ${formula}$", unsafe_allow_html=True)
    return df_values

# ------------------------
# Günlük Grafik Fonksiyonu
# ------------------------
def plot_daily_measurements(measurements, lang_texts):
    fig, ax = plt.subplots()
    max_len = max((len(g) for g in measurements), default=0)
    for i, group in enumerate(measurements):
        if len(group) == 0:
            continue
        label = f"{'Gün' if lang_texts['manual_header']=='Elle Veri Girişi Modu' else 'Day'} {i+1}"
        ax.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=label)
    if max_len > 0:
        ax.set_xticks(range(1, max_len+1))
    ax.set_xlabel("Ölçüm Sayısı" if lang_texts['manual_header']=='Elle Veri Girişi Modu' else "Measurement Number")
    ax.set_ylabel("Değer" if lang_texts['manual_header']=='Elle Veri Girişi Modu' else "Value")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Elle Giriş Modu
# (Bu mod mevcut; isterseniz boş girişleri kaldırıp dinamik hale getirebiliriz)
# ------------------------
def run_manual_mode(lang_texts):
    st.header(lang_texts["manual_header"])
    # örnek: 3 gün, her gün max 5 tekrar (kullanıcı hepsini doldurmayabilir)
    days = ['1. Gün', '2. Gün', '3. Gün']
    max_repeats = 5
    total_measurements = []

    for day in days:
        st.subheader(lang_texts["manual_subheader"].format(day))
        measurements = []
        for i in range(max_repeats):
            # default None yerine 0.0 verdiğiniz için kullanıcı eksik bırakırsa 0 girilmiş olur
            # Eğer isterseniz default=None ve text_input ile boş bırakılmasına izin verip sonra filtreleyebiliriz.
            value = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["extra_uncert_label"])
    overall_measurements = [val for day in total_measurements for val in day if val is not None]
    overall_avg = calculate_average(overall_measurements) if overall_measurements else 1.0

    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Extra Uncertainty {i+1} Name", value="", key=f"manual_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"manual_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Value", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0.0
            else:
                percent_value = st.number_input(f"{label} Percent (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                relative_value = percent_value / 100.0
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        # flatten ve filtrele (manuel modda kullanıcı eksik değerleri 0 giriyorsa bun da hesaba girer;
        # isterseniz 0'ları da yok sayacak şekilde değiştirebiliriz)
        repeatability_values = [v for day in total_measurements for v in day if v is not None]

        overall_measurements = [val for day in total_measurements for val in day if val is not None]
        overall_avg = calculate_average(overall_measurements) if overall_measurements else 0.0

        # repeatability within (gün içi) - using sample std dev over all repeats
        repeatability_within_days = calculate_repeatability(repeatability_values)

        # intermediate: use day means; if some days are all zeros or missing, consider them appropriately
        day_means = []
        for day in total_measurements:
            vals = [v for v in day if v is not None]
            if len(vals) > 0:
                day_means.append(calculate_average(vals))
            else:
                day_means.append(float('nan'))

        # Use only numeric day means
        valid_day_means = [dm for dm in day_means if not np.isnan(dm)]
        repeatability_between_days = calculate_repeatability(valid_day_means) if len(valid_day_means) > 1 else 0.0

        # relative extra
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))

        # Avoid divide-by-zero
        overall_avg_safe = overall_avg if (overall_avg is not None and overall_avg != 0) else 1.0

        combined_relative_unc = np.sqrt((repeatability_within_days/overall_avg_safe)**2 + (repeatability_between_days/overall_avg_safe)**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * overall_avg_safe
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_overall_uncertainty, overall_avg_safe)

        results_list = [
            ("Repeatability (within days)", f"{repeatability_within_days:.6f}", r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}"),
            ("Intermediate Precision (between days)", f"{repeatability_between_days:.6f}", r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n}}")
        ]

        for label, value, rel_val, input_type in extra_uncertainties:
            if input_type == lang_texts["percent"]:
                results_list.append((label, f"{value:.6f}", r"u_{extra} = \frac{\text{Percent}}{100} \cdot \bar{x}"))

        results_list.extend([
            ("Combined Relative Uncertainty", f"{combined_relative_unc:.6f}", r"u_c = \sqrt{u_{repeat}^2 + u_{IP}^2 + u_{extra}^2}"),
            ("Relative Repeatability", f"{(repeatability_within_days/overall_avg_safe):.6f}", r"u_{repeat,rel} = \frac{s}{\bar{x}}"),
            ("Relative Intermediate Precision", f"{(repeatability_between_days/overall_avg_safe):.6f}", r"u_{IP,rel} = \frac{s_{IP}}{\bar{x}}"),
            ("Relative Extra Uncertainty", f"{relative_extra_unc:.6f}", r"u_{extra,rel} = \sqrt{\sum u_{extra,i}^2}"),
            (lang_texts["average_value"], f"{overall_avg_safe:.6f}", r"\bar{x} = \frac{\sum x_i}{n}"),
            (lang_texts["expanded_uncertainty"], f"{expanded_overall_uncertainty:.6f}", r"U = 2 \cdot u_c \cdot \bar{x}"),
            (lang_texts["relative_expanded_uncertainty_col"], f"{relative_expanded_uncertainty:.6f}", r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
        ])

        display_results_with_formulas(results_list, title=lang_texts["overall_results"], lang_texts=lang_texts)

        # For plot, convert total_measurements to per-day groups but remove empty groups
        groups_for_plot = []
        for day in total_measurements:
            g = [v for v in day if v is not None]
            if len(g) > 0:
                groups_for_plot.append(g)
        plot_daily_measurements(groups_for_plot, lang_texts)

        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results.pdf",
                           mime="application/pdf")

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
        # allow tabs or whitespace separators
        df = pd.read_csv(io.StringIO(pasted_data), sep=r"\s+|\t", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        st.stop()

    # Name columns and rows for display (optional)
    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]

    # build measurements: for each column, collect numeric values and ignore empty/invalid
    measurements = []
    for col in df.columns:
        group = []
        for val in df[col]:
            try:
                v = float(val)
                if not np.isnan(v):
                    group.append(v)
            except:
                # ignore blanks or non-numeric
                continue
        # even if group is empty we append so day count is preserved
        measurements.append(group)

    # If no numeric data found at all -> stop
    all_values = [val for group in measurements for val in group]
    if len(all_values) == 0:
        st.error("Yapıştırılan veride geçerli sayısal veri bulunamadı!")
        st.stop()

    # overall average for extra uncertainty conversions
    overall_avg = np.mean(all_values) if len(all_values) > 0 else 0.0

    # Ekstra Belirsizlik
    num_extra_uncertainties = st.number_input(lang_texts["extra_uncert_count"], min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), [lang_texts["absolute"], lang_texts["percent"]], key=f"paste_type_{i}")
            if input_type == lang_texts["absolute"]:
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0.0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100.0
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value, input_type))

    if st.button(lang_texts["calculate_button"]):
        # ANOVA-style sums
        total_values = sum(len(m) for m in measurements)
        num_groups = len(measurements)

        # overall mean (weighted by actual observations)
        overall_mean = np.mean(all_values) if len(all_values) > 0 else 0.0

        # SS_between: use group means; if a group is empty, skip it (len=0)
        ss_between = 0.0
        for m in measurements:
            if len(m) == 0:
                continue
            group_mean = np.mean(m)
            ss_between += len(m) * (group_mean - overall_mean) ** 2

        # SS_within: sum of squared deviations inside groups (skip empty)
        ss_within = 0.0
        for m in measurements:
            if len(m) <= 1:
                # if group has 0 or 1 observation, its within ss is 0
                continue
            gm = np.mean(m)
            ss_within += sum((x - gm) ** 2 for x in m)

        # Degrees of freedom
        df_between = sum(1 for m in measurements if len(m) > 0) - 1  # number of non-empty groups - 1
        df_within = total_values - sum(1 for m in measurements if len(m) > 0)  # N - k_nonempty

        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 0.0

        # repeatability (within): standard deviation estimate
        repeatability = calc_repeatability_from_ms(ms_within)

        # use average number of measurements per day (n̄) across non-empty days
        nonempty_group_sizes = [len(m) for m in measurements if len(m) > 0]
        num_measurements_per_day = float(np.mean(nonempty_group_sizes)) if len(nonempty_group_sizes) > 0 else 1.0

        # intermediate precision (based on classical formula)
        intermediate_precision = calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day)

        # relative components (safe guard against zero division)
        average_value = overall_mean if overall_mean != 0 else 0.0
        relative_repeatability = (repeatability / average_value) if average_value != 0 else 0.0
        relative_intermediate_precision = (intermediate_precision / average_value) if average_value != 0 else 0.0
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties])) if len(extra_uncertainties) > 0 else 0.0

        combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
        expanded_uncertainty = 2 * combined_relative_unc * average_value
        relative_expanded_uncertainty = calc_relative_expanded_uncertainty(expanded_uncertainty, average_value) if average_value != 0 else float('nan')

        # Prepare results for display (avoid NaN by formatting or using 0 where appropriate)
        def fmt(x, digits=6):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "N/A"
                return f"{float(x):.{digits}f}"
            except:
                return str(x)

        results_list = [
            ("Repeatability (s) [sqrt(MS_within)]", fmt(repeatability, 6), r"s = \sqrt{MS_{within}}"),
            ("Intermediate Precision (s_IP)", fmt(intermediate_precision, 6), r"s_{IP} = \sqrt{\frac{MS_{between} - MS_{within}}{n̄}} (if MS_{between}>MS_{within})"),
            ("MS_within", fmt(ms_within, 6), r"MS_{within} = \frac{SS_{within}}{df_{within}}"),
            ("MS_between", fmt(ms_between, 6), r"MS_{between} = \frac{SS_{between}}{df_{between}}"),
            ("Combined Relative Uncertainty (u_c)", fmt(combined_relative_unc, 6), r"u_c = \sqrt{u_{repeat}^2 + u_{IP}^2 + u_{extra}^2}"),
            ("Expanded Uncertainty (k=2)", fmt(expanded_uncertainty, 6), r"U = 2 \cdot u_c \cdot \bar{x}"),
            (lang_texts["relative_expanded_uncertainty_col"], fmt(relative_expanded_uncertainty, 3), r"U_{rel} = \frac{U}{\bar{x}} \cdot 100")
        ]

        display_results_with_formulas(results_list, title=lang_texts["results"], lang_texts=lang_texts)

        # Görsel için boş olmayan grupları seç
        groups_for_plot = [g for g in measurements if len(g) > 0]
        plot_daily_measurements(groups_for_plot, lang_texts)

        # PDF İndir butonu
        pdf_buffer = create_pdf(results_list, lang_texts)
        st.download_button(label=lang_texts["download_pdf"],
                           data=pdf_buffer,
                           file_name="uncertainty_results.pdf",
                           mime="application/pdf")

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    lang_texts = languages[language]

    mode = st.radio("Veri Giriş Yöntemi / Data Input Method", 
                    ["Elle Giriş", "Yapıştırarak Giriş"] if language=="Türkçe" else ["Manual Input", "Paste Input"])
    if mode in ["Elle Giriş", "Manual Input"]:
        run_manual_mode(lang_texts)
    else:
        run_paste_mode(lang_texts)

if __name__ == "__main__":
    main()
