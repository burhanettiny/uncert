import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calculate_average(measurements):
    return np.mean(measurements) if measurements else float('nan')

def calculate_standard_uncertainty(measurements):
    return (np.std(measurements, ddof=1) / np.sqrt(len(measurements))) if len(measurements) > 1 else float('nan')

def calculate_repeatability(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return float('nan')

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

# ------------------------
# Elle Giriş Modu
# ------------------------
def run_manual_mode():
    st.header("Elle Veri Girişi Modu")
    days = ['1. Gün', '2. Gün', '3. Gün']
    total_measurements = []

    for day in days:
        st.subheader(f"{day} İçin Ölçüm Sonucu Girin")
        measurements = []
        for i in range(5):
            value = st.number_input(f"{day} - Tekrar {i+1}", value=0.0, step=0.01, format="%.2f", key=f"{day}_{i}")
            measurements.append(value)
        total_measurements.append(measurements)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader("Ekstra Belirsizlik Bütçesi")
    overall_measurements = [val for day in total_measurements for val in day]
    overall_avg = calculate_average(overall_measurements) if overall_measurements else 1

    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"manual_label_{i}")
        if label:
            input_type = st.radio(f"{label} için tür seçin", ["Mutlak Değer", "Yüzde"], key=f"manual_type_{i}")
            if input_type == "Mutlak Değer":
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"manual_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"manual_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value))

    if st.button("Sonuçları Hesapla (Elle Giriş)"):
        repeatability_values = []
        for i, day in enumerate(days):
            avg = calculate_average(total_measurements[i])
            uncertainty = calculate_standard_uncertainty(total_measurements[i])
            repeatability = calculate_repeatability(total_measurements[i])
            total_uncertainty = np.sqrt(uncertainty**2 + sum([rel[1]**2 for rel in extra_uncertainties]))
            st.write(f"### {day} Sonuçları")
            st.write(f"**Ortalama:** {avg:.4f}")
            st.write(f"**Belirsizlik (Ekstra dahil):** {total_uncertainty:.4f}")
            st.write(f"**Tekrarlanabilirlik:** {repeatability:.4f}")
            repeatability_values.extend(total_measurements[i])

        overall_measurements = [val for day in total_measurements for val in day]
        overall_avg = calculate_average(overall_measurements)
        repeatability_within_days = calculate_repeatability(repeatability_values)
        repeatability_between_days = calculate_repeatability([calculate_average(day) for day in total_measurements])
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
        combined_relative_unc = np.sqrt((repeatability_within_days/overall_avg)**2 + (repeatability_between_days/overall_avg)**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * overall_avg

        # Sonuç Tablosu
        parametreler = [
            "Tekrarlanabilirlik",
            "Güç İçi Tekrarlanabilirlik",
            "Günler Arası Tekrarlanabilirlik"
        ]
        degerler = [
            f"{repeatability_within_days:.4f}",
            f"{repeatability_within_days:.4f}",
            f"{repeatability_between_days:.4f}"
        ]
        formuller = [
            "s_r = √Σ(x_i - x̄)² / (n-1)",
            "s_r / mean",
            "s_between / mean"
        ]

        for label, abs_val, rel_val in extra_uncertainties:
            parametreler.append(label)
            degerler.append(f"{abs_val:.4f} (Mutlak), {rel_val*100:.2f}% (Relative)")
            formuller.append("Kullanıcı girişi")

        parametreler.extend([
            "Combined Relative Uncertainty",
            "Genişletilmiş Genel Belirsizlik (k=2)",
            "Ortalama Değer"
        ])
        degerler.extend([
            f"{combined_relative_unc:.4f}",
            f"{expanded_overall_uncertainty:.4f}",
            f"{overall_avg:.4f}"
        ])
        formuller.extend([
            "√(repeat² + intermediate² + extra²)",
            "Combined Relative × Ortalama × 2",
            "mean(X)"
        ])

        results_df = pd.DataFrame({
            "Parametre": parametreler,
            "Değer": degerler,
            "Formül": formuller
        })
        st.dataframe(results_df)

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması",
            "subtitle": "B. Yalçınkaya tarafından geliştirildi",
            "paste": "Verileri buraya yapıştırın",
            "results": "Sonuçlar",
            "daily_measurements": "Günlük Ölçüm Sonuçları",
            "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle"
        },
        "English": {
            "title": "Uncertainty Calculation Application",
            "subtitle": "Developed by B. Yalçınkaya",
            "paste": "Paste data here",
            "results": "Results",
            "daily_measurements": "Daily Measurement Results",
            "add_uncertainty": "Add Extra Uncertainty Budget"
        }
    }

    st.title(texts[language]["title"])
    st.caption(texts[language]["subtitle"])
    pasted_data = st.text_area(texts[language]["paste"])

    if not pasted_data:
        st.error("Lütfen veri yapıştırın!")
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
                continue
        measurements.append(group)

    all_values = [val for group in measurements for val in group]
    if not all_values:
        st.error("Geçerli veri bulunamadı!")
        st.stop()
    overall_avg = np.mean(all_values)

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(texts[language]["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(f"{label} için tür seçin_
