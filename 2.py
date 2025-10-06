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

    all_values = [val for day in total_measurements for val in day]
    overall_avg = calculate_average(all_values) if all_values else 1  # bölme için default

    # ------------------------
    # Ekstra Belirsizlik Bütçesi
    # ------------------------
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader("Ekstra Belirsizlik Bütçesi")
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
        repeatability_within_days = calculate_repeatability([val for day in total_measurements for val in day])
        repeatability_between_days = calculate_repeatability([calculate_average(day) for day in total_measurements if len(day) > 0])
        relative_repeatability = repeatability_within_days / overall_avg if overall_avg != 0 else float('nan')
        relative_intermediate_precision = repeatability_between_days / overall_avg if overall_avg != 0 else float('nan')
        relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
        combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
        expanded_overall_uncertainty = 2 * combined_relative_unc * overall_avg

        # Sonuç tablosu
        results_df = pd.DataFrame({
            "Parametre": [
                "Güç İçi Tekrarlanabilirlik",
                "Günler Arası Tekrarlanabilirlik",
                *[rel[0] for rel in extra_uncertainties],
                "Combined Relative Uncertainty"
            ],
            "Mutlak Değer": [
                f"{repeatability_within_days:.4f}",
                f"{repeatability_between_days:.4f}",
                *[f"{rel[1]:.4f}" for rel in extra_uncertainties],
                f"{combined_relative_unc*overall_avg:.4f}"
            ],
            "Yüzde (%)": [
                f"{(relative_repeatability*100):.2f}",
                f"{(relative_intermediate_precision*100):.2f}",
                *[f"{rel[2]*100:.2f}" for rel in extra_uncertainties],
                f"{combined_relative_unc*100:.2f}"
            ]
        })
        st.write("## Sonuçlar")
        st.dataframe(results_df)
        st.write(f"**Genel Ortalama:** {overall_avg:.4f}")
        st.write(f"**Genişletilmiş Genel Belirsizlik (k=2):** {expanded_overall_uncertainty:.4f}")

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
            "average_value": "Ortalama Değer",
            "expanded_uncertainty": "Expanded Uncertainty (k=2)",
            "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle"
        },
        "English": {
            "title": "Uncertainty Calculation Application",
            "subtitle": "Developed by B. Yalçınkaya",
            "paste": "Paste data here",
            "results": "Results",
            "daily_measurements": "Daily Measurement Results",
            "average_value": "Average Value",
            "expanded_uncertainty": "Expanded Uncertainty (k=2)",
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
        if group:
            measurements.append(group)

    all_values = [val for group in measurements for val in group]
    overall_avg = np.mean(all_values)

    # ------------------------
    # Ekstra Belirsizlik Bütçesi
    # ------------------------
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader(texts[language]["add_uncertainty"])
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"paste_label_{i}")
        if label:
            input_type = st.radio(f"{label} için tür seçin", ["Mutlak Değer", "Yüzde"], key=f"paste_type_{i}")
            if input_type == "Mutlak Değer":
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"paste_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"paste_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value))

    total_values = sum(len(m) for m in measurements)
    num_groups = len(measurements)
    average_value = np.mean(all_values)
    ss_between = sum(len(m) * (np.mean(m) - average_value) ** 2 for m in measurements)
    ss_within = sum(sum((x - np.mean(m)) ** 2 for x in m) for m in measurements)
    df_between = num_groups - 1
    df_within = total_values - num_groups
    ms_between = ss_between / df_between if df_between > 0 else float('nan')
    ms_within = ss_within / df_within if df_within > 0 else float('nan')

    repeatability_within_days = calc_repeatability_from_ms(ms_within)
    repeatability_between_days = calc_repeatability([np.mean(m) for m in measurements if len(m) > 0])
    relative_repeatability = repeatability_within_days / average_value if average_value != 0 else float('nan')
    relative_intermediate_precision = repeatability_between_days / average_value if average_value != 0 else float('nan')
    relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
    combined_relative_unc = np.sqrt(relative_repeatability**2 + relative_intermediate_precision**2 + relative_extra_unc**2)
    expanded_overall_uncertainty = 2 * combined_relative_unc * average_value

    results_df = pd.DataFrame({
        "Parametre": [
            "Güç İçi Tekrarlanabilirlik",
            "Günler Arası Tekrarlanabilirlik",
            *[rel[0] for rel in extra_uncertainties],
            "Combined Relative Uncertainty"
        ],
        "Mutlak Değer": [
            f"{repeatability_within_days:.4f}",
            f"{repeatability_between_days:.4f}",
            *[f"{rel[1]:.4f}" for rel in extra_uncertainties],
            f"{combined_relative_unc*average_value:.4f}"
        ],
        "Yüzde (%)": [
            f"{(relative_repeatability*100):.2f}",
            f"{(relative_intermediate_precision*100):.2f}",
            *[f"{rel[2]*100:.2f}" for rel in extra_uncertainties],
            f"{combined_relative_unc*100:.2f}"
        ]
    })
    st.write(texts[language]["results"])
    st.dataframe(results_df)
    st.write(f"**Genel Ortalama:** {average_value:.4f}")
    st.write(f"**Genişletilmiş Genel Belirsizlik (k=2):** {expanded_overall_uncertainty:.4f}")

    # ------------------------
    # Günlük Grafik
    # ------------------------
    fig1, ax1 = plt.subplots()
    for i, group in enumerate(measurements):
        ax1.plot(range(1, len(group)+1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
    ax1.set_xlabel("Ölçüm Sayısı")
    ax1.set_ylabel("Değer")
    ax1.set_title(texts[language]["daily_measurements"])
    ax1.legend()
    st.pyplot(fig1)

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    mode = st.radio("Veri Giriş Yöntemi / Data Input Method", ["Elle Giriş", "Yapıştırarak Giriş"])
    if mode == "Elle Giriş":
        run_manual_mode()
    else:
        run_paste_mode()

if __name__ == "__main__":
    main()
