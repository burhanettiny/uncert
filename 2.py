import numpy as np
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calculate_average(measurements):
    return np.mean(measurements) if measurements else float('nan')

def calculate_standard_uncertainty(measurements):
    return np.std(measurements, ddof=1) if len(measurements) > 1 else float('nan')

def calc_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return 0.0

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    st.title("Belirsizlik Hesaplama Uygulaması")
    st.caption("B. Yalçınkaya tarafından geliştirildi")

    pasted_data = st.text_area("Verileri buraya yapıştırın (boşluk veya tab ile ayrılmış)")

    if not pasted_data:
        st.warning("Lütfen veri yapıştırın!")
        st.stop()

    try:
        pasted_data = pasted_data.replace(',', '.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Verileri kontrol edin. ({str(e)})")
        st.stop()

    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]

    # Sayısal değerleri çek
    measurements = []
    for col in df.columns:
        group = []
        for val in df[col]:
            try:
                group.append(float(val))
            except:
                continue
        measurements.append(group)

    if not any(measurements):
        st.error("Geçerli sayısal veri bulunamadı!")
        st.stop()

    overall_avg = np.mean([val for group in measurements for val in group])
    num_days = len(measurements)
    num_measurements_per_day = len(measurements[0])

    # ------------------------
    # Ekstra Belirsizlik
    # ------------------------
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader("Ekstra Belirsizlik Bütçesi (Genel uygulanacak)")
    for i in range(num_extra_uncertainties):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", value="", key=f"extra_label_{i}")
        if label:
            input_type = st.radio(f"{label} türü", ["Mutlak Değer", "Yüzde"], key=f"extra_type_{i}")
            if input_type == "Mutlak Değer":
                value = st.number_input(f"{label} Değeri", min_value=0.0, value=0.0, step=0.01, key=f"extra_val_{i}")
                relative_value = value / overall_avg if overall_avg != 0 else 0
            else:
                percent_value = st.number_input(f"{label} Yüzde (%)", min_value=0.0, value=0.0, step=0.01, key=f"extra_percent_{i}")
                relative_value = percent_value / 100
                value = relative_value * overall_avg
            extra_uncertainties.append((label, value, relative_value))

    # ------------------------
    # Günlük Hesaplamalar
    # ------------------------
    daily_results = []
    for i, day_measurements in enumerate(measurements):
        avg = calculate_average(day_measurements)
        repeat = calculate_standard_uncertainty(day_measurements)  # Gün içi tekrarlanabilirlik
        daily_results.append({
            "Gün": f"{i+1}. Gün",
            "Ortalama": avg,
            "Tekrarlanabilirlik (std)": repeat
        })

    # ------------------------
    # Genel Hesaplamalar
    # ------------------------
    overall_measurements = [val for group in measurements for val in group]
    overall_avg = calculate_average(overall_measurements)

    # MS hesapları
    ss_between = sum(len(m)*(np.mean(m)-overall_avg)**2 for m in measurements)
    ss_within = sum(sum((x-np.mean(m))**2 for x in m) for m in measurements)
    df_between = num_days -1
    df_within = num_days*num_measurements_per_day - num_days
    ms_between = ss_between / df_between if df_between>0 else 0
    ms_within = ss_within / df_within if df_within>0 else 0

    repeatability_within = calc_repeatability(ms_within)
    intermediate_precision = calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day)

    relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
    combined_relative_unc = np.sqrt((repeatability_within/overall_avg)**2 + (intermediate_precision/overall_avg)**2 + relative_extra_unc**2)
    expanded_overall_unc = 2 * combined_relative_unc * overall_avg

    # ------------------------
    # Sonuç Tablosu
    # ------------------------
    report_df = pd.DataFrame(daily_results)
    st.subheader("Günlük Ölçümler")
    st.dataframe(report_df)

    st.subheader("Genel Sonuçlar")
    st.write(f"**Genel Ortalama:** {overall_avg:.4f}")
    st.write(f"**Tekrarlanabilirlik (Gün içi, std):** {repeatability_within:.4f}")
    st.write(f"**Tekrarlanabilirlik (Günler arası / Intermediate Precision):** {intermediate_precision:.4f}")
    st.write(f"**Relative Ek Belirsizlik:** {relative_extra_unc:.4f}")
    st.write(f"**Genişletilmiş Genel Belirsizlik (k=2):** {expanded_overall_unc:.4f}")

    # ------------------------
    # Grafikler
    # ------------------------
    fig, ax = plt.subplots(figsize=(8,5))
    for i, group in enumerate(measurements):
        x = list(range(1,len(group)+1))
        ax.errorbar(x, group, yerr=np.std(group, ddof=1), fmt='o', capsize=5, label=f"Gün {i+1}")
    ax.axhline(y=overall_avg, color='black', linestyle='-', linewidth=2, label="Genel Ortalama")
    ax.set_xlabel("Ölçüm Sayısı")
    ax.set_ylabel("Değer")
    ax.set_title("Günlük Ölçümler ve Hata Barları")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
