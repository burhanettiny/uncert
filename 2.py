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
# Ana Fonksiyon
# ------------------------
def main():
    st.title("Belirsizlik Hesaplama ve Rapor Uygulaması")
    st.caption("B. Yalçınkaya tarafından geliştirildi")

    mode = st.radio("Veri Giriş Yöntemi / Data Input Method", ["Elle Giriş", "Yapıştırarak Giriş"])

    if mode == "Elle Giriş":
        run_manual_mode()
    else:
        run_paste_mode()

# ------------------------
# Yapıştırarak Giriş Modu
# ------------------------
def run_paste_mode():
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

    # Kolon ve indeks isimleri
    df.columns = [f"{i+1}. Gün" for i in range(df.shape[1])]
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]

    # Sayısal olmayan veya boş hücreleri filtrele
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

    # Ekstra belirsizlik
    num_extra_uncertainties = st.number_input("Ekstra Belirsizlik Bütçesi Sayısı", min_value=0, max_value=10, value=0, step=1)
    extra_uncertainties = []
    st.subheader("Ekstra Belirsizlik Bütçesi")
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
    repeatability_within_days = []

    for i, day_measurements in enumerate(measurements):
        avg = calculate_average(day_measurements)
        std_unc = calculate_standard_uncertainty(day_measurements)
        repeat = calculate_repeatability(day_measurements)
        combined_unc = np.sqrt(std_unc**2 + sum([v**2 for _,v,_ in extra_uncertainties]))
        expanded_unc = combined_unc * 2
        repeatability_within_days.extend(day_measurements)
        daily_results.append({
            "Gün": f"{i+1}. Gün",
            "Ortalama": avg,
            "Tekrarlanabilirlik": repeat,
            "Belirsizlik (Ekstra dahil)": combined_unc,
            "Genişletilmiş Belirsizlik (k=2)": expanded_unc
        })

    # ------------------------
    # Genel Hesaplamalar
    # ------------------------
    overall_measurements = [val for group in measurements for val in group]
    overall_avg = calculate_average(overall_measurements)
    repeatability_within = calculate_repeatability(repeatability_within_days)
    repeatability_between = calculate_repeatability([calculate_average(day) for day in measurements])
    relative_extra_unc = np.sqrt(sum([rel[2]**2 for rel in extra_uncertainties]))
    combined_relative_unc = np.sqrt((repeatability_within/overall_avg)**2 + (repeatability_between/overall_avg)**2 + relative_extra_unc**2)
    expanded_overall_unc = 2 * combined_relative_unc * overall_avg

    # ------------------------
    # Sonuç Tablosu
    # ------------------------
    report_df = pd.DataFrame(daily_results)
    st.subheader("Günlük ve Genel Belirsizlik Raporu")
    st.dataframe(report_df)

    st.write("### Genel Sonuçlar")
    st.write(f"**Genel Ortalama:** {overall_avg:.4f}")
    st.write(f"**Tekrarlanabilirlik (Gün içi):** {repeatability_within:.4f}")
    st.write(f"**Tekrarlanabilirlik (Günler arası):** {repeatability_between:.4f}")
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

# ------------------------
# Elle Giriş Modu
# ------------------------
def run_manual_mode():
    st.info("Elle giriş modunda, gün ve tekrar sayısını kendiniz belirleyebilirsiniz. Yapıştırarak giriş için yapıştırma modunu kullanın.")
    # Elle giriş, paste_mode ile aynı mantıkla uygulanabilir
    st.warning("Elle giriş raporu henüz entegre edilmedi. Lütfen yapıştırma modunu kullanın veya isteğe göre genişletebiliriz.")

if __name__ == "__main__":
    main()
