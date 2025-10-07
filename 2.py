import numpy as np
import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ------------------------
# Dil Metinleri
# ------------------------
languages = {
    "Türkçe": {
        "paste_title": "Belirsizlik Hesaplama Uygulaması",
        "paste_subtitle": "B. Yalçınkaya tarafından geliştirildi",
        "paste_area": "Verileri günlük dikey olacak şekilde buraya yapıştırın",
        "results": "Sonuçlar",
        "daily_measurements": "Günlük Ölçüm Sonuçları",
        "extra_uncert_count": "Ekstra Belirsizlik Bütçesi Sayısı",
        "extra_uncert_type": "{} için tür seçin",
        "absolute": "Mutlak Değer",
        "percent": "Yüzde",
        "add_uncertainty": "Ekstra Belirsizlik Bütçesi Ekle",
        "calculate_button": "Sonuçları Hesapla",
        "average_value": "Ortalama Değer",
        "expanded_uncertainty": "Genişletilmiş Genel Belirsizlik (k=2)",
        "relative_expanded_uncertainty_col": "Göreceli Genişletilmiş Belirsizlik (%)"
    }
}

# ------------------------
# Hesaplama Fonksiyonları
# ------------------------
def calc_repeatability_from_ms(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else 0

def calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    return np.sqrt(max(ms_between - ms_within, 0)/num_measurements_per_day) if num_measurements_per_day>0 else 0

def calc_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else 0

# ------------------------
# PDF
# ------------------------
def create_pdf(results_list, lang_texts):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)
    y = height-50
    c.drawString(50, y, lang_texts["results"])
    y -= 30
    for param, value, formula in results_list:
        c.drawString(50, y, f"{param}: {value}   Formula: {formula}")
        y -= 20
        if y < 50:
            c.showPage()
            y = height-50
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------
# Sonuç Gösterimi
# ------------------------
def display_results_with_formulas(results_list, title, lang_texts):
    st.write(f"## {title}")
    df_values = pd.DataFrame([(p,v) for p,v,f in results_list], columns=[lang_texts["results"], "Değer"])
    st.dataframe(df_values)
    return df_values

# ------------------------
# Günlük Grafik
# ------------------------
def plot_daily_measurements(measurements, lang_texts):
    fig, ax = plt.subplots()
    for i, group in enumerate(measurements):
        if len(group)==0:
            continue
        label = f"Gün {i+1}"
        ax.plot(range(1,len(group)+1), group, marker='o', linestyle='-', label=label)
    ax.set_xticks(range(1, max([len(g) for g in measurements if len(g)>0]+[1])))
    ax.set_xlabel("Ölçüm Sayısı")
    ax.set_ylabel("Değer")
    ax.set_title(lang_texts["daily_measurements"])
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Yapıştırarak Giriş
# ------------------------
def run_paste_mode(lang_texts):
    st.title(lang_texts["paste_title"])
    st.caption(lang_texts["paste_subtitle"])
    pasted_data = st.text_area(lang_texts["paste_area"])
    if not pasted_data:
        st.stop()
    try:
        pasted_data = pasted_data.replace(',','.')
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None)
    except Exception as e:
        st.error(f"Hata! ({str(e)})")
        st.stop()

    # Ölçümleri gruplara ayır
    measurements=[]
    for col in df.columns:
        group=[]
        for val in df[col]:
            try:
                group.append(float(val))
            except:
                continue
        measurements.append(group)

    all_values=[v for g in measurements for v in g]
    if len(all_values)==0:
        st.error("Geçerli veri bulunamadı!")
        st.stop()
    average_value = np.mean(all_values)

    # Ekstra belirsizlik
    num_extra = st.number_input(lang_texts["extra_uncert_count"], min_value=0,max_value=10,value=0)
    extra_unc=[]
    st.subheader(lang_texts["add_uncertainty"])
    for i in range(num_extra):
        label = st.text_input(f"Ekstra Belirsizlik {i+1} Adı", key=f"label_{i}")
        if label:
            input_type = st.radio(lang_texts["extra_uncert_type"].format(label), ["Mutlak Değer","Yüzde"], key=f"type_{i}")
            if input_type=="Mutlak Değer":
                value=st.number_input(f"{label} Değeri", min_value=0.0,value=0.0, step=0.01,key=f"value_{i}")
                rel=value/average_value if average_value!=0 else 0
            else:
                percent=st.number_input(f"{label} Yüzde (%)", min_value=0.0,value=0.0,step=0.01,key=f"percent_{i}")
                rel=percent/100
                value=rel*average_value
            extra_unc.append((label,value,rel,input_type))

    if st.button(lang_texts["calculate_button"]):
        valid_groups=[g for g in measurements if len(g)>0]
        total_values=sum(len(g) for g in valid_groups)
        num_groups=len(valid_groups)

        if num_groups==0:
            st.error("Geçerli veri yok!")
            st.stop()

        ss_between=sum(len(g)*(np.mean(g)-average_value)**2 for g in valid_groups)
        ss_within=sum(sum((x-np.mean(g))**2 for x in g) for g in valid_groups)
        df_between=num_groups-1
        df_within=total_values-num_groups

        ms_between=ss_between/df_between if df_between>0 else 0
        ms_within=ss_within/df_within if df_within>0 else 0

        repeatability=calc_repeatability_from_ms(ms_within)
        num_measurements_per_day=np.mean([len(g) for g in valid_groups])
        intermediate_precision=calc_intermediate_precision(ms_within, ms_between, num_measurements_per_day)

        rel_repeat=repeatability/average_value if average_value!=0 else 0
        rel_intermediate=intermediate_precision/average_value if average_value!=0 else 0
        rel_extra=np.sqrt(sum([r[2]**2 for r in extra_unc]))
        combined_rel=np.sqrt(rel_repeat**2+rel_intermediate**2+rel_extra**2)
        expanded=2*combined_rel*average_value
        rel_expanded=calc_relative_expanded_uncertainty(expanded,average_value)

        results_list=[
            ("Repeatability",f"{repeatability:.4f}",r"s = sqrt(sum((x_i - x̄)^2)/(n-1))"),
            ("Intermediate Precision",f"{intermediate_precision:.4f}",r"s_IP = sqrt((MS_between-MS_within)/n)"),
            ("Combined Relative Uncertainty",f"{combined_rel:.4f}",r"u_c = sqrt(u_repeat^2 + u_IP^2 + u_extra^2)"),
            ("Relative Repeatability",f"{rel_repeat:.4f}",r"u_repeat,rel = s / x̄"),
            ("Relative Intermediate Precision",f"{rel_intermediate:.4f}",r"u_IP,rel = s_IP / x̄"),
            ("Relative Extra Uncertainty",f"{rel_extra:.4f}",r"u_extra,rel = sqrt(sum u_extra,i^2)"),
            (lang_texts["average_value"],f"{average_value:.4f}",r"x̄ = sum(x_i)/n"),
            (lang_texts["expanded_uncertainty"],f"{expanded:.4f}",r"U = 2 * u_c * x̄"),
            (lang_texts["relative_expanded_uncertainty_col"],f"{rel_expanded:.4f}",r"U_rel = U / x̄ * 100")
        ]
        display_results_with_formulas(results_list, lang_texts["results"], lang_texts)
        plot_daily_measurements(measurements, lang_texts)
        pdf_buf=create_pdf(results_list, lang_texts)
        st.download_button(lang_texts["results"], data=pdf_buf, file_name="uncertainty_results.pdf", mime="application/pdf")

# ------------------------
# Ana Fonksiyon
# ------------------------
def main():
    language=st.selectbox("Dil / Language", ["Türkçe"])
    lang_texts=languages[language]
    run_paste_mode(lang_texts)

if __name__=="__main__":
    main()
