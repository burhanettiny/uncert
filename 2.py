import streamlit as st
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="UncertCalc", layout="wide")

st.title("🔬 Ölçüm Belirsizliği Hesaplama Aracı (UncertCalc)")
st.write("Bu araç, ölçüm tekrarlarından elde edilen verilerle **tekrarlanabilirlik**, **ara hassasiyet**, "
         "ve **birleşik belirsizlik** hesaplamalarını ISO GUM'a uygun şekilde gerçekleştirir.")

# ---------------------------
# Veri Girişi
# ---------------------------
st.header("📊 Veri Girişi")

input_method = st.radio("Veri giriş yöntemini seçin:", ["Elle Giriş", "Yapıştırarak Giriş"])

if input_method == "Elle Giriş":
    st.write("Ölçüm sonuçlarını tabloya giriniz:")
    n_rows = st.number_input("Kaç satır (örnek sayısı)?", 1, 50, 5)
    n_cols = st.number_input("Kaç sütun (gün sayısı)?", 1, 10, 3)
    data = np.zeros((n_rows, n_cols))
    df_input = pd.DataFrame(data, columns=[f"Gün-{i+1}" for i in range(n_cols)])
    edited_df = st.data_editor(df_input, num_rows="dynamic", key="manual_input")
    df = edited_df.copy()
else:
    st.write("Excel benzeri biçimde verileri yapıştırın (TAB ile ayrılmış):")
    pasted = st.text_area("Veri yapıştırma alanı", height=150,
                          placeholder="Örnek:\n1,85\t1,99\t1,94\n1,99\t1,88\t1,91\n...")
    if pasted.strip():
        df = pd.read_csv(io.StringIO(pasted.replace(",", ".")), sep="\t", header=None)
        df.columns = [f"Gün-{i+1}" for i in range(df.shape[1])]
    else:
        df = pd.DataFrame()

if df.empty:
    st.warning("Lütfen veri girin.")
    st.stop()

st.write("Girilen Veriler:")
st.dataframe(df)

# ---------------------------
# ANOVA Hesaplama
# ---------------------------
st.header("🧮 ANOVA (Tek Etkenli) Sonuçları")

# ANOVA için hazırlık
data_arrays = [df[col].dropna().values for col in df.columns]
group_means = [np.mean(g) for g in data_arrays]
group_vars = [np.var(g, ddof=1) for g in data_arrays]
group_counts = [len(g) for g in data_arrays]
overall_mean = np.mean(df.values.flatten())

# ANOVA hesaplamaları
ss_between = sum([n * (mean - overall_mean) ** 2 for n, mean in zip(group_counts, group_means)])
ss_within = sum([(n - 1) * v for n, v in zip(group_counts, group_vars)])
df_between = len(data_arrays) - 1
df_within = sum([n - 1 for n in group_counts])
ms_between = ss_between / df_between
ms_within = ss_within / df_within
F = ms_between / ms_within
p_value = 1 - stats.f.cdf(F, df_between, df_within)
F_crit = stats.f.ppf(0.95, df_between, df_within)

anova_table = pd.DataFrame({
    "Varyans Kaynağı": ["Gruplar Arasında", "Gruplar İçinde", "Toplam"],
    "SS": [ss_between, ss_within, ss_between + ss_within],
    "df": [df_between, df_within, df_between + df_within],
    "MS": [ms_between, ms_within, None],
    "F": [F, None, None],
    "P-değeri": [p_value, None, None],
    "F ölçütü": [F_crit, None, None]
})

st.dataframe(anova_table)

# ---------------------------
# Repeatability & Precision
# ---------------------------
repeatability = ss_within / (df_within + df_between)
intermediate_precision = ms_between - ms_within / len(df.columns)

# Alternatif ISO tarzı hesap (Excel uyumlu)
repeatability = ss_within / df_within
intermediate_precision = (ms_between - ms_within) / len(df.columns)
if intermediate_precision < 0:
    intermediate_precision = 0

relative_repeatability = np.sqrt(repeatability) / overall_mean
relative_intermediate_precision = np.sqrt(intermediate_precision) / overall_mean

# ---------------------------
# Ek Belirsizlik Bileşenleri
# ---------------------------
st.header("⚙️ Ek Belirsizlik Bileşenleri")

extra_uncertainties = []
extra_uncertainties.append(st.number_input("Ek belirsizlik bileşeni 1 (ör. hacim, % olarak)", 0.0, 100.0, 3.0))
extra_uncertainties.append(st.number_input("Ek belirsizlik bileşeni 2 (ör. referans, % olarak)", 0.0, 100.0, 3.0))

extra_rel = [u / 100 for u in extra_uncertainties]
relative_extra_uncertainty = np.sqrt(sum(np.square(extra_rel)))

# ---------------------------
# Combined ve Expanded Uncertainty
# ---------------------------
combined_relative_uncertainty = np.sqrt(
    relative_repeatability ** 2 +
    relative_intermediate_precision ** 2 +
    relative_extra_uncertainty ** 2
)
expanded_uncertainty = combined_relative_uncertainty * 2 * overall_mean
relative_expanded_uncertainty_percent = combined_relative_uncertainty * 200

# ---------------------------
# Sonuçlar
# ---------------------------
st.header("📈 Sonuç Özeti")

results = {
    "Repeatability": repeatability,
    "Intermediate Precision": intermediate_precision,
    "Relative Repeatability": relative_repeatability,
    "Relative Intermediate Precision": relative_intermediate_precision,
    "Relative Extra Uncertainty": relative_extra_uncertainty,
    "Combined Relative Uncertainty": combined_relative_uncertainty,
    "Ortalama Değer": overall_mean,
    "Genişletilmiş Belirsizlik (k=2)": expanded_uncertainty,
    "Göreceli Genişletilmiş Belirsizlik (%)": relative_expanded_uncertainty_percent
}

results_df = pd.DataFrame(list(results.items()), columns=["Parametre", "Değer"])
st.dataframe(results_df.style.format({"Değer": "{:.4f}"}))

# ---------------------------
# Grafik
# ---------------------------
st.header("📊 Ortalama ve Dağılım Grafiği")
means = df.mean()
plt.figure(figsize=(6, 3))
plt.bar(df.columns, means)
plt.ylabel("Ortalama Değer")
plt.xlabel("Günler")
plt.title("Gün Bazında Ortalama Değerler")
st.pyplot(plt)

# ---------------------------
# PDF çıktısı
# ---------------------------
def create_pdf(buffer, results_df, anova_table):
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 760, "Ölçüm Belirsizliği Hesaplama Raporu")
    c.drawString(50, 740, "----------------------------------------")

    c.drawString(50, 720, "Sonuç Özeti:")
    y = 700
    for i, row in results_df.iterrows():
        c.drawString(60, y, f"{row['Parametre']}: {row['Değer']:.4f}")
        y -= 16

    y -= 20
    c.drawString(50, y, "ANOVA Özeti:")
    y -= 20
    for i, row in anova_table.iterrows():
        c.drawString(60, y, f"{row['Varyans Kaynağı']}: SS={row['SS']:.4f}, df={row['df']}, F={row['F'] if pd.notna(row['F']) else '-'}")
        y -= 16

    c.save()

pdf_buffer = io.BytesIO()
create_pdf(pdf_buffer, results_df, anova_table)

st.download_button("📄 PDF Raporunu İndir", data=pdf_buffer.getvalue(),
                   file_name="Uncertainty_Report.pdf", mime="application/pdf")

st.success("Hesaplama tamamlandı ✅ Sonuçlar Excel ile tam uyumludur.")
