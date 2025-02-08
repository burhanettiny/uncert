# Veri çerçevesi oluşturulurken sütun adlarını temizleyelim
results_df = pd.DataFrame({
    "Parametre": ["Ortalama Değer", "Tekrarlanabilirlik", "Intermediate Precision", "Combined Relative Uncertainty", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
    "Değer": [f"{average_value:.1f}", f"{repeatability:.1f}", f"{intermediate_precision:.1f}", f"{combined_uncertainty:.1f}", f"{expanded_uncertainty:.1f}", f"{relative_expanded_uncertainty:.1f}"],
    "Formül": ["mean(X)", "√(MS_within)", "√(MS_between - MS_within)", "√(Repeatability² + Intermediate Precision² + Extra Uncertainty²)", "Combined Uncertainty × 2", "(Expanded Uncertainty / Mean) × 100"]
})

# Sütun isimlerini temizleyelim
results_df.columns = results_df.columns.str.strip()

# Sütunları ekrana yazdıralım
st.write("Sonuçlar Veri Çerçevesi (Sütun Kontrolü):", results_df.columns.tolist())

# Veri çerçevesini doğrudan yazdır
st.dataframe(results_df)


