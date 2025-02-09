import numpy as np
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

def calculate_repeatability(ms_within):
    return np.sqrt(ms_within) if ms_within >= 0 else float('nan')

def calculate_intermediate_precision_grouped(measurements):
    # Eğer grup içindeki MS değeri, gruplararası MS değerinden büyükse
    group_stdevs = [np.std(group, ddof=1) for group in measurements]
    group_sizes = [len(group) for group in measurements]
    
    numerator = sum(stdev**2 * (size - 1) for stdev, size in zip(group_stdevs, group_sizes))
    denominator = sum(size - 1 for size in group_sizes)
    
    if denominator > 0:
        return np.sqrt(numerator / denominator)
    return float('nan')

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')

def main():
    language = st.selectbox("Dil / Language", ["Türkçe", "English"])
    texts = {
        "Türkçe": {
            "title": "Belirsizlik Hesaplama Uygulaması",
            "subtitle": "B. Yalçınkaya tarafından geliştirildi",
            "upload": "Excel dosyanızı yükleyin",
            "paste": "Verileri buraya yapıştırın",
            "extra_uncertainty": "Ek Belirsizlik Bütçesi",
            "results": "Sonuçlar",
            "error_bar": "Hata Bar Grafiği",
            "daily_measurements": "Günlük Ölçüm Sonuçları"
        },
        "English": {
            "title": "Uncertainty Calculation Application",
            "subtitle": "Developed by B. Yalçınkaya",
            "upload": "Upload your Excel file",
            "paste": "Paste data here",
            "extra_uncertainty": "Extra Uncertainty Budget",
            "results": "Results",
            "error_bar": "Error Bar Graph",
            "daily_measurements": "Daily Measurement Results"
        }
    }
    
    st.title(texts[language]["title"])
    st.caption(texts[language]["subtitle"])
    
    uploaded_file = st.file_uploader(texts[language]["upload"], type=["xlsx", "xls"])
    pasted_data = st.text_area(texts[language]["paste"])
    
    # Önce Ek Belirsizlik Bütçesi Etiketi girilsin:
    custom_extra_uncertainty_label = st.text_input("Ek Belirsizlik Bütçesi Etiketi", value="Ek Belirsizlik Bütçesi")
    # Bu etiket başlığı ile Ek Belirsizlik Bütçesi değeri girilsin:
    extra_uncertainty = st.number_input(custom_extra_uncertainty_label, min_value=0.0, value=0.0, step=0.01)
    
    measurements = []
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
    elif pasted_data:
        try:
            # Replace commas with periods for decimal values
            pasted_data = pasted_data.replace(',', '.')
            
            # Try reading the data as space-separated values
            df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
        except Exception as e:
            st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
            st.stop()  # Use st.stop() to stop execution if there's an error
    else:
        st.stop()  # Stop if neither a file nor pasted data is provided
    
    # Veri çerçevesinin kolon sayısını kontrol et
    num_columns = df.shape[1]
    columns = [f"{i+1}. Gün" for i in range(num_columns)]
    
    # Kolon isimlerini dinamik olarak ayarla
    df.columns = columns
    df.index = [f"{i+1}. Ölçüm" for i in range(len(df))]
    measurements = df.T.values.tolist()
    num_measurements_per_day = len(df)
    
    st.write("Yapıştırılan Veri:")
    st.dataframe(df, use_container_width=True)
    
    if len(measurements) > 1:
        total_values = sum(len(m) for m in measurements)
        num_groups = len(measurements)
        average_value = np.mean([val for group in measurements for val in group])
        
        ss_between = sum(len(m) * (np.mean(m) - average_value) ** 2 for m in measurements)
        ss_within = sum(sum((x - np.mean(m)) ** 2 for x in m) for m in measurements)
        
        df_between = num_groups - 1
        df_within = total_values - num_groups
        
        ms_between = ss_between / df_between if df_between > 0 else float('nan')
        ms_within = ss_within / df_within if df_within > 0 else float('nan')
        
        # Orijinal parametreler:
        repeatability = calculate_repeatability(ms_within)
        
        # Intermediate Precision hesaplama:
        intermediate_precision = ms_within
        uyarı_mesajı = ""
        
        if ms_between > ms_within:
            intermediate_precision = calculate_intermediate_precision_grouped(measurements)
            uyarı_mesajı = "Grup için MS değeri, Gruplararası MS değerinden büyük olduğundan Intermediate Precision değeri 'urepro' hesaplanarak belirlenmiştir."
        
        # Relative değerler:
        relative_repeatability = repeatability / average_value if average_value != 0 else float('nan')
        relative_intermediate_precision = intermediate_precision / average_value if average_value != 0 else float('nan')
        relative_extra_uncertainty = extra_uncertainty / 100  # Ek Belirsizlik Bütçesi değeri 100'e bölünür.
        
        # Combined Relative Uncertainty:
        combined_relative_uncertainty = np.sqrt(
            relative_repeatability**2 +
            relative_intermediate_precision**2 +
            relative_extra_uncertainty**2
        )
        
        # Expanded Uncertainty (k=2):
        expanded_uncertainty = 2 * combined_relative_uncertainty * average_value
        
        # Relative Expanded Uncertainty (%):
        relative_expanded_uncertainty = calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value)
        
        # Sonuçlar tablosu (tüm değerler noktadan sonra 4 basamak):
        results_df = pd.DataFrame({
            "Parametre": [
                "Tekrarlanabilirlik",
                "Intermediate Precision",
                custom_extra_uncertainty_label,
                "Combined Relative Uncertainty",
                "Relative Repeatability",
                "Relative Intermediate Precision",
                "Relative Ek Belirsizlik"
            ],
            "Değer": [
                f"{repeatability:.4f}",
                f"{intermediate_precision:.4f}",
                f"{extra_uncertainty:.4f}",
                f"{combined_relative_uncertainty:.4f}",
                f"{relative_repeatability:.4f}",
                f"{relative_intermediate_precision:.4f}",
                f"{relative_extra_uncertainty:.4f}"
            ],
            "Formül": [
                "√(MS_within)",
                "√((MS_between - MS_within) / N)",
                f"({custom_extra_uncertainty_label} değeri)",
                "√((Relative Repeatability)² + (Relative Intermediate Precision)² + (Relative Ek Belirsizlik)²)",
                "(Repeatability / Mean)",
                "(Intermediate Precision / Mean)",
                f"({custom_extra_uncertainty_label} / 100)"
            ]
        })
        
        additional_row = pd.DataFrame({
            "Parametre": ["Ortalama Değer", "Expanded Uncertainty (k=2)", "Relative Expanded Uncertainty (%)"],
            "Değer": [
                f"{average_value:.4f}",
                f"{expanded_uncertainty:.4f}",
                f"{relative_expanded_uncertainty:.4f}"
            ],
            "Formül": [
                "mean(X)",
                "Combined Relative Uncertainty × Mean × 2",
                "(Expanded Uncertainty / Mean) × 100"
            ]
        })
        
        results_df = pd.concat([results_df, additional_row], ignore_index=True)
        
        # Intermediate Precision için "*" ekleyin
        if uyarı_mesajı:
            results_df.loc[results_df['Parametre'] == 'Intermediate Precision', 'Değer'] += ' *'
            st.write(uyarı_mesajı)
        
        st.write("Sonuçlar Veri Çerçevesi:")
        st.dataframe(results_df)
        
        # Hata Bar Grafiği:
        fig, ax = plt.subplots()
        x_labels = [f"{i+1}. Gün" for i in range(num_columns)] + ["Ortalama"]
        x_values = [np.mean(day) for day in measurements] + [average_value]
        y_errors = [np.std(day, ddof=1) for day in measurements] + [0]
        ax.errorbar(x_labels, x_values, yerr=y_errors, fmt='o', capsize=5, ecolor='red', linestyle='None')
        ax.set_ylabel("Değer")
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_title(texts[language]["error_bar"])
        st.pyplot(fig)
        
        # Günlük Ölçüm Grafiği:
        fig, ax = plt.subplots()
        for i, group in enumerate(measurements):
            ax.plot(range(1, len(group) + 1), group, marker='o', linestyle='-', label=f"Gün {i+1}")
        ax.set_xlabel("Ölçüm Sayısı")
        ax.set_ylabel("Değer")
        ax.set_title(texts[language]["daily_measurements"])
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
