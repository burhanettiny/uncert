elif pasted_data:
    try:
        # Replace commas with periods for decimal values
        pasted_data = pasted_data.replace(',', '.')
        
        # Try reading the data as space-separated values
        df = pd.read_csv(io.StringIO(pasted_data), sep="\s+", header=None, engine='python')
    except Exception as e:
        st.error(f"Hata! Lütfen verileri doğru formatta yapıştırın. ({str(e)})")
        return
