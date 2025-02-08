def calculate_intermediate_precision(ms_within, ms_between, num_measurements_per_day):
    # MS_between > MS_within kontrolü ile hata engelleniyor
    if ms_between > ms_within:
        return np.sqrt((ms_between - ms_within) / num_measurements_per_day)
    return float('nan')

def calculate_combined_uncertainty(repeatability, intermediate_precision, extra_uncertainty):
    # Doğru formül ile combined uncertainty hesaplanıyor
    return np.sqrt(repeatability**2 + intermediate_precision**2 + extra_uncertainty**2)

def calculate_relative_expanded_uncertainty(expanded_uncertainty, average_value):
    # Expanded uncertainty hesaplanıyor ve relatif genişletilmiş belirsizlik hesaplanıyor
    return (expanded_uncertainty / average_value) * 100 if average_value != 0 else float('nan')
