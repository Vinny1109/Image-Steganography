import pandas as pd
import os
from difflib import SequenceMatcher

# --- Configuration ---
# Sesuaikan path ini agar cocok dengan struktur folder Anda

# Path ke folder dengan gambar asli yang belum dimodifikasi
ORIGINAL_IMAGES_FOLDER = "./ori/low"

# Path ke folder dengan gambar stego dan CSV awal
STEGO_FOLDER = "./dwt/low/"

# Path ke file CSV input
INPUT_CSV_PATH = "./dwt/dwt_extracted_results_low.csv"

# Path tempat CSV akhir yang terperinci akan disimpan
OUTPUT_CSV_PATH = "./dwt/dwt_low_analysis.csv"

# --- Helper Function for Bit Error Rate (BER) ---

def calculate_ber_percentage(expected_str, extracted_str):
    """Menghitung Bit Error Rate (BER) antara dua string sebagai persentase."""
    expected_str = str(expected_str)
    extracted_str = str(extracted_str)
    try:
        expected_bits = ''.join(format(byte, '08b') for byte in bytearray(expected_str, 'utf-8', errors='ignore'))
        extracted_bits = ''.join(format(byte, '08b') for byte in bytearray(extracted_str, 'utf-8', errors='ignore'))
    except TypeError:
        return 100.0 if expected_str != extracted_str else 0.0
    
    max_len = max(len(expected_bits), len(extracted_bits))
    expected_bits = expected_bits.ljust(max_len, '0')
    extracted_bits = extracted_bits.ljust(max_len, '0')
    
    if max_len == 0:
        return 0.0
        
    error_bits = sum(1 for e, r in zip(expected_bits, extracted_bits) if e != r)
    ber = (error_bits / max_len) * 100
    return ber

# --- Helper Function for Text Match Analysis ---

def analyze_text_match(expected, extracted):
    """Membandingkan dan memberikan analisis tentang kesamaan dua string."""
    expected = str(expected)
    extracted = str(extracted)
    
    if expected == extracted:
        return "Match"
    
    similarity_ratio = SequenceMatcher(None, expected, extracted).ratio()
    
    if similarity_ratio > 0.9:
        return "Slight Difference"
    
    # Periksa apakah awal pesan yang diekstraksi cocok dengan yang diharapkan
    if extracted.startswith(expected[:50]):
        return "End Data Corruption"
        
    return "Significant Data Corruption"

# --- Main Analysis Script ---

def main():
    """Fungsi utama untuk melakukan analisis."""
    print("Memulai analisis...")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: File input tidak ditemukan di '{INPUT_CSV_PATH}'")
        return

    df = pd.read_csv(INPUT_CSV_PATH)
    
    new_data = []

    for index, row in df.iterrows():
        stego_filename = row['filename']
        original_filename = stego_filename.replace('_stego', '')
        original_path = os.path.join(ORIGINAL_IMAGES_FOLDER, original_filename)
        stego_path = os.path.join(STEGO_FOLDER, stego_filename)
        
        # Inisialisasi data baris tanpa metrik gambar
        row_data = {
            'filename': stego_filename,
            'original size': None,
            'stego size': None,
            'size percentage increase': None,
        }

        if os.path.exists(original_path) and os.path.exists(stego_path):
            try:
                # Hanya hitung ukuran file, tidak perlu membuka gambar
                original_size = os.path.getsize(original_path)
                stego_size = os.path.getsize(stego_path)
                row_data['original size'] = original_size
                row_data['stego size'] = stego_size

                if original_size > 0:
                    increase = ((stego_size - original_size) / original_size) * 100
                    row_data['size percentage increase'] = increase
                else:
                    row_data['size percentage increase'] = float('inf')

            except Exception as e:
                print(f"Tidak dapat memproses {stego_filename}: {e}")
        else:
            print(f"Peringatan: File hilang untuk {stego_filename}. Asli: '{original_path}', Stego: '{stego_path}'")
        
        # --- Tambahkan data dari CSV asli ---
        expected_message = row['expected']
        extracted_message = row['extracted']
        row_data['extracted'] = extracted_message
        row_data['expected'] = expected_message

        # --- Lakukan analisis baru dan perhitungan BER ---
        row_data['analysis'] = analyze_text_match(expected_message, extracted_message)
        row_data['BER %'] = calculate_ber_percentage(expected_message, extracted_message)

        new_data.append(row_data)

    final_df = pd.DataFrame(new_data)
    
    # Tentukan urutan kolom akhir, tanpa metrik gambar
    column_order = [
        'filename', 
        'original size', 
        'stego size', 
        'size percentage increase',
        'extracted', 
        'expected', 
        'analysis',
        'BER %'
    ]
    final_df = final_df[column_order]

    final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')
    print(f"\nAnalisis selesai! Hasil telah disimpan ke '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()