'''
Author: Mason Edgar
ECE 529 - Algorithm Project
Image Steganography
'''
#------ External Libraries ------#
import os
import cv2
import struct
import bitstring
import numpy  as np
import dct_zigzag as zz
import dct_data_embedding as stego
import dct_image_preparation   as img
import csv

# Folder berisi file stego PNG
STEGO_FOLDER = "./dct/low"
OUTPUT_CSV = "./dct/dct_extracted_results_low.csv"

# Pastikan folder output ada
output_dir = os.path.dirname(OUTPUT_CSV)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ambil semua file PNG yang ada _steg pada namanya, urut nama
stego_files = sorted([f for f in os.listdir(STEGO_FOLDER) if f.lower().endswith('.png') and '_steg' in f])

EXPECTED_MESSAGE = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur dictum justo eget est maximus, in mollis massa porttitor. Mauris gravida scelerisque orci id eleifend. Sed fermentum orci a velit eleifend laoreet. Cras semper sed nibh eget vehicula. Donec sed eros arcu. Aenean tempor, felis ac dictum tincidunt, odio nulla consectetur dolor, nec molestie lorem felis et sem. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Fusce aliquet est venenatis euismod bibendum. Ut ultricies a nulla quis sollicitudin. Nam sagittis venenatis ac."

count = 0
with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["filename", "original size", "stego size", "resolution", "extracted", "expected"])
    for stego_file in stego_files:
        try:
            stego_path = os.path.join(STEGO_FOLDER, stego_file)
            print(f"Processing: {stego_path}")

            stego_image = cv2.imread(stego_path)
            if stego_image is None:
                writer.writerow([stego_file, "[FAILED TO READ IMAGE]", "", "", "", ""])
                print(f"{stego_file}: [FAILED TO READ IMAGE]")
                continue

            stego_size = os.path.getsize(stego_path)
            # Cari file asli (tanpa _steg)
            original_file = stego_file.replace('_steg', '')
            original_path = os.path.join(os.path.dirname(STEGO_FOLDER), original_file)
            if os.path.exists(original_path):
                ori_size = os.path.getsize(original_path)
            else:
                ori_size = "[ORIGINAL NOT FOUND]"

            height, width = stego_image.shape[:2]
            stego_image_f32 = np.float32(stego_image)
            stego_image_YCC = img.YCC_Image(cv2.cvtColor(stego_image_f32, cv2.COLOR_BGR2YCrCb))

            # FORWARD DCT STAGE
            dct_blocks = [cv2.dct(block) for block in stego_image_YCC.channels[0]]  # Only care about Luminance layer

            # QUANTIZATION STAGE
            dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

            # Sort DCT coefficients by frequency
            sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

            # DATA EXTRACTION STAGE
            recovered_data = stego.extract_encoded_data_from_DCT(sorted_coefficients)
            recovered_data.pos = 0

            try:
                data_len = int(recovered_data.read('uint:32') / 8)
                extracted_data = bytes()
                for _ in range(data_len): 
                    if recovered_data.len - recovered_data.pos >= 8:
                        extracted_data += struct.pack('>B', recovered_data.read('uint:8'))
                    else:
                        break
                secret_message = extracted_data.decode('utf-8', errors='replace')
            except Exception as e:
                secret_message = f"[EXTRACTION ERROR: {e}]"

            writer.writerow([stego_file, ori_size, stego_size, f"{width}x{height}", secret_message, EXPECTED_MESSAGE])
            print(f"Extracted from {stego_file}")
            count+=1
            print(f"Proses file ke {count}")

        except Exception as e:
            writer.writerow([stego_file, "ERROR", "", "", "", str(e).replace(',', ';')])
            print(f"Error processing {stego_file}: {e}")

print(f"All results saved to {OUTPUT_CSV}")
