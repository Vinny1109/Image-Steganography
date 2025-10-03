#------ External Libraries ------#
import cv2
import struct
import bitstring
import numpy  as np
import dct_zigzag as zz
import os
import csv
#================================#
#---------- Source Files --------#
import dct_image_preparation as img
import dct_data_embedding as stego
#================================#

NUM_CHANNELS = 3
FOLDER_PATH = "./ori/low"
OUTPUT_FOLDER = "./dct/low"
OUTPUT_CSV = "./dct/dct_stego_results_low.csv"

# Pastikan folder output ada
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Mendapatkan semua file PNG di folder, urut nama
image_files = sorted([f for f in os.listdir(FOLDER_PATH) if f.lower().endswith('.png')])

count = 0
with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["filename", "original_size", "stego_size", "resolution", "embedded_message"])
    for image_file in image_files:
        try:
            COVER_IMAGE_FILEPATH = os.path.join(FOLDER_PATH, image_file)
            filename, ext = os.path.splitext(image_file)
            STEGO_IMAGE_FILEPATH = os.path.join(OUTPUT_FOLDER, f"{filename}_stego{ext}")

            print(f"Processing: {COVER_IMAGE_FILEPATH}")

            raw_cover_image = cv2.imread(COVER_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
            if raw_cover_image is None:
                writer.writerow([image_file, "[FAILED TO READ IMAGE]", "", "", ""])
                print(f"{image_file}: [FAILED TO READ IMAGE]")
                continue

            height, width = raw_cover_image.shape[:2]
            ori_size = os.path.getsize(COVER_IMAGE_FILEPATH)
            # Force Image Dimensions to be 8x8 compliant
            pad_height, pad_width = height, width
            while(pad_height % 8): pad_height += 1
            while(pad_width  % 8): pad_width  += 1
            valid_dim = (pad_width, pad_height)
            padded_image    = cv2.resize(raw_cover_image, valid_dim)
            cover_image_f32 = np.float32(padded_image)
            cover_image_YCC = img.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))

            # Placeholder for holding stego image data
            stego_image = np.empty_like(cover_image_f32)
            embedded_message = ""
            for chan_index in range(NUM_CHANNELS):
                # FORWARD DCT STAGE
                dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

                # QUANTIZATION STAGE
                dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

                # Sort DCT coefficients by frequency
                sorted_coefficients = [zz.zigzag(block) for block in dct_quants]

                array_coefficients = np.array(sorted_coefficients)
                valid_coefficients = array_coefficients[array_coefficients != 0]
                print(f"Valid DCT coefficients available: {len(valid_coefficients)}")

                max_capacity_bits = len(valid_coefficients)
                max_capacity_bytes = max_capacity_bits // 8
                max_capacity_chars = max_capacity_bytes

                print(f"Maksimum kapasitas penyisipan: {max_capacity_bits} bits ({max_capacity_bytes} bytes, {max_capacity_chars} karakter)")

                if (chan_index == 0):
                    SECRET_MESSAGE_STRING = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur dictum justo eget est maximus, in mollis massa porttitor. Mauris gravida scelerisque orci id eleifend. Sed fermentum orci a velit eleifend laoreet. Cras semper sed nibh eget vehicula. Donec sed eros arcu. Aenean tempor, felis ac dictum tincidunt, odio nulla consectetur dolor, nec molestie lorem felis et sem. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Fusce aliquet est venenatis euismod bibendum. Ut ultricies a nulla quis sollicitudin. Nam sagittis venenatis ac."
                    # Potong pesan jika lebih panjang dari kapasitas
                    max_chars = min(len(SECRET_MESSAGE_STRING), max_capacity_chars)
                    embedded_message = SECRET_MESSAGE_STRING[:max_chars]
                    secret_data = ""
                    for char in embedded_message.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                    print("hasil encode: ",secret_data)
                    print(f"test2 Valid DCT Coefficients Available: {len(sorted_coefficients)}")

                    embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                    desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
                else:
                    desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]
                print(f"test2 desorted DCT Coefficients Available: {len(desorted_coefficients)}")

                dct_dequants = [np.multiply(data, img.JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]
                idct_blocks = [cv2.idct(block) for block in dct_dequants]
                stego_image[:,:,chan_index] = np.asarray(img.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))

            stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)
            final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))
            cv2.imwrite(STEGO_IMAGE_FILEPATH, final_stego_image)
            stego_size = os.path.getsize(STEGO_IMAGE_FILEPATH)
            # Catat ke csv
            writer.writerow([image_file, ori_size, stego_size, f"{width}x{height}", embedded_message])
            print(f"Saved stego image: {STEGO_IMAGE_FILEPATH}\n")
            count += 1
            print(f"Processed file: {count}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            writer.writerow([image_file, f"[ERROR: {e}]", "", "", ""])
