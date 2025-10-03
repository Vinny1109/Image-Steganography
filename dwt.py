import numpy as np
import cv2
import pywt
import os
import pandas as pd
import re

def text_to_binary(text):
    """Convert text to binary string with 8 bits per character"""
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    """Convert binary string to text"""
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    try:
        return ''.join(chr(int(char, 2)) for char in chars if len(char) == 8)
    except ValueError:
        return "Error: Non-decodable binary sequence"

def embed_text_in_image(image, text):
    """Embed text into image using DWT on Cb channel"""
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cb, cr = cv2.split(ycbcr)

    h, w = cb.shape
    pad_h = h % 2
    pad_w = w % 2
    cb_padded = cv2.copyMakeBorder(cb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    coeffs = pywt.dwt2(cb_padded, 'haar')
    LL, (LH, HL, HH) = coeffs

    binary_text = text_to_binary(text)
    text_length = len(binary_text)
    header = format(text_length, '032b')
    full_data = header + binary_text

    HH_flat = HH.flatten()
    HL_flat = HL.flatten()
    
    available_bits = len(HH_flat) + len(HL_flat)
    required_bits = len(full_data)
    
    if required_bits > available_bits:
        raise ValueError(f"Insufficient capacity: Need {required_bits} bits, Available {available_bits} bits")

    data_index = 0
    for i in range(len(HH_flat)):
        if data_index < len(full_data):
            HH_flat[i] = (np.int16(HH_flat[i]) & ~1) | int(full_data[data_index])
            data_index += 1
    
    for i in range(len(HL_flat)):
        if data_index < len(full_data):
            HL_flat[i] = (np.int16(HL_flat[i]) & ~1) | int(full_data[data_index])
            data_index += 1

    HH_modified = HH_flat.reshape(HH.shape)
    HL_modified = HL_flat.reshape(HL.shape)
    
    cb_modified_padded = pywt.idwt2((LL, (LH, HL_modified, HH_modified)), 'haar')
    cb_modified = cb_modified_padded[:h, :w]

    return cv2.cvtColor(cv2.merge([y, cb_modified.astype('uint8'), cr]), cv2.COLOR_YCrCb2BGR)

def extract_text_from_image(image):
    """Extract text from stego image using DWT"""
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cb, _ = cv2.split(ycbcr)

    h, w = cb.shape
    pad_h = h % 2
    pad_w = w % 2
    cb_padded = cv2.copyMakeBorder(cb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    coeffs = pywt.dwt2(cb_padded, 'haar')
    _, (_, HL, HH) = coeffs

    HH_flat = HH.flatten().astype(np.int16)
    HL_flat = HL.flatten().astype(np.int16)

    extracted_bits_hh = (HH_flat & 1)
    extracted_bits_hl = (HL_flat & 1)
    
    binary_data = ''.join(map(str, np.concatenate([extracted_bits_hh, extracted_bits_hl])))
    
    if len(binary_data) < 32:
        return "Error: Not enough data to read header"
        
    header = binary_data[:32]
    try:
        text_length = int(header, 2)
    except ValueError:
        return "Error: Invalid header format"
    
    if text_length > len(binary_data) - 32:
        return "Error: Header indicates length larger than available data"

    text_binary = binary_data[32:32 + text_length]
    return binary_to_text(text_binary)

def embed_text_in_folder(folder_path, text, output_folder, csv_path):
    """Embeds text in all images in a folder and saves a summary CSV."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    results = []
    print("Embedding and verifying text... This may take a moment.")
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"Error reading {filename}, skipping.")
                continue
                
            try:
                original_size = os.path.getsize(image_path)
                h, w, _ = image.shape
                resolution = f"{w}x{h}"

                stego_image = embed_text_in_image(image, text)
                name, ext = os.path.splitext(filename)
                out_filename = f"{name}_stego{ext}"
                out_path = os.path.join(output_folder, out_filename)
                cv2.imwrite(out_path, stego_image)
                
                verified_text = extract_text_from_image(stego_image)
                sanitized_verified_text = sanitize_text(verified_text)
                stego_size = os.path.getsize(out_path)

                results.append({
                    'filename': filename,
                    'original size': original_size,
                    'stego size': stego_size,
                    'resolution': resolution,
                    'embedded_message': sanitized_verified_text
                })
                print(f"Embedded and verified text in {filename}, saved as {out_path}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nEmbedding summary saved to {csv_path}")

def sanitize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\x00-\x7F]', '', text)

def extract_texts_from_folder(folder_path, csv_path, expected_text):
    """Extracts texts from all images in a folder and saves results to a CSV."""
    results = []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Error reading {filename}, skipping.")
                results.append({'filename': filename, 'stego size': 'N/A', 'resolution': 'N/A', 'extracted': 'Error reading image', 'expected': expected_text})
                continue

            try:
                stego_size = os.path.getsize(image_path)
                h, w, _ = image.shape
                resolution = f"{w}x{h}"
                extracted = extract_text_from_image(image)
                sanitized = sanitize_text(extracted)
                results.append({'filename': filename, 'stego size': stego_size, 'resolution': resolution, 'extracted': sanitized, 'expected': expected_text})
            except Exception as e:
                results.append({'filename': filename, 'stego size': 'N/A', 'resolution': 'N/A', 'extracted': f"Error: {str(e)}", 'expected': expected_text})

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nExtraction complete. Results saved to {csv_path}")

# --- FUNGSI MAIN DIUBAH UNTUK MENYESUAIKAN OUTPUT SATU FILE ---
def main():
    print("\nDWT-Based Steganography System")
    print("-----------------------------")
    
    while True:
        choice = input("\nChoose an option:\n1. Embed text into a single image\n2. Extract text from a single image\n3. Embed text into a folder of images (and create CSV)\n4. Extract text from a folder of images (and create CSV)\n5. Exit\n> ")
        
        if choice == '1': # Embed untuk satu file
            try:
                image_path = input("Enter cover image path: ")
                text = input("Enter text to embed: ")
                output_path = input("Enter output path for stego image (e.g., stego.png): ")

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    print("Error: Could not read image file.")
                    continue
                
                # Mengumpulkan informasi seperti pada fungsi folder
                original_size = os.path.getsize(image_path)
                h, w, _ = image.shape
                resolution = f"{w}x{h}"

                stego_image = embed_text_in_image(image, text)
                cv2.imwrite(output_path, stego_image)

                stego_size = os.path.getsize(output_path)
                verified_text = sanitize_text(extract_text_from_image(stego_image))

                # Mencetak ringkasan langsung ke terminal
                print("\n--- Embed Summary ---")
                print(f"  Original Filename : {os.path.basename(image_path)}")
                print(f"  Original Size     : {original_size} bytes")
                print(f"  Stego Size        : {stego_size} bytes")
                print(f"  Resolution        : {resolution}")
                print(f"  Verified Message  : {verified_text}")
                print(f"  Stego image saved to '{output_path}'")
                print("---------------------")

            except Exception as e:
                print(f"An error occurred: {str(e)}")
        
        elif choice == '2': # Extract untuk satu file
            try:
                stego_path = input("Enter stego image path: ")
                expected_text = input("Enter the expected text for comparison: ")

                stego_image = cv2.imread(stego_path, cv2.IMREAD_COLOR)
                if stego_image is None:
                    print("Error: Could not read stego image.")
                    continue

                # Mengumpulkan informasi seperti pada fungsi folder
                stego_size = os.path.getsize(stego_path)
                h, w, _ = stego_image.shape
                resolution = f"{w}x{h}"
                extracted_text = sanitize_text(extract_text_from_image(stego_image))
                
                # Mencetak ringkasan langsung ke terminal
                print("\n--- Extract Summary ---")
                print(f"  Stego Filename    : {os.path.basename(stego_path)}")
                print(f"  Stego Size        : {stego_size} bytes")
                print(f"  Resolution        : {resolution}")
                print(f"  Extracted Text    : {extracted_text}")
                print(f"  Expected Text     : {expected_text}")
                if extracted_text == expected_text:
                    print("  Status              : ✅ Match")
                else:
                    print("  Status              : ❌ No Match")
                print("-----------------------")

            except Exception as e:
                print(f"An error occurred: {str(e)}")
        
        elif choice == '3': # Embed untuk folder (tetap menghasilkan CSV)
            folder_path = input("Enter folder path containing original images: ")
            text = input("Enter text to embed: ")
            output_folder = input("Enter output folder for stego images: ")
            csv_path = input("Enter output CSV file path for the summary (e.g., embed_summary.csv): ")
            embed_text_in_folder(folder_path, text, output_folder, csv_path)

        elif choice == '4': # Extract untuk folder (tetap menghasilkan CSV)
            folder_path = input("Enter folder path containing stego images: ")
            expected_text_input = input("Enter the expected text for comparison: ")
            csv_path = input("Enter output CSV file path for extraction results (e.g., extract_results.csv): ")
            extract_texts_from_folder(folder_path, csv_path, expected_text_input)

        elif choice == '5':
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()