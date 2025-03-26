import cv2
import numpy as np
import os

def resize_and_pad(image, size=256):
    """Resize and pad an image to a square while preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a black square image
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    square_img = np.zeros((size, size, 3), dtype=np.uint8)

    # Place the resized image in the center
    square_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return square_img

def normalize_image(image):
    """Normalize image pixel values to range [0,1]."""
    return image.astype(np.float32) / 255.0  # Convert to float32 and scale

def process_images(dataset_folder, output_folder, size=256):
    """Process images: Resize, pad, normalize, and save in separate folders."""
    categories = {"cat": "cat_processed", "banana": "banana_processed"}

    for category, processed_folder in categories.items():
        input_folder = os.path.join(dataset_folder, category)
        output_category_folder = os.path.join(output_folder, processed_folder)

        if not os.path.exists(output_category_folder):
            os.makedirs(output_category_folder)
            print(f"📂 Created directory: {output_category_folder}")

        if not os.path.exists(input_folder):
            print(f"⚠ Skipping {category}: Folder does not exist → {input_folder}")
            continue

        print(f"🔄 Processing {category} images...")

        count_processed, count_skipped = 0, 0

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(input_folder, filename)
                image = cv2.imread(img_path)

                if image is None:
                    print(f"⚠ Skipping {filename} (Could not read image)")
                    count_skipped += 1
                    continue

                processed_img = resize_and_pad(image, size)
                normalized_img = normalize_image(processed_img)

                # Save processed image
                output_path = os.path.join(output_category_folder, filename)
                final_image = (normalized_img * 255).astype(np.uint8)
                cv2.imwrite(output_path, final_image)

                print(f"✅ Saved {filename} → {output_path}")
                count_processed += 1

        print(f"\n✅ {category.capitalize()} Processing Complete!")
        print(f"✔ {count_processed} images processed.")
        print(f"❌ {count_skipped} images skipped.\n")

# Example usage
dataset_folder = r"C:\Users\GaddoStaffu\Desktop\IMAGE DATASET\dataset"
output_folder = r"C:\Users\GaddoStaffu\Desktop\IMAGE DATASET\processed"

process_images(dataset_folder, output_folder)
