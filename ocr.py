import fitz
import pytesseract
from PIL import Image
import io
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PDF_FILE = "cv.pdf"

TEXT_DIR = "extracted_texts"
IMAGE_DIR = "extracted_images"
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

base_name = os.path.splitext(os.path.basename(PDF_FILE))[0]
text_output_path = os.path.join(TEXT_DIR, f"{base_name}.txt")
image_output_dir = os.path.join(IMAGE_DIR, base_name)
os.makedirs(image_output_dir, exist_ok=True)

full_text = ""

with fitz.open(PDF_FILE) as pdf:
    for i, page in enumerate(pdf):
        full_text += f"\n--- Page {i + 1} ---\n"

        # Extract detailed text info
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        font = span["font"]
                        text = span["text"].strip()
                        if not text:
                            continue
                        if "Bold" in font or span["size"] > 12:  # Threshold for bold/heading
                            line_text += f"\n**{text}**\n"
                        else:
                            line_text += f"{text} "
                    full_text += line_text.strip() + "\n"

        # Extract images
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            img_filename = os.path.join(image_output_dir, f"page_{i + 1}_image_{img_index + 1}.{img_ext}")

            with open(img_filename, "wb") as img_file:
                img_file.write(image_bytes)

            img = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(img, lang='eng')
            full_text += f"\n--- OCR Image {img_index + 1} ---\n{ocr_text.strip()}\n"

with open(text_output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("✅ Processing complete!")
print(f"Text saved to: {text_output_path}")
print(f"Images saved in: {image_output_dir}")
