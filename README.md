
---

# 🧠 AI-Powered CV to Job Matcher

This project is an AI-based web scraping and job matching tool that leverages OCR, NLP, and LLMs to read a user's CV and suggest job opportunities that align with the candidate’s skills and experience.

---

## 🚀 Features

- 📝 **CV Parsing**: Converts PDF CVs into structured text using OCR and text block analysis.
- 🧠 **Semantic Matching**: Embeds CV content into a vector space for similarity-based job relevance matching using `Weaviate`.
- 🌐 **Web Scraping**: Scrapes job listings using `crawl4ai` and extracts structured data using LLM-based extraction.
- 📄 **Output**: Saves top matched jobs in a CSV for easy viewing and integration.

---

## 📁 Project Structure

```
├── ocr.py                # Converts PDF to structured text
├── main.py               # Embeds CV into Weaviate, runs scraper and LLM matching
├──  venue.py             # Contains scraping and job extraction logic
├── models/
│   └── venue.py          # Job schema model
├── utils/
│   ├── data_utils.py     # Helpers for saving data and filtering
│   └── scraper_utils.py  # Scraper-related utilities
├── extracted_texts/      # Output folder for OCR-processed text
├── extracted_images/     # Extracted images from the CV (if any)
├── complete_venues.csv   # Final matched job listings
```

---

## ⚙️ Technologies Used & Why

### 🔹 Python 🐍
Used for overall backend scripting due to its rich ecosystem in AI, OCR, and web scraping.

### 🔹 PyMuPDF (`fitz`) 📄
Used to extract both text and images from the uploaded PDF CV. It provides precise control over layout blocks, fonts, and text formatting.

### 🔹 Tesseract OCR (`pytesseract`) 🔍
Used to convert any images (scanned signatures, certificates, etc.) inside the CV to text. Helps enhance the completeness of the CV content before analysis.

### 🔹 LangChain 🦜🔗
Used to split the CV into logical chunks and interface with vector databases like Weaviate. It simplifies working with large documents and embeddings.

### 🔹 HuggingFace Transformers 🤗
Provides high-quality language embeddings that represent the semantic meaning of the CV content. These are then stored in Weaviate.

### 🔹 Weaviate (Vector Database) 🧬📊
Used to store the semantic vector representation of the CV chunks and perform similarity search against job prompts. Weaviate is fast, scalable, and integrates smoothly with LangChain.

### 🔹 crawl4ai 🤖
An intelligent web scraping framework powered by LLMs. It allows automatic extraction of structured data from HTML pages using prompt-driven extraction logic.

### 🔹 AsyncIO ⏱️🔄
Used to perform concurrent scraping across multiple pages without blocking, improving speed and performance.

### 🔹 LLM Extraction Strategy (`groq/deepseek`) 🧠
Used to extract only the most relevant jobs based on the user’s CV context. It ensures meaningful matches instead of random job listings.

---

## 🧪 How It Works

1. **Step 1 – Add your CV**  
   Place your `cv.pdf` in the project's root directory.

2. . **Step 2 – Run OCR & Job Matching**  
   Execute the main script to process your CV and match it against current job listings:
   ```bash
   python main.py
   ```

   This script performs the following:
   - Extracts text from your CV
   - Splits your CV into chunks
   - Embeds and uploads them to Weaviate (if not already processed)
   - Scrapes job listings
   - Matches listings using LLM-based extraction
   - Saves the final matched jobs

   - 📂 Output: 
     - `extracted_texts/cv.txt`  – Structured text extracted from your CV
     - `complete_venues.csv`  – List of matched job opportunities

---

## 🧩 Internal Workflow

```
cv.pdf
  ↓ (ocr.py)
cv.txt
  ↓ (main.py)                     ↓ (scrape.py)
Vectorized Embeddings → Weaviate ← Scraped Jobs ← Dialog.lk
         ↓                                ↓
     LLM-based Matching (LLMExtractionStrategy)
         ↓
     Output CSV: complete_venues.csv
```

---

## ✅ Requirements

### ⚙️ System Requirements (Host Machine)
- `Tesseract-OCR` – for optical character recognition
- `Docker` & `Docker Compose` – required if hosting `Weaviate` locally
- `Conda` (Miniconda or Anaconda) – recommended for managing the Python environment


### 📦 Project Environment Setup
1. Create and activate the Conda environment:
     ```bash
      conda create -n job-matcher python=3.9
      conda activate job-matcher
     ```

2. Install Python dependencies
     ```bash
     pip install -r requirements.txt
     ```

3. Install Chromium for Playwright
     ```bash
     playwright install chromium
     ```


### 📁 Additional Setup
- Rename the `.env_sample` file to `.env` and configure the necessary environment variables.
  - To start Weaviate locally:
     ```bash
    cd docker
    docker-compose up -d
     ```
---

## 🧠 Future Improvements

- Upload CVs via a simple web interface
- Improve job source diversity (LinkedIn, Glassdoor, etc.)

---

## 👨‍💻 Author

Made with ❤️ by Gihan Lakmal 

