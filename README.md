# 🤖 Intelligent Document Receipt Analyzer

An AI-powered document analysis application using **InternVL2.5-4B** vision-language model with GPU acceleration. Analyze receipts, invoices, forms, contracts, and various document types with advanced computer vision.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.2.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)

## ✨ Features

- 🎯 **6 Analysis Modes**:
  - Quick Summary
  - Detailed Analysis
  - Structured Extraction
  - Receipt/Invoice Analysis
  - Form Analysis
  - Table Extraction

- 🚀 **GPU Acceleration**: Powered by NVIDIA CUDA for fast inference
- 🎨 **Modern Glass UI**: Beautiful gradient theme with glassmorphism effects
- 📄 **Multi-Document Support**: Receipts, invoices, forms, contracts, IDs, business cards, certificates
- 🔍 **Optional OCR**: Tesseract OCR integration for text extraction
- 📊 **Structured Output**: Extracts key fields like vendor name, date, amount, currency, etc.

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.12 or higher
- **RAM**: 8 GB
- **Storage**: 10 GB free space

### Recommended (for GPU acceleration)
- **GPU**: NVIDIA GPU with 6+ GB VRAM (e.g., RTX 4050, RTX 3060, or better)
- **CUDA**: 12.1 or higher
- **RAM**: 16 GB
- **Storage**: 20 GB free space (for model cache)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/intelligent-document-processing.git
cd intelligent-document-processing
```

### 2. Create Virtual Environment (Python 3.12)
```bash
python -m venv .venv312
```

### 3. Activate Virtual Environment

**Windows:**
```powershell
.venv312\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv312/bin/activate
```

### 4. Install PyTorch with CUDA (for GPU support)
```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```bash
pip install torch==2.9.1 torchvision==0.24.1
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

### 6. Install Tesseract OCR (Optional)
- **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

## 🚀 Usage

### Run the Application
```bash
python app.py
```

The application will:
1. Check GPU availability
2. Load the InternVL2.5-4B model (first run downloads ~8GB)
3. Launch Gradio interface at `http://localhost:7860`

### Using the Interface
1. **Upload Document**: Click "📄 Upload Your Document" or drag & drop
2. **Select Analysis Mode**: Choose from 6 analysis types
3. **Optional OCR**: Enable for raw text extraction
4. **Analyze**: Click "🚀 Analyze Document"
5. **View Results**: Structured output with extracted fields

## 📋 Output Format

All analysis modes extract:
- **DOCUMENT_TYPE**: Type of document
- **VENDOR_NAME**: Company/organization name
- **INVOICE_NUMBER**: Receipt/invoice number
- **DATE**: Transaction or document date
- **CURRENCY**: Currency type (USD, EUR, etc.)
- **TOTAL_AMOUNT**: Total amount with currency
- **SUMMARY**: Detailed summary of the document

## 🔧 Configuration

### GPU Settings
The application automatically detects and uses GPU if available. To force CPU mode:
```python
device = torch.device("cpu")
```

### Model Settings
Change model in `app.py`:
```python
MODEL_ID = "OpenGVLab/InternVL2_5-4B"  # Current model
```

### Performance Tuning
- **Reduce memory usage**: Set `max_new_tokens` lower in generation config
- **Faster inference**: Ensure FP16 precision on GPU
- **Batch processing**: Modify `process_document()` for multiple files

## 📊 Supported Document Types

| Category | Examples |
|----------|----------|
| 📄 **Financial** | Receipts, Invoices, Bank Statements, Tax Forms |
| 📋 **Forms** | Applications, Surveys, Registration Forms |
| 📜 **Legal** | Contracts, Agreements, Certificates |
| 🆔 **Identity** | ID Cards, Passports, Driver's Licenses |
| 💼 **Business** | Business Cards, Letters, Reports |
| 📊 **Data** | Tables, Charts, Spreadsheets |

## 🛠️ Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu121
   ```

### Model Loading Errors
1. Check internet connection (first run downloads model)
2. Ensure sufficient disk space (~10GB)
3. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface`

### Memory Errors
- Reduce `max_new_tokens` in generation config
- Use CPU mode if GPU memory insufficient
- Close other GPU-intensive applications

---

**⚡ Performance Note:** GPU acceleration recommended for faster processing. CPU mode will work but may be slower.
