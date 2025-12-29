import gradio as gr
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch
import warnings
from datetime import datetime
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings("ignore")

# Load InternVL2.5-4B Model - Excellent for general document understanding
# 4B parameter model with strong vision-language capabilities
MODEL_ID = "OpenGVLab/InternVL2_5-4B"

# GPU Detection and Diagnostics
print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA is NOT available. Possible reasons:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. NVIDIA drivers not installed or outdated")
    print("3. CUDA toolkit not installed")
    print("\nTo install PyTorch with CUDA, visit: https://pytorch.org/get-started/locally/")
print("=" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

print(f"\nLoading model {MODEL_ID} on {device}...")
try:
    # Load InternVL2.5 model with appropriate settings
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    
    # Load image processor for InternVL
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"Model loaded successfully on {device}!")
    if torch.cuda.is_available():
        print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
        print(f"Using FP16 precision for faster inference")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to CPU mode or check installation")
    model = None
    tokenizer = None
    image_processor = None

# --- Preprocessing Functions ---

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(gray_image):
    return cv2.GaussianBlur(gray_image, (5, 5), 0)

def binarize_image(blur_reduced_image):
    return cv2.adaptiveThreshold(
        blur_reduced_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        11,
        4
    )

def deskew_image(image):
    coords = cv2.findNonZero(image)
    if coords is None:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1] - 90
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def preprocess_pipeline(image_path_or_array):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    elif isinstance(image_path_or_array, np.ndarray):
        # Gradio passes RGB, OpenCV uses BGR
        image = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGB2BGR)
    else:
        return None, None

    gray = convert_to_grayscale(image)
    blur = reduce_noise(gray)
    binary = binarize_image(blur)
    deskewed = deskew_image(binary)
    return image, deskewed

# --- Extraction Functions ---

def extract_text_tesseract(image):
    # image can be numpy array or PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    try:
        text = pytesseract.image_to_string(image)
        return text if text.strip() else "No text detected"
    except Exception as e:
        return f"Error in Tesseract extraction: {str(e)}"

def format_analysis_output(raw_text, analysis_type):
    """Format the output in a more structured and readable way"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    formatted = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    📄 DOCUMENT ANALYSIS REPORT                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Analysis Type: {analysis_type.upper():<48} ║
║  Timestamp: {timestamp:<52} ║
╚══════════════════════════════════════════════════════════════════╝

{raw_text}

╔══════════════════════════════════════════════════════════════════╗
║                         ✅ ANALYSIS COMPLETE                      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return formatted

def analyze_document_with_vision(image, analysis_type="summary"):
    """
    Analyze document using InternVL2.5 vision model
    analysis_type: 'summary', 'detailed', 'extract', 'receipt', 'form', 'table'
    """
    if model is None or tokenizer is None or image_processor is None:
        return "❌ Error: Model not loaded. Please check installation."
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
    
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess image using the image processor
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pixel_values = pixel_values.to(device).to(torch.float16)
    else:
        pixel_values = pixel_values.to(device)
    
    # Define prompts based on analysis type
    prompts = {
        "summary": """Analyze this document and extract information in the following JSON-like format:

DOCUMENT_TYPE: [Receipt/Invoice/Bill/Statement/etc.]

VENDOR_NAME: [Company or store name]

INVOICE_NUMBER: [Invoice/Receipt number if visible, otherwise "N/A"]

DATE: [Transaction or document date]

CURRENCY: [Currency symbol or code]

TOTAL_AMOUNT: [Final total amount with currency]

SUMMARY: [Brief summary of what this document is about, including key items or purpose]
""",

        "detailed": """Analyze this document and extract all information in the following structured format:

DOCUMENT_TYPE: [Receipt/Invoice/Bill/Statement/Contract/Form/etc.]

VENDOR_NAME: [Company, store, or organization name]

INVOICE_NUMBER: [Invoice/Receipt/Reference number if visible, otherwise "N/A"]

DATE: [Transaction or document date, include time if visible]

CURRENCY: [Currency symbol or code (USD, EUR, GBP, etc.)]

TOTAL_AMOUNT: [Final total amount with currency]

SUMMARY: [Detailed summary including:
- Purpose of the document
- Main items or services (if applicable)
- Key terms or conditions (if applicable)
- Important notes or special information
- Payment status or method (if visible)]

ADDITIONAL_DETAILS:
- Subtotal: [Amount if visible]
- Tax: [Amount and rate if visible]
- Discounts: [Amount if applicable]
- Payment Method: [Method if visible]
- Customer Info: [Name, contact if visible]
- Addresses: [Any addresses found]
""",

        "extract": """Extract all key information in this structured format:

DOCUMENT_TYPE: [Type]

VENDOR_NAME: [Company/Organization name]

INVOICE_NUMBER: [Number/ID]

DATE: [All dates found]

CURRENCY: [Currency type]

TOTAL_AMOUNT: [Total with currency]

SUMMARY: [What this document represents and key points]

OTHER_FIELDS:
• Line Items: [List items with quantities and prices if applicable]
• Contact Information: [Phone, email, website]
• Addresses: [Physical addresses]
• Terms: [Payment terms, due dates, etc.]
""",

        "receipt": """Extract receipt/invoice information in this format:

DOCUMENT_TYPE: Receipt/Invoice

VENDOR_NAME: [Store/Company name]

INVOICE_NUMBER: [Receipt/Invoice number]

DATE: [Transaction date and time]

CURRENCY: [Currency (USD, EUR, etc.)]

TOTAL_AMOUNT: [Final total with currency]

SUMMARY: [Brief description of purchase including number of items and payment method]

DETAILED_BREAKDOWN:
• Items: [List each item with quantity and price]
• Subtotal: [Amount]
• Tax: [Amount and rate]
• Discounts: [Amount if any]
• Payment Method: [How it was paid]
""",

        "form": """Extract form information in this format:

DOCUMENT_TYPE: Form/Application

VENDOR_NAME: [Organization issuing the form]

INVOICE_NUMBER: [Form number/reference if visible, otherwise "N/A"]

DATE: [Form date or submission date]

CURRENCY: [If financial form, otherwise "N/A"]

TOTAL_AMOUNT: [If financial form, otherwise "N/A"]

SUMMARY: [What the form is for and its current status]

FORM_FIELDS:
• [Field name]: [Value]
• Checkboxes: [Status of selections]
• Signatures: [Status]
""",

        "table": """Extract table data in this format:

DOCUMENT_TYPE: [Document containing tables]

VENDOR_NAME: [Organization if identifiable]

INVOICE_NUMBER: [Reference number if visible, otherwise "N/A"]

DATE: [Document date]

CURRENCY: [If financial data present]

TOTAL_AMOUNT: [If applicable]

SUMMARY: [Description of what the table(s) contain and their purpose]

TABLE_DATA:
[Preserve table structure with headers and rows]
"""
    }
    
    prompt = prompts.get(analysis_type, prompts["summary"])
    
    # Prepare question for InternVL2.5
    question = prompt
    
    try:
        # Verify GPU usage
        if torch.cuda.is_available():
            print(f"✓ Processing on GPU")
        
        # Generate response using InternVL's chat method
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={
                    'max_new_tokens': 1024,
                    'do_sample': False,
                }
            )
        
        return format_analysis_output(response, analysis_type)
        
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}\n\nPlease ensure the model is properly loaded."

# --- Main Gradio Processing ---

def process_document(input_image, analysis_type, show_ocr):
    if input_image is None:
        return "⚠️ No image uploaded. Please upload a document to analyze.", ""
    
    # Analyze using Vision Model
    vision_analysis = analyze_document_with_vision(input_image, analysis_type)
    
    # Get OCR text if requested
    ocr_text = ""
    if show_ocr:
        _, processed_img_cv = preprocess_pipeline(input_image)
        ocr_text = extract_text_tesseract(processed_img_cv)
    
    return vision_analysis, ocr_text

# Custom CSS for Glass theme
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
}

.prose h1, .prose h2, .prose h3 {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
}

.prose p, .prose li {
    color: rgba(255, 255, 255, 0.95) !important;
}

textarea {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(245, 247, 255, 0.98) 100%) !important;
    border: 2px solid rgba(102, 126, 234, 0.4) !important;
    border-radius: 15px !important;
    font-family: 'Courier New', monospace !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    color: #1a1a2e !important;
    font-size: 14px !important;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5) !important;
}

.btn-primary:hover {
    box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.7) !important;
    transform: translateY(-2px);
}

.tab-nav button {
    background: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
}

.tab-nav button.selected {
    background: rgba(255, 255, 255, 0.3) !important;
}
"""

# UI Definition with Glass Theme
with gr.Blocks(title="Intelligent Document Analyzer", css=custom_css, theme=gr.themes.Glass()) as demo:
    
    gr.Markdown("""
    # 🤖 Intelligent Document Receipt
    ### Powered by InternVL2.5-4B Vision-Language Model
    """)
    
    gr.Markdown("""
    Upload any document type for AI-powered analysis: receipts, invoices, forms, contracts, ID cards, business cards, certificates, and more!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload & Configure")
            input_img = gr.Image(
                label="📄 Upload Your Document", 
                type="numpy", 
                height=350,
                sources=["upload", "clipboard"]
            )
            
            analysis_type = gr.Radio(
                choices=[
                    ("📝 Quick Summary", "summary"),
                    ("🔍 Detailed Analysis", "detailed"),
                    ("📊 Structured Extraction", "extract"),
                    ("🧾 Receipt/Invoice", "receipt"),
                    ("📋 Form Analysis", "form"),
                    ("📈 Table Extraction", "table")
                ],
                value="detailed",
                label="Analysis Mode",
                info="Select the type of analysis you need"
            )
            
            show_ocr = gr.Checkbox(
                label="Show OCR Text (Tesseract)", 
                value=False,
                info="Display raw text extraction alongside AI analysis"
            )
            
            submit_btn = gr.Button(
                "🚀 Analyze Document", 
                variant="primary", 
                size="lg",
                scale=1
            )
            
            gr.Markdown("""
            ---
            **💡 Pro Tips:**
            - Use **Detailed Analysis** for comprehensive extraction
            - Use **Receipt/Invoice** for financial documents
            - Use **Structured Extraction** for organized key-value pairs
            - Enable OCR for raw text comparison
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Analysis Results")
            
            vision_output = gr.TextArea(
                label="AI Vision Analysis",
                lines=25,
                max_lines=30,
                placeholder="Your AI-powered document analysis will appear here...\n\nUpload a document and click 'Analyze Document' to begin.",
                interactive=False
            )
            
            ocr_output = gr.TextArea(
                label="📋 OCR Text Output (Tesseract)",
                lines=10,
                placeholder="Enable 'Show OCR Text' to see raw text extraction...",
                interactive=False,
                visible=False
            )
    
    # Update OCR visibility based on checkbox
    show_ocr.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[show_ocr],
        outputs=[ocr_output]
    )
    
    # Submit button action
    submit_btn.click(
        process_document, 
        inputs=[input_img, analysis_type, show_ocr], 
        outputs=[vision_output, ocr_output]
    )
    
    gr.Markdown("""
    ---
    ### 📊 Supported Document Types
    
    | Category | Examples |
    |----------|----------|
    | 📄 **Financial** | Receipts, Invoices, Bank Statements, Tax Forms |
    | 📋 **Forms** | Applications, Surveys, Registration Forms |
    | 📜 **Legal** | Contracts, Agreements, Certificates |
    | 🆔 **Identity** | ID Cards, Passports, Driver's Licenses |
    | 💼 **Business** | Business Cards, Letters, Reports |
    | 📊 **Data** | Tables, Charts, Spreadsheets |
    
    ---
    **⚡ Performance Note:** GPU acceleration recommended for faster processing. CPU mode will work but may be slower.
    """)

if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        show_error=True
    )
