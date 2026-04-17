import os
import sys
import cv2
import json

# Add src to path so we can import ocr_engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr_engine.engines.onnx import ONNXOCREngine
from ocr_engine.utils.visualization import draw_ocr_results_with_replace

# --- 1. Initialize Manual ONNX Server OCR Engine ---
print("Initializing ONNX Server OCR Engine...")
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, "..")

# Paths to ONNX models
det_model_path = os.path.join(root_dir, "models", "PP-OCRv5_server_det.onnx")
rec_model_path = os.path.join(root_dir, "models", "en_PP-OCRv5_mobile_rec.onnx")
dict_path = os.path.join(root_dir, "src", "ocr_engine", "resources", "en_dict.txt")

# Handle case where ONNX might not exist yet
if not os.path.exists(det_model_path) or not os.path.exists(rec_model_path):
    print("Warning: ONNX model files not found. Please run src/convert_to_onnx.py first.")
    exit(1)

engine = ONNXOCREngine(det_model_path, rec_model_path, dict_path)

# --- 2. Load image ---
image_path = os.path.join(root_dir, "data", "samples", "1.jpg")
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit(1)

# --- 3. Run Modular OCR Flow ---
print("Running full ONNX Server OCR Pipeline...")
boxes, texts = engine.predict(img)
print(f"Found {len(boxes)} text boxes.")

# --- 4. Output to JSON for comparison ---
os.makedirs(os.path.join(root_dir, "results", "ocr"), exist_ok=True)
with open(os.path.join(root_dir, "results", "ocr", "inference_onnx_server.json"), "w") as f:
    json.dump({"boxes": [[[float(c) for c in pt] for pt in box] for box in boxes], "texts": texts}, f, indent=2)

# --- 5. Redraw image using modular visualization ---
print("Redrawing image with text replacement...")
output_img = draw_ocr_results_with_replace(img, boxes, texts, font_scale_min=0.3)

# --- 6. Save result ---
output_path = os.path.join(root_dir, "results", "ocr", "inference_onnx_server.jpg")
cv2.imwrite(output_path, output_img)

print(f"Saved ONNX Server OCR output to: {output_path}")
