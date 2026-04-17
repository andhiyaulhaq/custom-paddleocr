import os
import sys
import cv2
import numpy as np
import json

# Add src to path so we can import ocr_engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ocr_engine.engines.onnx import ONNXTextDetector, ONNXTextRecognizer
from ocr_engine.utils.geometry import get_rotate_crop_image

# --- 1. Initialize Manual ONNX Mobile OCR Engine ---
print("Initializing ONNX Mobile Text Detector & Recognizer...")
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, "..")

# Paths to ONNX models
# Replacing server detection with mobile version as requested
det_model_path = os.path.join(root_dir, "models", "PP-OCRv5_mobile_det.onnx")
rec_model_path = os.path.join(root_dir, "models", "en_PP-OCRv5_mobile_rec.onnx")
dict_path = os.path.join(root_dir, "src", "ocr_engine", "resources", "en_dict.txt")

# Handle case where ONNX might not exist yet
if not os.path.exists(det_model_path) or not os.path.exists(rec_model_path):
    print("Warning: ONNX model files not found. Please run src/convert_to_onnx.py first.")
    print("For more info, see the 'Export Models to ONNX' section in README.md")
    print(f"Expected: {os.path.relpath(det_model_path)} and {os.path.relpath(rec_model_path)}")
    exit(1)

det_model = ONNXTextDetector(model_path=det_model_path)
rec_model = ONNXTextRecognizer(model_path=rec_model_path, dict_path=dict_path)

# --- 2. Load image ---
image_path = os.path.join(root_dir, "data", "samples", "1.jpg")
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image at {image_path}")
    exit(1)
output_img = img.copy()

# --- 3. Run Manual OCR Flow ---
print("Running ONNX Mobile Text Detection...")
boxes = det_model.predict(img)
print(f"Found {len(boxes)} text boxes.")

print("Extracting crops & Running ONNX Text Recognition...")
img_crop_list = []
for box in boxes:
    pts = np.array(box, dtype=np.float32)
    img_crop = get_rotate_crop_image(img, pts)
    img_crop_list.append(img_crop)

rec_results = rec_model.predict(img_crop_list)
texts = [res[0] for res in rec_results]

# --- Output to JSON for comparison ---
os.makedirs(os.path.join(root_dir, "results", "ocr"), exist_ok=True)
with open(os.path.join(root_dir, "results", "ocr", "inference_onnx_mobile.json"), "w") as f:
    json.dump({"boxes": [[[float(c) for c in pt] for pt in box] for box in boxes], "texts": texts}, f, indent=2)

# --- 4. Remove original text (white fill) ---
print("Redrawing image...")
for box in boxes:
    pts = np.array(box, dtype=np.int32)
    cv2.fillPoly(output_img, [pts], (255, 255, 255))

# --- 5. Draw clean OCR text ---
for box, text in zip(boxes, texts):
    if not text:
        continue
    pts = np.array(box, dtype=np.int32)

    # Get top-left corner
    x = int(min(pts[:, 0]))
    y = int(min(pts[:, 1]))

    # Estimate box size
    h = int(np.linalg.norm(pts[0] - pts[3]))
    w = int(np.linalg.norm(pts[0] - pts[1]))

    # Start small
    font_scale = 0.3
    thickness = 1

    # Increase scale until it fits the box
    while True:
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        if text_w > w * 0.9 or text_h > h * 0.9:
            font_scale -= 0.02
            break
        font_scale += 0.02

    font_scale = max(font_scale, 0.3)
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    # If text is still too wide at the smallest scale, squeeze it horizontally
    if text_w > w and font_scale <= 0.3:
        mask_h = text_h + baseline
        text_mask = np.zeros((mask_h, text_w), dtype=np.uint8)
        cv2.putText(text_mask, text, (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        
        # Rescale text width to fit exactly into the box using INTER_AREA for better quality
        squeezed_mask = cv2.resize(text_mask, (w, mask_h), interpolation=cv2.INTER_AREA)
        
        # Draw the squeezed text with Alpha Blending
        text_y_pos = y + (h - mask_h) // 2
        img_h, img_w = output_img.shape[:2]
        r_y, r_x = max(0, text_y_pos), max(0, x)
        r_h, r_w = min(img_h - r_y, mask_h), min(img_w - r_x, w)
        
        if r_h > 0 and r_w > 0:
            # Use alpha blending to keep anti-aliasing smooth
            alpha = squeezed_mask[:r_h, :r_w, np.newaxis].astype(np.float32) / 255.0
            roi = output_img[r_y:r_y+r_h, r_x:r_x+r_w].astype(np.float32)
            blended = roi * (1.0 - alpha) + (0, 0, 0) * alpha
            output_img[r_y:r_y+r_h, r_x:r_x+r_w] = blended.astype(np.uint8)
    else:
        # Calculate horizontal centering
        text_x = x + (w - text_w) // 2
        text_y = y + (h + text_h) // 2
        cv2.putText(
            output_img,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
    cv2.polylines(output_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# --- 6. Save result ---
output_path = os.path.join(root_dir, "results", "ocr", "inference_onnx_mobile.jpg")
cv2.imwrite(output_path, output_img)

print(f"Saved ONNX Mobile OCR output to: {output_path}")
