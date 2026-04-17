import cv2
import numpy as np

def draw_ocr_results_with_replace(image, boxes, texts, font_scale_min=0.3, thickness=1):
    """
    Draws OCR results onto the image, replacing the original text area.
    
    Args:
        image: Original image.
        boxes: List of bounding boxes.
        texts: List of recognized text strings.
        font_scale_min: Minimum font scale allowed before squeezing.
        thickness: Font thickness.
        
    Returns:
        Image with OCR results drawn.
    """
    output_img = image.copy()
    
    # 1. Clear original text (white fill)
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(output_img, [pts], (255, 255, 255))
    
    # 2. Draw clean OCR text
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

        # Find best font scale
        font_scale = font_scale_min
        while True:
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            if text_w > w * 0.9 or text_h > h * 0.9:
                font_scale -= 0.02
                break
            font_scale += 0.02

        font_scale = max(font_scale, font_scale_min)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Handle horizontal squeezing for tight boxes
        if text_w > w and font_scale <= font_scale_min:
            mask_h = text_h + baseline
            text_mask = np.zeros((mask_h, text_w), dtype=np.uint8)
            cv2.putText(text_mask, text, (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
            
            # Squeeze horizontally with high-quality interpolation
            squeezed_mask = cv2.resize(text_mask, (w, mask_h), interpolation=cv2.INTER_AREA)
            
            # Draw using alpha blending for smooth anti-aliasing
            text_y_pos = y + (h - mask_h) // 2
            img_h, img_w = output_img.shape[:2]
            r_y, r_x = max(0, text_y_pos), max(0, x)
            r_h, r_w = min(img_h - r_y, mask_h), min(img_w - r_x, w)
            
            if r_h > 0 and r_w > 0:
                alpha = squeezed_mask[:r_h, :r_w, np.newaxis].astype(np.float32) / 255.0
                roi = output_img[r_y:r_y+r_h, r_x:r_x+r_w].astype(np.float32)
                blended = roi * (1.0 - alpha) + (0, 0, 0) * alpha
                output_img[r_y:r_y+r_h, r_x:r_x+r_w] = blended.astype(np.uint8)
        else:
            # Standard center-aligned rendering
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
        
        # Draw bounding box outline
        cv2.polylines(output_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
    return output_img
