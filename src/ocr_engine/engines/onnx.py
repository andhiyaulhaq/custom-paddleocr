import cv2
import numpy as np
import onnxruntime as ort
import math
from ..utils.postprocess import DBPostProcess, CTCLabelDecode
from ..utils.geometry import get_rotate_crop_image

def create_predictor(model_path):
    # ONNX Runtime session
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

class ONNXTextDetector:
    def __init__(self, model_path):
        self.session = create_predictor(model_path)
        self.postprocess = DBPostProcess()
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, img):
        h, w, _ = img.shape
        ratio = float(960) / max(h, w)
        resize_h = max(int(round(h * ratio / 32) * 32), 32)
        resize_w = max(int(round(w * ratio / 32) * 32), 32)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        
        resized_img = cv2.resize(img, (resize_w, resize_h))
        resized_img_rgb = resized_img[:, :, ::-1] # BGR to RGB
        norm_img = (resized_img_rgb.astype("float32") * (1./255.) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        norm_img = norm_img.astype("float32").transpose((2, 0, 1))
        norm_img = np.expand_dims(norm_img, 0)
        
        out = self.session.run(None, {self.input_name: norm_img})[0]
        
        boxes = self.postprocess(out[:, 0, :, :], [[h, w, ratio_h, ratio_w]])[0]
        return boxes

class ONNXTextRecognizer:
    def __init__(self, model_path, dict_path):
        self.session = create_predictor(model_path)
        self.postprocess = CTCLabelDecode(dict_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img_list):
        if len(img_list) == 0: return []
        
        max_wh_ratio = 0
        for img in img_list:
            max_wh_ratio = max(max_wh_ratio, img.shape[1] * 1.0 / img.shape[0])
            
        norm_img_batch = []
        for img in img_list:
            h, w = img.shape[:2]
            ratio = w / float(h)
            resize_w = math.ceil(48 * ratio)
            resize_w = min(resize_w, int(48 * max_wh_ratio))
            resized_img = cv2.resize(img, (resize_w, 48))
            resized_img_rgb = resized_img[:, :, ::-1] # BGR to RGB
            
            # normalize first
            norm_im = (resized_img_rgb.astype("float32") / 127.5) - 1.0
            norm_im = norm_im.transpose((2, 0, 1))
            
            # pad with 0.0 (middle gray after norm)
            padding_im = np.zeros((3, 48, int(48 * max_wh_ratio)), dtype=np.float32)
            padding_im[:, :, 0:resize_w] = norm_im
            
            norm_img_batch.append(padding_im)
            
        norm_img_batch = np.array(norm_img_batch)
        
        out = self.session.run(None, {self.input_name: norm_img_batch})[0]
        
        return self.postprocess(out)

class ONNXOCREngine:
    """
    High-level engine that combines text detection and recognition.
    """
    def __init__(self, det_model_path, rec_model_path, dict_path):
        self.detector = ONNXTextDetector(det_model_path)
        self.recognizer = ONNXTextRecognizer(rec_model_path, dict_path)

    def predict(self, img):
        """
        Runs full OCR pipeline: detection -> cropping -> recognition.
        
        Args:
            img: BGR image (OpenCV format).
            
        Returns:
            Tuple of (boxes, texts).
        """
        # 1. Detection
        boxes = self.detector.predict(img)
        if len(boxes) == 0:
            return [], []

        # 2. Cropping
        img_crop_list = []
        for box in boxes:
            pts = np.array(box, dtype=np.float32)
            img_crop = get_rotate_crop_image(img, pts)
            img_crop_list.append(img_crop)

        # 3. Recognition
        rec_results = self.recognizer.predict(img_crop_list)
        texts = [res[0] for res in rec_results]
        
        return boxes, texts
