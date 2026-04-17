import cv2
import numpy as np
import math
import paddle.inference as paddle_infer
from ..utils.postprocess import DBPostProcess, CTCLabelDecode

def create_predictor(model_dir):
    config = paddle_infer.Config(f"{model_dir}/inference.json", f"{model_dir}/inference.pdiparams")
    config.disable_gpu()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    return paddle_infer.create_predictor(config)

class RawTextDetector:
    def __init__(self, model_dir):
        self.predictor = create_predictor(model_dir)
        self.postprocess = DBPostProcess()
        
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
        
        input_names = self.predictor.get_input_names()
        self.predictor.get_input_handle(input_names[0]).copy_from_cpu(norm_img)
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        out = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        
        boxes = self.postprocess(out[:, 0, :, :], [[h, w, ratio_h, ratio_w]])[0]
        return boxes

class RawTextRecognizer:
    def __init__(self, model_dir, dict_path):
        self.predictor = create_predictor(model_dir)
        self.postprocess = CTCLabelDecode(dict_path)

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
        
        input_names = self.predictor.get_input_names()
        self.predictor.get_input_handle(input_names[0]).copy_from_cpu(norm_img_batch)
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        out = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        
        return self.postprocess(out)
