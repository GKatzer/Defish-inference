import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt

class YOLO_ONNX_Inference:
    def __init__(self, onnx_path, img_size=960, conf_thres=0.25, iou_thres=0.45):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load the model through ONNX Runtime
        providers = ['CPUExecutionProvider']  # 'CUDAExecutionProvider' can be used if there is a GPU
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get information about the model input
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        print(f"📦 Model loaded. Input size: {input_shape}")
    
    def preprocess(self, image):
        """Image preparation for YOLO"""
        h, w = image.shape[:2]
        
        # Letterbox
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create the canvas
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        y_offset = (self.img_size - new_h) // 2
        x_offset = (self.img_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Convert for the model
        image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0
        
        # Transpose to the format (1, 3, H, W)
        input_tensor = image_rgb.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_tensor, (w, h), (x_offset, y_offset), scale
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def postprocess(self, outputs, original_shape, offset, scale):
        """Postprocessing for the output (1, 5, 8400)"""
        original_w, original_h = original_shape
        x_offset, y_offset = offset

        # Get the matrix (1, 5, 8400)
        pred = outputs[0]  # shape (1, 5, 8400)

        # Permute the dimensions → (8400, 5)
        pred = pred[0].T

        # Split
        boxes_xywh = pred[:, :4]      # (8400, 4)
        conf = pred[:, 4]             # (8400,)

        # Filter by the threshold
        mask = conf > self.conf_thres
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        conf = conf[mask]

        # xywh → xyxy
        boxes = np.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove the letterbox padding
        boxes[:, [0, 2]] -= x_offset
        boxes[:, [1, 3]] -= y_offset
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale

        # Clip the excess values
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)

        # Apply NMS
        keep = self.nms(boxes, conf, self.iou_thres)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf[i]),
                "class_id": 0      # one class
            })

        return detections

    def predict_image(self, image):
        input_tensor, original_shape, offset, scale = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, original_shape, offset, scale)
        return detections

    def draw_detections(self, image, detections):
        """Drawing the detections on the image"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Colour for the class
            color = (0, 255, 0)  # Green
            
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            
            # Label
            label = f"{class_id}: {confidence:.2f}"

            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        return image

    def predict_with_visualization(self, image):
        preds = self.predict_image(image)
        vis = self.draw_detections(image.copy(), preds)
        return preds, vis

    def get_bboxes(self, image):
        """
        Returns the bboxes in the coordinates of the original image
        """
        input_tensor, original_shape, offset, scale = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        detections = self.postprocess(
            outputs,
            original_shape,
            offset,
            scale
        )

        # Remove everything unnecessary
        bboxes = []
        for det in detections:
            bboxes.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"]
            })

        return bboxes

# Usage
def main():
    # Initialisation
    detector = YOLO_ONNX_Inference("best.onnx", img_size=960)
    
    # Prediction
    detections, result_image = detector.predict(
        r"C:\Projects\Fish_Guard_ML_dev\blue-runners--947498612-5c634cd3c9e77c0001566e32.webp"
    )
    
    # Print the results
    print("=" * 60)
    print(f"Objects detected: {len(detections)}")
    print("=" * 60)
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        print(f"Object {i+1}:")
        print(f"  Class: {det['class_id']}")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Coordinates: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        print(f"  Size: {x2-x1:.0f}×{y2-y1:.0f} pixels")
        print()
    
    # Save and display
    cv2.imwrite('result_onnx_runtime.jpg', result_image)
    
    # Display through matplotlib
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.imshow(result_rgb)
    plt.axis('off')
    plt.title(f'Objects detected: {len(detections)}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()