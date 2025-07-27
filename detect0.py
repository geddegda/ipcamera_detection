import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # change the model here if you wish
RTSP_URL = "0"
#RTSP_URL = "rtsp://usr:pwd@192.168.1.200:554/h264Preview_01_sub" # sub or main
ts = 0
save_dir=f"/home/guillaume/ram_videos/recording_{ts}"

results = model.predict(
        RTSP_URL, 
        stream=True, 
        conf= 0.6, 
        stream_buffer= False,
        save= True,
        classes = [0])

for result in results:
    ts = time.strftime("%Y%m%d_%H%M%S")
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
  
    if result.boxes is not None and result.boxes.cls.numel() > 0:
        result.save(filename=f"/home/guillaume/ram_videos/result_{ts}.jpg")  
