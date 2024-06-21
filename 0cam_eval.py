import cv2
import time
import torch
from ultralytics import YOLO

# 加载YOLOv8模型
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s')
model = YOLO("yolov8n.pt",task="detect")  # 加载预训练模型（建议用于训练）

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取每一帧
while True:
    start_time = time.time()  # 记录开始时间

    ret, frame = cap.read()

    if not ret:
        print("无法接收帧")
        break

    # 使用YOLOv8模型进行检测
    results = model(frame)

    # 提取检测结果并绘制在帧上
    for result in results[0]:
        x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
        conf, cls = result.boxes.conf[0].cpu().numpy(),result.boxes.cls[0].cpu().numpy()
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # for result in results[0]:
    #     # 提取检测框的坐标、置信度和类别
    #     x1, y1, x2, y2, conf, cls = result['xmin'], result['ymin'], result['xmax'], result['ymax'], result['confidence'], result['class']
    #     label = f"{model.names[int(cls)]} {conf:.2f}"
    #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 计算并显示帧率
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示处理后的帧
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
