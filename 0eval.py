from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt",task="detect")  # 加载预训练模型（建议用于训练）
result = model(source=r"C:\my\Project\052SuperGlue_train\SuperGlue-pytorch\coco\val2014\COCO_val2014_000000215867.jpg")
print(type(result[0]),len(result[0]))
print(type(result[0][0]),len(result[0][0]),dir(result[0][0]))
print(type(result[0][1]),len(result[0][1]),dir(result[0][1]))
print(type(result[0][2]),len(result[0][2]),dir(result[0][2]))
print(type(result[0][2]),len(result[0][2]),result[0][2].boxes)
# print(result[0][0][0])
# print(result[0][0][0][0])
# print(result.result)