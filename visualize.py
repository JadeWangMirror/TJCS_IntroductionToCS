import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 定义 YOLOv11pose 的骨骼点连接顺序
skeleton = [
    (0, 1), (1, 3),  # 鼻子 -> 左眼 -> 左耳
    (0, 2), (2, 4),  # 鼻子 -> 右眼 -> 右耳
    (5, 6),          # 左肩 -> 右肩
    (5, 7), (7, 9),  # 左肩 -> 左肘 -> 左腕
    (6, 8), (8, 10), # 右肩 -> 右肘 -> 右腕
    (11, 12),        # 左髋 -> 右髋
    (5, 11), (11, 13), (13, 15),  # 左肩 -> 左髋 -> 左膝 -> 左脚踝
    (6, 12), (12, 14), (14, 16)   # 右肩 -> 右髋 -> 右膝 -> 右脚踝
]

# 读取 JSON 文件
def load_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

# 从 JSON 数据生成图像
def generate_image_from_json(json_data, output_image_path):
    # 创建一个空白图像
    image_width = 1280  # 假设图像宽度
    image_height = 1710  # 假设图像高度
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # 绘制关键点和骨骼连接线
    for result in json_data:
        keypoints = result.get("keypoints", [])
        boxes = result.get("boxes", [])

        # 绘制关键点
        for person_keypoints in keypoints:
            for point in person_keypoints:
                x, y, confidence = point
                if confidence > 0.5:  # 只绘制置信度大于 0.5 的关键点
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # 红色点

            # 绘制骨骼连接线
            for connection in skeleton:
                start_idx, end_idx = connection
                start_point = person_keypoints[start_idx]
                end_point = person_keypoints[end_idx]

                # 检查点是否有效
                if start_point[2] > 0.5 and end_point[2] > 0.5:
                    cv2.line(image, (int(start_point[0]), int(start_point[1])), 
                             (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)  # 绿色线

        # 绘制边界框
        for box in boxes:
            x1, y1, x2, y2, confidence, _ = box
            if confidence > 0.5:  # 只绘制置信度大于 0.5 的边界框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 蓝色框

    # 保存图像
    cv2.imwrite(output_image_path, image)
    print(f"Saved image to {output_image_path}")

# 主函数
if __name__ == "__main__":
    # 输入 JSON 文件路径
    json_file_path = "results.json"

    # 输出图像路径
    output_image_path = "annotated_image_from_json.jpg"

    # 加载 JSON 数据
    json_data = load_json(json_file_path)

    # 生成图像
    generate_image_from_json(json_data, output_image_path)