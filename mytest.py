import json
from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("yolo11n-pose.pt")

# Define path to the image file
source = "chr.jpg"

# Run inference on the source
results = model(source)  # list of Results objects

# Define the skeleton connection for YOLOv11pose
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

# Initialize a list to store results in JSON format
results_json = []

# Loop through each result
for result in results:
    # Extract keypoints
    keypoints = result.keypoints.data.tolist() if result.keypoints is not None else None
    
    # Extract bounding boxes
    boxes = result.boxes.data.tolist() if result.boxes is not None else None
    
    # Extract masks
    masks = result.masks.data.tolist() if result.masks is not None else None
    
    # Extract probabilities (for classification tasks)
    probs = result.probs.data.tolist() if result.probs is not None else None
    
    # Extract oriented bounding boxes (OBB)
    obb = result.obb.data.tolist() if result.obb is not None else None
    
    # Create a dictionary for the current result
    result_dict = {
        "keypoints": keypoints,
        "boxes": boxes,
        "masks": masks,
        "probs": probs,
        "obb": obb
    }
    
    # Append the result dictionary to the JSON list
    results_json.append(result_dict)

    # Plot the result on the original image
    annotated_frame = result.plot()
    
    # Draw skeleton connections on the annotated frame
    for person_keypoints in keypoints:
        for connection in skeleton:
            start_idx, end_idx = connection
            start_point = person_keypoints[start_idx]
            end_point = person_keypoints[end_idx]
            
            # Check if both points are valid (not (0, 0))
            if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                cv2.line(annotated_frame, (int(start_point[0]), int(start_point[1])), 
                         (int(end_point[0]), int(end_point[1])), color=(0, 255, 0), thickness=2)

    # Save the annotated image
    output_image_path = f"annotated_result_{len(results_json)-1}.jpg"
    cv2.imwrite(output_image_path, annotated_frame)
    print(f"Saved annotated image to {output_image_path}")

# Convert the list of results to a JSON string
results_json_str = json.dumps(results_json, indent=4)

# Print the JSON string
print(results_json_str)

# Optionally, save the JSON string to a file
with open("results.json", "w") as json_file:
    json_file.write(results_json_str)