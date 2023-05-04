import json
import cv2
# Load ảnh và tìm keypoints và descriptors
img = cv2.imread('/Users/macbookair/Library/CloudStorage/GoogleDrive-ttrung2h@gmail.com/My Drive/Project/Extract_Info_From_Card/Images/std.jpg')
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# Chuyển đổi keypoints và descriptors sang kiểu dữ liệu có thể lưu vào file JSON
keypoints_list = []
for kp in keypoints:
    keypoints_list.append((kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id))
descriptors_list = descriptors.tolist()

# Lưu keypoints và descriptors vào file JSON
with open('keypoints_descriptors.json', 'w') as f:
    json.dump({'keypoints': keypoints_list, 'descriptors': descriptors_list,'shape': img.shape}, f)
