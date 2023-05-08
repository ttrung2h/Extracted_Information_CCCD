import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
class CardAlignment:
    def __init__(self, img_path,blur_size=5,keypoint_src = 'keypoints_descriptors.json'):
        # Check exist file
        if not os.path.exists(img_path):
            raise FileNotFoundError('Image not found')
        if not os.path.exists(keypoint_src):
            raise FileNotFoundError('Keypoint file not found')
        # Load image
        self.image = cv2.imread(img_path)

        #Load keypoints and descriptors
        self.keypoints_descriptors_dst = json.load(open(keypoint_src))
        self.blur_size = (blur_size,blur_size)
        

        #Load shape of standard image
        self.img_dst_shape = self.keypoints_descriptors_dst['shape']
    

    def warp_perspective_image(self,processing_img,check = False):
        # Sử dụng SIFT để tìm key points và descriptors cho ảnh gốc và ảnh mục tiêu
        sift = cv2.SIFT_create()
        
        # Get key points and descriptors for image
        keypoints_src, descriptors_src = sift.detectAndCompute(processing_img, None)

        # Load keypoints and descriptors for standard image
        keypoints_st = self.keypoints_descriptors_dst['keypoints']
        descriptors_st = self.keypoints_descriptors_dst['descriptors']

        # Tạo lại đối tượng keypoints từ dữ liệu trong file JSON
        keypoints_dst = []
        for kp_data in keypoints_st:
            kp = kp = cv2.KeyPoint(x=kp_data[0][0], y=kp_data[0][1], size=kp_data[1], angle=kp_data[2], response=kp_data[3], octave=int(kp_data[4]), class_id=int(kp_data[5]))
            keypoints_dst.append(kp)
        
        
        # Tạo lại đối tượng descriptors từ dữ liệu trong file JSON
        descriptors_dst = np.array(descriptors_st, dtype=np.float32)
        
        
        matches = cv2.BFMatcher()
        matches = matches.knnMatch(descriptors_src, descriptors_dst, k=2)
        

        # Find good matches if distance is less than 75% of the second best match
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if check == True:
            #Percent of good matches to all matches
            print("Number of matches: ",len(matches))
            print("Number of good matches: ",len(good_matches))
            percent = len(good_matches)/len(matches)
            print(f"Percent of good matches: {percent}")
        
        # Check if there are enough good matches
        if len(good_matches) > 10:
            src_pts = np.float32([ keypoints_src[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints_dst[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            # Tìm ma trận homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Xoay ảnh sau khi tiền xử lí theo ma trận homography
            img_src_rotated = cv2.warpPerspective(processing_img, H, (self.img_dst_shape[1], self.img_dst_shape[0]))
            return img_src_rotated
        
        else:
            print("Not enough key points")
            return self.image
        
    def scan(self):
        """
        Scan image and return image after preprocessing
        """
        #wrap image
        img_src_rotated = self.warp_perspective_image(self.image)
        self.img_src_rotated = img_src_rotated
       
        #preprocessing
        gray_image = cv2.cvtColor(img_src_rotated, cv2.COLOR_BGR2GRAY)
        self.gray_image = gray_image
        blur_image = cv2.GaussianBlur(gray_image, self.blur_size, 0)
        self.blur_image = blur_image
        
        #resize image
        self.processed_image =  blur_image       
        return self.processed_image
    
    def visualization(self):
        """
        Visualize image including original image, gray image, blur image and processing image
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 9))
        [axi.set_axis_off() for axi in axes.ravel()] # turn axis off
        
        axes[0][0].set_title('Original image')
        axes[0][0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        
        axes[0][1].set_title('Gray image')        
        axes[0][1].imshow(self.gray_image, cmap='gray')
        
        axes[1][0].set_title('Blur image')
        axes[1][0].imshow(self.blur_image, cmap='gray')

        axes[1][1].set_title('Processed image')
        axes[1][1].imshow(self.processed_image, cmap='gray')
        fig.tight_layout()
        plt.show()

# if __name__ == '__main__':
#     img_path = '/Users/macbookair/Library/CloudStorage/GoogleDrive-ttrung2h@gmail.com/My Drive/Project/Extract_Info_From_Card/Images/image1.jpg'
#     card = CardAlignment(img_path)
#     img_processed = card.scan()
#     print(img_processed.shape)
#     card.visualization()

