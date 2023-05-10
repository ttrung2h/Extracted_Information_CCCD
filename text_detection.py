import cv2
import numpy as np
import matplotlib.pyplot as plt
from card_alignment import CardAlignment
from tqdm import tqdm
class TextDetection:
    """
    This class is used to detect text in image
    and get position name, id, dob from card id 
    """
    def __init__(self,image):
        self.default_config = {"Id":[(1000,620),(2050,820)],"Name": [(650,860),(2350,1000)],"DOB": [(1450,980),(1950,1100)]}
        self.processed_image = image

    def show_gray_image(self,img,name = None):
        plt.imshow(img,cmap = "gray")
        if name != None:
            plt.title(name)
        plt.show()
    
    
    def crop_image(self,point1,point2):

        # Crop image and using contour to find exactly text area
        croped_img = self.processed_image[point1[1]:point2[1],point1[0]:point2[0]]
        # Check image in gray or rgb
        if len(croped_img) == 3:
            gray = cv2.cvtColor(croped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = croped_img
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min, y_min, x_max, y_max = croped_img.shape[1], 0, 0, croped_img.shape[0]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < x_min:
                x_min = x
            if x + w > x_max:
                x_max = x + w
        
        if x_min > 10:
            x_min = x_min - 10
        croped_img = croped_img[y_min:y_max,x_min:x_max]
        
        
        return croped_img
    
    def detect(self,show_process = False,extract_infor_img = False,display = False):
        """
            Create dictionary to store information
            Key is information and value is image of information
        """
        self.bouding_box_info = {}
        infor_dict = {}
        for info,pos in self.default_config.items():
            infor_dict[info] = self.crop_image(pos[0],pos[1])
            
            if extract_infor_img:
                cv2.imwrite(f"croped_{info}.jpg",infor_dict[info])
        
        # Show image with bounding box
        if display:
            img = self.processed_image.copy()
            for info,pos in self.default_config.items():
                
                cv2.rectangle(img, pos[0], pos[1], color = (255,0,0), thickness = 2)
            
            plt.imshow(img,cmap='gray')
            plt.title("Detected infor in image")
            plt.show()
        
        #Show each croped image
        if show_process:
            fig, axes = plt.subplots(nrows=1, ncols=3)
            index = 0
            for key,value in infor_dict.items():
                axes[index].imshow(value, cmap='gray')
                axes[index].set_xticks([])
                axes[index].set_yticks([])
                axes[index].set_title(key)
                index+=1
            plt.show()
        
        return infor_dict

# if __name__ == '__main__':
#     img_path = "/Users/macbookair/Library/CloudStorage/GoogleDrive-ttrung2h@gmail.com/My Drive/Project/Extract_Info_From_Card/Images/image1.jpg"
#     scan_img = CardAlignment(img_path)
#     img_processed = scan_img.scan()
#     text_detection = TextDetection(img_processed)
#     # text_detection.show_gray_image(img_processed)
#     infor_dict = text_detection.detect()
#     text_detection.show_detected_img()