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
        self.default_config = {"Id":[(1010,650),(2020,820)],"Name": [(650,870),(2200,1000)],"DOB": [(1450,980),(1950,1100)]}
        self.processed_image = image

    def show_gray_image(self,img,name = None):
        plt.imshow(img,cmap = "gray")
        if name != None:
            plt.title(name)
        plt.show()
    
    def crop_image(self,point1,point2,show_process = False):
        croped_img = self.processed_image[point1[1]:point2[1],point1[0]:point2[0]]
        
        if show_process:
            self.show_gray_image(croped_img)
        return croped_img
    
    def show_detected_img(self):
        # Draw the point on the image
        img = self.processed_image
        
        for info,pos in tqdm(self.default_config.items()):
            cv2.rectangle(img, pos[0], pos[1], color = (255,0,0), thickness = 2)
        
        plt.imshow(img,cmap='gray')
        plt.title("Detected infor in image")
        plt.show()
    
    def detect(self,show_process = False,extract_infor_img = False):
        """
            Create dictionary to store information
            Key is information and value is image of information
        """
        infor_dict = {}
        for info,pos in tqdm(self.default_config.items()):
            infor_dict[info] = self.crop_image(pos[0],pos[1],show_process=show_process)
            
            if extract_infor_img:
                cv2.imwrite(f"croped_{info}.jpg",infor_dict[info])

        return infor_dict

# if __name__ == '__main__':
#     img_path = "/Users/macbookair/Library/CloudStorage/GoogleDrive-ttrung2h@gmail.com/My Drive/Project/Extract_Info_From_Card/Images/image1.jpg"
#     scan_img = CardAlignment(img_path)
#     img_processed = scan_img.scan()
#     text_detection = TextDetection(img_processed)
#     # text_detection.show_gray_image(img_processed)
#     infor_dict = text_detection.detect()
#     text_detection.show_detected_img()