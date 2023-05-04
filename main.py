from card_alignment import CardAlignment
from text_detection import TextDetection
from text_recognition import TextRecognition
from time import time
import os
from warnings import filterwarnings
import cv2
filterwarnings("ignore")

def show_result(result):
    """Show the result of text recognition"""
    for key, value in result.items():
        print(f"{key}:{value}")


def main():
    index = 1
    for img in os.listdir("Images"):
        img_path = f"Images/{img}"
        if img_path == "Images/.DS_Store":
            continue
        print(f"Image {index}: {img_path}")
        scan_img = CardAlignment(img_path)
        img_processed,percent = scan_img.scan()
        save_path = f"Images_Card_AfterProcessing/processed_{index}.jpg"
        cv2.imwrite(save_path,img_processed)
        index += 1
    
    # ## Check how long it takes to run the program
    # start_aligment = time()
    # ## card alignment
    # img_path = "Images/image21.jpg"
    # scan_img = CardAlignment(img_path)
    # img_processed,percent = scan_img.scan()
    # end_aligment = time()
    # print("Time to run algment image: ",end_aligment - start_aligment)
    
    # ## text detection
    # start_detection = time()
    # text_detection = TextDetection(img_processed)
    # infor_dict = text_detection.detect(show_process=False,extract_infor_img=True)
    # text_detection.show_detected_img()
    # end_detection = time()
    # print("Time to run detection text: ",end_detection - start_detection)
   
    
    # ## text recognition
    # start_recognition = time()
    # text_recognition = TextRecognition(infor_dict)
    # result,prob = text_recognition.predict()
    # end_recognition = time()
    # print("Time to run recognition text: ",end_recognition - start_recognition)
    # print("Time to run the program: ",end_recognition - start_aligment)
    # text_recognition.extract_to_file()
    # show_result(result)
    # show_result(prob)


if __name__ == '__main__':
   main()