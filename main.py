from card_alignment import CardAlignment
from text_detection import TextDetection
from text_recognition import TextRecognition
from time import time
import os
from tqdm import tqdm
import json
import re
from warnings import filterwarnings
filterwarnings("ignore")

def show_result(result):
    """Show the result of text recognition"""
    for key, value in result.items():
        print(f"{key}:{value}")

def get_result(img_path):
    ## Check how long it takes to run the program
    start_aligment = time()
    ## card alignment
    scan_img = CardAlignment(img_path)
    img_processed = scan_img.scan()
    end_aligment = time()
    # print("Time to run algment image: ",end_aligment - start_aligment)
    
    ## text detection
    start_detection = time()
    text_detection = TextDetection(img_processed)
    infor_dict = text_detection.detect(show_process=True,extract_infor_img=True,display=True)
    end_detection = time()
    # print("Time to run detection text: ",end_detection - start_detection)
   
    
    ## text recognition
    start_recognition = time()
    text_recognition = TextRecognition(infor_dict)
    result = text_recognition.predict()
    end_recognition = time()
    # print("Time to run recognition text: ",end_recognition - start_recognition)
    # print("Time to run the program: ",end_recognition - start_aligment)
    text_recognition.extract_to_file()
    return result


def test_auto(img_folder = "Test/Images/",label_folder = "Test/Labels/"):
    list_images = sorted(os.listdir(img_folder),key = lambda x : re.findall(r'\d+', x))
    list_labels= sorted(os.listdir(label_folder),key = lambda x : re.findall(r'\d+', x))
    #reset log file
    with open("log_test.txt", 'w') as file:
        file.write('')
    wrong_cases = 0
    for i in tqdm(range(len(list_images))):
        img = img_folder+str(list_images[i])
        label = label_folder+str(list_labels[i])

        predict,prob = get_result(img)
        with open(label) as f:
            true_label = json.load(f)
        
        #write log
        with open("log_test.txt","a") as result:
            result.write('-----------------------------------'+'\n')
            result.write('-'+list_images[i]+'\n')

            for key,value in predict.items():
                if value != true_label[key]:
                    result.write("\t"+"Predict : "+str(value)+"| True : "+str(true_label[key])+'\n')
                    wrong_cases+=1

    with open("log_test.txt","a") as result:       
        result.write(f"Total wrong case {wrong_cases}"+"\n")



if __name__ == '__main__':
    result,prob = get_result("Test/Images/image43.jpg")
    show_result(result)
    # test_auto()
    