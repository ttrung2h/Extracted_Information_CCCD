import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
class TextRecognition:
    def __init__(self,infor_dict,device = 'cpu',pretrained = False):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']=pretrained
        config['device'] = device
        self.detector = Predictor(config)
        self.infor_dict = infor_dict

    def remove_not_digit(self,string):
        return ''.join(ch for ch in string if ch.isdigit())
    
    def format_date(self,date_predict):
        
        if len(date_predict) >= 8:
            date = self.remove_not_digit(date_predict)
            day,month,year = date[:2],date[2:4],date[4:]
            return f"{day}/{month}/{year}"    
        else:
            return date_predict   

    def format_id(self,string):
        return self.remove_not_digit(string)
    
    
    def predict(self):
        self.result = {}
        self.prob = {}
        # Detect each image and save in list
        for info,img in self.infor_dict.items():
            pil_image = Image.fromarray(img)
            text_pred,prob = self.detector.predict(pil_image,return_prob = True)
            ## Format output

            #Format id
            if info == "Id":
                text_pred = self.format_id(text_pred)

            #Format dob
            if info == "DOB":
                text_pred = self.format_date(text_pred)

            self.prob[info] = round(prob*100,2)
            
            if self.prob[info] < 60:
                text_pred = None
            
            self.result[info] = text_pred

        # Get only length id more than 5
        if self.result["Id"]!= None and len(self.result["Id"]) <5:
            self.result["Id"] = None
                    
        return self.result,self.prob
    
    def extract_to_file(self):
        with open("./result/result.txt","w") as f:
            for key,value in self.result.items():
                f.write(f"{key}:{value} \n")