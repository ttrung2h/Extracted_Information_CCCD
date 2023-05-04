import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tqdm import tqdm
class TextRecognition:
    def __init__(self,infor_dict,device = 'cpu',pretrained = False):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained']=pretrained
        config['device'] = device
        self.detector = Predictor(config)
        self.infor_dict = infor_dict

    def predict(self):
        self.result = {}
        self.prob = {}
        # Detect each image and save in list
        for info,img in tqdm(self.infor_dict.items()):
            pil_image = Image.fromarray(img)
            text_pred,prob = self.detector.predict(pil_image,return_prob = True)
            self.prob[info] = prob*100
            self.result[info] = text_pred
        return self.result,self.prob
    
    def extract_to_file(self):
        with open("./result/result.txt","w") as f:
            for key,value in self.result.items():
                f.write(f"{key}:{value} \n")