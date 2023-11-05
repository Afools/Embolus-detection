import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torchvision import models
from network_files import FasterRCNN
from backbone.resnet50_fpn_model import resnet50_fpn_backbone

class FPNrcnn(object):
    def __init__(self,model_path,num_classes=2,batch_size=64,num_workers=0):
        self.batch_size=batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.transfrom = transforms.Compose([transforms.ToTensor()])

        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        self.model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights_dict = torch.load(model_path, map_location='cpu')
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        self.model.load_state_dict(weights_dict)
        self.model.to(self.device)

    def predict(self,data):
        category_index={'1':'tumor'}
        img=self.transfrom(np.array(data.convert('RGB')))
        img=torch.unsqueeze(img,0)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(img.to(self.device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            predict_boxes = predict_boxes[predict_scores >= 0.8].astype(np.int32)
            predict_classes = predict_classes[predict_scores >= 0.8].astype(np.int32)
            predict_scores = predict_scores[predict_scores >= 0.8]
            predict_boxes = predict_boxes.tolist()
            predict_scores = predict_scores.tolist()
            return [predict_box + [predict_score] for predict_box, predict_score in zip(predict_boxes, predict_scores)]
        