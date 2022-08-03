
import numpy as np
import argparse
import time, os
import pandas as pd
from tqdm import tqdm
from random import sample
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from transformers import BertModel, RobertaModel
import os.path
from modules.embracenet import EmbraceNet
from modules.model_cross import MULTModel


class Multimodal_Net(nn.Module):
    def __init__(self, args):
        super(Multimodal_Net, self).__init__()
        self.args = args

        self.features_buffer = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(0.5)

        # dim size 
        self.hidden_size = args.hidden_dim

        if self.args.expType == 'vis' or self.args.expType == 'text':
            self.cls_hidden_size = self.hidden_size

        else: self.cls_hidden_size = 2 * self.hidden_size


        # model component
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # self. resnet = torchvision.models.vgg19(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.args.expType != 'vis':
            self.roberta = self.get_pretrain_lm()

            for name ,param in self.roberta.named_parameters():
                    param.requires_grad = False
        
        if self.args.expType in ['text','wo_fusion']:
            for name ,param in self.roberta.named_parameters():
                param.requires_grad = False
        else:
            unfreeze_layers = ['layer.10','layer.11','roberta.pooler','out.']
            for name ,param in self.roberta.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break

        self.fc_resnet = nn.Linear(self.resnet.fc.out_features, self.hidden_size)

        self.fc_roberta = nn.Linear(768, self.hidden_size)
        
        # Multimodal Transformer
        
        
        self.mult = MULTModel(orig_d_l = 768, orig_d_v = 2048)
    
        # EmbraceNet 
        self.embracenet = EmbraceNet(device=self.device, input_size_list=[self.hidden_size, self.hidden_size], embracement_size=2 * self.hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.cls_hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        
    def get_pretrain_lm(self):
        if self.args.datasets == 'Twitter':
            return RobertaModel.from_pretrained('roberta-base')
        else: # Weibo
            return BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

    def get_features(self,**kwargs):
        self.features_buffer = kwargs
    
    def forward(self, image, text, types, mask):
        
        # 1. only use resent 
        if self.args.expType == 'vis':
            image = self.resnet(image) # [n,1000]
            features = F.leaky_relu(self.fc_resnet(image)) # [n,hidden_size]
            self.get_features(expType = 'vis',vis_f = features)
            # features = self.class_classifier(image) 
        
        # 2. only use roberta
        elif self.args.expType == 'text':
            hs_out,cls_out= self.roberta(text, attention_mask=mask)
            features = F.leaky_relu(self.fc_roberta(cls_out))
            self.get_features(expType = 'text',text_f = features)

        # 3. resnet + roberta + concat 
        elif self.args.expType == 'wo_fusion':
            image = self.resnet(image) # [n,1000]
            image = F.leaky_relu(self.fc_resnet(image)) # [n,hidden_size]
            hs_out,cls_out= self.roberta(text, attention_mask=mask)
            text = F.leaky_relu(self.fc_roberta(cls_out))
            features = torch.cat((text, image), 1)
            self.get_features(expType = 'wo_fusion',vis_f = image, text_f = text, hs_f = features)

        # 4. resnet_f + roberta_hs + mult
        elif self.args.expType == 'mulT':
            image_features = self.backbone(image)  # [n,c,h,w]
            n, c, h, w = image_features.shape
            image_features = image_features.view(n,c,-1).permute(0,2,1)

            hs_out,cls_out= self.roberta(text, attention_mask=mask)

            l_v, v_l = self.mult(hs_out,image_features) # return l,v
            features = torch.cat((l_v, v_l), 1)
            self.get_features(expType = 'mulT',vis_f = v_l, text_f = l_v, hs_f = features)

        # 5. resnet_f + roberta_hs + mult + embrancenet
        elif self.args.expType == 'all':
            image_features = self.backbone(image)  # [n,c,h,w]
            n, c, h, w = image_features.shape
            image_features = image_features.view(n,c,-1).permute(0,2,1)
            hs_out,cls_out= self.roberta(text, attention_mask=mask)
            l_v, v_l = self.mult(hs_out,image_features) # return l,v
            features = self.embracenet([l_v, v_l], availabilities=None)
            self.get_features(expType = 'all',vis_f = v_l, text_f = l_v, hs_f = features)

        ### Fake or real
        class_output = self.class_classifier(features)

        return class_output