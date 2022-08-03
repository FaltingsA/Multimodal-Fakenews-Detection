import torch 

import numpy as np
from sklearn import metrics
import utils
import warnings
warnings.filterwarnings('ignore')
from utils import to_np, to_var



def evaluate(model,test_loader):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_data, test_labels, test_types) in enumerate(test_loader):
        test_text, test_image, test_mask, test_labels, test_types = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels), to_var(test_types)
        test_outputs = model(test_image, test_text, test_types, test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    return test_accuracy, test_f1,  test_precision, test_recall, test_aucroc, test_confusion_matrix, test_pred, test_true

def evaluate_with_vis(model, test_loader, vis_path):
        
        vis_tool = utils.VisulTool()
        if torch.cuda.is_available():
                model.cuda()
        model.eval()
        test_score = []
        test_pred = []
        test_true = []
        scatter_image_list, scatter_text_list, scatter_fuse_list = [], [], []
        for i, (test_data, test_labels, test_types) in enumerate(test_loader):
                
                test_text, test_image, test_mask, test_labels, test_types = to_var(
                test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels), to_var(test_types)
                test_outputs = model(test_image, test_text, test_types, test_mask)
                _, test_argmax = torch.max(test_outputs, 1)
                if i == 0:
                        test_score = to_np(test_outputs.squeeze())
                        test_pred = to_np(test_argmax.squeeze())
                        test_true = to_np(test_labels.squeeze())
                else:
                        test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
                        test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                        test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

                #Vis for batch
                features_dict = model.features_buffer
                
                if features_dict['expType'] == 'vis':
                        scatter_image = to_np(torch.flatten(features_dict['vis_f'], start_dim=1))
                        scatter_text = None
                        scatter_fuse = None
                        scatter_image_list.append(scatter_image)
                elif features_dict['expType'] == 'text':
                        scatter_text = to_np(torch.flatten(features_dict['text_f'], start_dim=1))
                        scatter_text_list.append(scatter_text)
                        scatter_image = None
                        scatter_fuse = None
                else:
                        scatter_image = to_np(torch.flatten(features_dict['vis_f'], start_dim=1))
                        scatter_text = to_np(torch.flatten(features_dict['text_f'], start_dim=1))
                        scatter_fuse = to_np(torch.flatten(features_dict['hs_f'], start_dim=1))
                        scatter_image_list.append(scatter_image)
                        scatter_text_list.append(scatter_text)
                        scatter_fuse_list.append(scatter_fuse)
                
                vis_tool.vis_scatter_batch(scatter_image, scatter_text, scatter_fuse, \
                         to_np(test_labels.squeeze()),i,vis_path)
                         
        # Vis for epoch
        if len(scatter_image_list): scatter_image_epoch = np.concatenate(scatter_image_list,axis = 0) 
        else: scatter_image_epoch = None
        if len(scatter_text_list):scatter_text_epoch = np.concatenate(scatter_text_list,axis = 0)
        else: scatter_text_epoch = None
        if len(scatter_fuse_list):scatter_fuse_epoch = np.concatenate(scatter_fuse_list,axis = 0) 
        else: scatter_fuse_epoch = None

        vis_tool.vis_scatter_epoch(scatter_image_epoch, scatter_text_epoch, scatter_fuse_epoch, \
                         test_true,vis_path)

        test_accuracy = metrics.accuracy_score(test_true, test_pred)
        test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
        test_precision = metrics.precision_score(test_true, test_pred, average='macro')
        test_recall = metrics.recall_score(test_true, test_pred, average='macro')
        test_score_convert = [x[1] for x in test_score]
        test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
        test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

        return test_accuracy, test_f1,  test_precision, test_recall, test_aucroc, test_confusion_matrix, test_pred, test_true
