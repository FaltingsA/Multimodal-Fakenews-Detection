import torch 
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
import logging
from model import Multimodal_Net
import utils
import argparse
from build_rumor_datasets import load_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import os
from dataset import RumorDataset
import utils
import copy
from eval import evaluate_with_vis


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
def to_np(x):
    return x.data.cpu().numpy()

def evaluate(model, test_loader, vis_path):
        
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
                # for k,v in features_dict.items():
                #     print(k)
                #     if not isinstance(v,str): print(v.shape)

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
                # vis for batch 
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


def test(args):
        output_path = os.path.join(args.output_dir , args.datasets, args.expType,args.expCode)
        checkpoint_path = os.path.join(output_path, "checkpoint")
        # checkpoint_list = os.listdir(checkpoint_path)
        # best_model = os.path.join(checkpoint_path,checkpoint_list[-1])
        # utils.load_checkpoint(os.path.join(checkpoint_path, 'best.pth.tar'), model)
        log_path = os.path.join(output_path, "test.log")
        vis_path = os.path.join(output_path, "image")
        utils.set_logger(log_path)
        logging.info('prepare test data')
        _, test = load_data(args)
        test_dataset = RumorDataset(test)
        logging.info('Test data_length => TEXT: %d, Image: %d, label: %d, Mask: %d'
                % test_dataset.get_len())
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)
        logging.info('start testing model')
        model = Multimodal_Net(args)
        
        utils.load_checkpoint(os.path.join(checkpoint_path, 'best.pth.tar'), model)

        test_accuracy, test_f1,  test_precision, test_recall, test_aucroc,\
                 test_confusion_matrix, test_pred, test_true = evaluate_with_vis(model,test_loader,vis_path)

        logging.info("Classification Acc: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, AUC-ROC: %.4f"
                % (test_accuracy, test_f1, test_precision, test_recall, test_aucroc))
        logging.info("Classification report:\n%s\n"
                % (metrics.classification_report(test_true, test_pred)))
        logging.info("Classification confusion matrix:\n%s\n"
                % (test_confusion_matrix))

def parse_arguments(parser):
    
    parser.add_argument('--output_dir', type=str, default='../data/result/', help='')
    parser.add_argument('--datasets', type=str, default='Twitter', help='[Twitter, Weibo]')
    parser.add_argument('--mode', type=str, default='test', help='[train, test]')
    parser.add_argument('--max_seqlen', type=int, default=256, help='')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--expType', type=str, default='all', help='[vis,text,wo_fusion,mulT,all]')
    parser.add_argument('--expCode', type=str, default='all_epoch50_debug', help='describe exp hyparam setting')
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--num_epochs', type=int, default=1, help='tiny model => 50 /100, pre-train big model => 10 / 20 / 50')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')

    return parser

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    utils.setup_seed(args.seed)
    test(args)