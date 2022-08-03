from email.policy import default
import logging
import numpy as np
import argparse
import  os


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import os.path
from model import Multimodal_Net
from dataset import RumorDataset
from build_rumor_datasets import load_data
import utils
from utils import to_np, to_var
from eval import evaluate
import wandb
import logging
wandb.init(project="test-project", entity="imcc-ocr")


def prepare_data(args):
    logging.info('loading data')

    train, test = load_data(args)

    train_dataset = RumorDataset(train)
    logging.info('Train data_length => TEXT: %d, Image: %d, label: %d, Mask: %d'
               % train_dataset.get_len())
    test_dataset = RumorDataset(test)
    logging.info('Test data_length => TEXT: %d, Image: %d, label: %d, Mask: %d'
               % test_dataset.get_len())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader

def main(args):

    output_path = os.path.join(args.output_dir , args.datasets, args.expType,args.expCode)

    if  os.path.exists(output_path): 
        cmd = 'rm -rf {} \n'.format(output_path)
        os.system(cmd)

    os.makedirs(output_path) 

    # log path
    log_path = os.path.join(output_path, "train.log")
    vis_path = os.path.join(output_path, "image")
    metric_path = os.path.join(output_path, "metric")
    checkpoint_path = os.path.join(output_path, "checkpoint")
    if not os.path.exists(vis_path): os.mkdir(vis_path)
    if not os.path.exists(metric_path): os.mkdir(metric_path)
    if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)

    utils.set_logger(log_path)

    logging.info('loading data')

    train, test = load_data(args)

    train_dataset = RumorDataset(train)

    logging.info('Train data_length => TEXT: %d, Image: %d, label: %d, Mask: %d'
               % train_dataset.get_len())
    test_dataset = RumorDataset(test)
    logging.info('Test data_length => TEXT: %d, Image: %d, label: %d, Mask: %d'
               % test_dataset.get_len())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    logging.info('building model')
    model = Multimodal_Net(args)
    logging.info("expType: %s, train in expCode: %s, batch_size %s, epoch = %s, lr = %s" \
        % (args.expType, args.expCode, args.batch_size, args.num_epochs, args.learning_rate))

    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, list(model.parameters())),
    # lr=args.learning_rate)
    # scheduler = StepLR(optimizer, step_size= 10, gamma= 1)

    best_validate_acc = 0.000
    logging.info('start training model')
    
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        # lr = args.learning_rate / (1. + 10 * p) ** 0.75
        lr = args.learning_rate
        optimizer.lr = lr
        # rgs.lambd = lambd
        # start_time = time.time()
        train_cost_vector = []

        train_true, train_pred, train_score = [], [], []
        # in one batch 
        for i, (train_data, train_labels, train_types) in enumerate(train_loader):
            train_text, train_image,  train_mask, train_labels, train_types = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(train_types )

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs = model(train_image, train_text, train_types, train_mask)

            ## Fake or Real loss
            class_loss = criterion(class_outputs, train_labels)
            # Event Loss
            loss = class_loss
            wandb.log({"train_loss": loss})
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)
            if i == 0:
                train_score = to_np(class_outputs.squeeze())
                train_pred = to_np(argmax.squeeze())
                train_true = to_np(train_labels.squeeze())
            else:
                train_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
                train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
                train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)
           
            # accuracy = (train_labels == argmax.squeeze()).float().mean()

            train_cost_vector.append(loss.item())  # loss.data[0]
            # acc_vector.append(accuracy.item())  # loss.data[0]
        
        train_acc = metrics.accuracy_score(train_true, train_pred)

        # do eval for one batch 
        model.eval()
        validate_score = []
        validate_pred = []
        validate_true = []
        val_cost_vector = []
        for i, (validate_data, validate_labels, validate_types) in enumerate(test_loader):
            validate_text, validate_image, validate_mask, validate_labels, validate_types = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(validate_types)

            validate_outputs = model(validate_image, validate_text, validate_types, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            wandb.log({"val_loss": vali_loss})
            val_cost_vector.append(vali_loss.item())  # vali_loss.data[0]
            if i == 0:
                validate_score = to_np(validate_outputs.squeeze())
                validate_pred = to_np(validate_argmax.squeeze())
                validate_true = to_np(validate_labels.squeeze())
            else:
                validate_score = np.concatenate((validate_score, to_np(validate_outputs.squeeze())), axis=0)
                validate_pred = np.concatenate((validate_pred, to_np(validate_argmax.squeeze())), axis=0)
                validate_true = np.concatenate((validate_true, to_np(validate_labels.squeeze())), axis=0)
            
        validate_acc = metrics.accuracy_score(validate_true, validate_pred)
        validate_f1 = metrics.f1_score(validate_true, validate_pred, average='macro')
        validate_precision = metrics.precision_score(validate_true, validate_pred, average='macro')
        validate_recall = metrics.recall_score(validate_true, validate_pred, average='macro')
        
        # reset model to train mode
        model.train()

        # logging & record for one epoch
        wandb.log({"Train_Acc":train_acc, "Val_Acc":validate_acc})
        
        logging.info('Epoch [%d/%d],  epoch_train_loss: %.4f, epoch_val_loss: %.4f, Train_Acc: %.4f,  Val_Acc: %.4f, Val_F1: %.4f, Val_Pre: %.4f, Val_Rec: %.4f.'
              % (epoch + 1, args.num_epochs, np.mean(train_cost_vector), np.mean(val_cost_vector),train_acc,\
                   validate_acc, validate_f1, validate_precision, validate_recall))

        if validate_acc > best_validate_acc:
            is_best = validate_acc > best_validate_acc
            best_validate_acc = validate_acc
            
            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict': optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=checkpoint_path)
           
    # Test the Model
    logging.info('start testing model')
    model = Multimodal_Net(args)
    # loda best model weights
    utils.load_checkpoint(os.path.join(checkpoint_path, 'best.pth.tar'), model, optimizer)
    
    test_accuracy, test_f1,  test_precision, test_recall, test_aucroc, test_confusion_matrix, test_pred, test_true = evaluate(model,test_loader)
    logging.info("showing the cache model metrics")
    logging.info("Classification Acc: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_f1, test_precision, test_recall, test_aucroc))
    logging.info("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    logging.info("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))




def parse_arguments(parser):
    
    parser.add_argument('--output_dir', type=str, default='../data/result/', help='')
    parser.add_argument('--datasets', type=str, default='Twitter', help='[Twitter, Weibo]') # datasets
    parser.add_argument('--mode', type=str, default='train', help='[train, test]')
    parser.add_argument('--max_seqlen', type=int, default=256, help='')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--expType', type=str, default='all', help='[vis,text,wo_fusion,mulT,all]') #  type
    parser.add_argument('--expCode', type=str, default='all_epoch50_debug', help='describe exp hyparam setting') # path 
    parser.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default = 32, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--lambd', type=int, default= 1, help='')
    parser.add_argument('--text_only', type=bool, default= False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--num_epochs', type=int, default=50, help='tiny model => 50 /100, pre-train big model => 10 / 20 / 50')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')

    return parser

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    args = parser.parse_args()
    utils.setup_seed(args.seed)
    main(args)

