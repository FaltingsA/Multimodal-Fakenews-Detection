from collections import OrderedDict
import random
from torch import nn
import json
import logging
import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from torch.autograd import Variable

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
def to_np(x):
    return x.data.cpu().numpy()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s %(filename)s]: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(asctime)s %(filename)s]: %(message)s')) #'[%(asctime)s %(filename)s:%(lineno)d %(levelname)s]: %(message)s'
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    
    Example:
    fea_hooks = get_feas_by_hook(model) # 调用函数，完成注册即可

    x  = torch.randn([32, 3, 224, 224])
    out = model(x)
    print('The number of hooks is:', len(fea_hooks)
    print('The shape of the first Conv2D feature is:', fea_hooks[0].fea.shape)

    """
    fea_hooks = []
    for n, m in model.named_modules():
        print (n,m)
        # if isinstance(m, torch.nn.Conv2d):
        #     cur_hook = HookTool()
        #     m.register_forward_hook(cur_hook.hook_fun)
        #     fea_hooks.append(cur_hook)

    return fea_hooks




class VisulTool(object):
    def __init__(self, env='default', dim =2):
        self.tsne = TSNE(n_components=dim)


    def vis_scatter_batch(self,image,text,fuse,label,batch_idx,vis_dir):

        vis_path = os.path.join(vis_dir,'scatter_label')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path) 
        
        if  image is not None:
            image_tsne = self.tsne.fit_transform(image)  #image图像降至一维
            visual = np.vstack((image_tsne.T,label)).T
            self.plot_feature_scatter(visual,vis_path,'visual_batch_{}'.format(batch_idx))

        if  text is not None:
            text_tsne = self.tsne.fit_transform(text) 
            textual = np.vstack((text_tsne.T,label)).T
            self.plot_feature_scatter(textual,vis_path,'textual_batch_{}'.format(batch_idx))

        if  fuse is not None:
            fuse_tsne = self.tsne.fit_transform(fuse) 
            fusetual = np.vstack((fuse_tsne.T,label)).T
            self.plot_feature_scatter(fusetual,vis_path,'fusetual_batch_{}'.format(batch_idx))
        
    
    def vis_scatter_epoch(self,image,text,fuse,label, vis_dir):
        vis_path = os.path.join(vis_dir,'scatter_label')
        if not os.path.exists(vis_path):   
            os.makedirs(vis_path) 
        if  image is not None:
            image_tsne = self.tsne.fit_transform(image)  #image图像降至一维
            visual = np.vstack((image_tsne.T,label)).T
            self.plot_feature_scatter(visual,vis_path,'visual_all')

        if  text is not None:
            text_tsne = self.tsne.fit_transform(text) 
            textual = np.vstack((text_tsne.T,label)).T
            self.plot_feature_scatter(textual,vis_path,'textual_all')

        if  fuse is not None:
            fuse_tsne = self.tsne.fit_transform(fuse) 
            fusetual = np.vstack((fuse_tsne.T,label)).T
            self.plot_feature_scatter(fusetual,vis_path,'fusetual_all')


    def plot_feature_scatter(self, data, vis_path,pattern):
        col = ['Dim1', 'Dim2', 'class']
        data = pd.DataFrame(data, columns=col)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=data, hue=col[-1], x=col[0], y=col[1])  #画散点图
        plt.savefig(os.path.join(vis_path,pattern+'.jpg'))



# another way to get feature map of backbone
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
 
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
 
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
 
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
