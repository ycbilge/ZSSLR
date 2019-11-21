'''
some of the functions are taken from https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision.
'''


import logging
import os
import torch
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
import json
from sklearn.metrics import confusion_matrix




def l2_norm(input):
    norm = input.norm(p=2, dim=1, keepdim=True)
    input_normalized = input.div(norm)
    return input_normalized

#normalized
def calculate_accuracy(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_overall_normalized_acc = np.sum(cm_normalized.diagonal()) / float(len(cm_normalized.diagonal()))

    return cm_overall_normalized_acc, cm_normalized

def load_checkpoint(checkpoint, model, optimizer=None):
	if not os.path.exists(checkpoint):
		raise("File doesn't exist {}".format(checkpoint))
	checkpoint = torch.load(checkpoint)
	model.load_state_dict(checkpoint['state_dict'])

	if optimizer:
		optimizer.load_state_dict(checkpoint['optim_dict'])

	return model




def plot_confusionmatrix(confusionmatrix, sub_list, title):
    df_cn = pd.DataFrame(confusionmatrix, index=[x for x in sub_list], columns=[x for x in sub_list])
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.rc('font', size=8)
    plt.figure(figsize=(32,16))
    sn.heatmap(df_cn, annot=True)
    plt.savefig(title, aspect='auto')
    plt.close()

def plot_chart(loss_train, loss_val, acc_train, acc_val, title, savepath):
    plt.title(title)
    plt.plot(loss_train, color='palegreen', label='Train Loss')
    plt.plot(loss_val, color='darkgreen', label='Val Loss')
    plt.plot(acc_train, color='lightsalmon', label='Train Acc')
    plt.plot(acc_val, color='r', label='Val Acc')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(savepath, aspect='auto')
    plt.close()


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_checkpoint(state, is_best, checkpoint):
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")

    if is_best:
        filepath = os.path.join(checkpoint, 'best.pth.tar')
        #shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        torch.save(state, filepath)

def save_dict_to_json(best_val_acc, best_test_acc, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {" val-accuracy " : best_val_acc,
             " test-accuracy " : best_test_acc
             }
        json.dump(d, f, indent=4)
