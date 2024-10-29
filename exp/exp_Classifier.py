import os
import time
import warnings
import numpy as np

from data_provider.Dataset import Classifier_Dataset
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,loss_accuracy_f1_curve,score_my,confusion_maxtrix_graph
import torch
import torch.nn as nn
import torch.multiprocessing
from torch import optim
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)

import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties

from sklearn.utils import shuffle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d


class Exp_Anomaly_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Classification, self).__init__(args)

    def _build_model(self):
        # if self.args.model == 'DPQAM64Classifier':
        #     dpqam64_pretrained_model = DPQAM64(self.args)
        #     checkpoint_path = '/kaggle/input/dp64qam_abnormaldetection/pytorch/default/1/checkpoint.pth'
        #     dpqam64_pretrained_model.load_state_dict(torch.load(checkpoint_path))
        #     dpqam64_pretrained_model.eval()
        #
        #     Transformer_model_time = dpqam64_pretrained_model.Transformer_model_time
        #     Reduce_model_time = dpqam64_pretrained_model.Reduce_model_time
        #     Transformer_model_freq = dpqam64_pretrained_model.Transformer_model_freq
        #     Reduce_model_freq = dpqam64_pretrained_model.Reduce_model_freq
        #     model = DPQAM64Classifier(self.args, Transformer_model_time, Reduce_model_time, Transformer_model_freq,
        #                               Reduce_model_freq).float()

        model = self.model_dict[self.args.model](self.args).float()
        return model


        pass
    def _get_data(self, flag):
        shuffle_flag = False
        drop_last = True
        data_set = Classifier_Dataset(self.args, flag, quickLoad=True, ratio=0.6)
        data_loader = DataLoader(data_set,
                                 batch_size=self.args.batch_size,
                                 shuffle=shuffle_flag,
                                 drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def F1_vali(self,vali_true,vali_pred ,test_true,test_pred):
        true = np.append(vali_true,test_true,axis=0)
        pred = np.append(vali_pred, test_pred, axis=0)
        return score_my(pred,true)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.delta)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_loss_history = []  # 记录训练损失
        vali_loss_history = []  # 记录验证损失
        test_loss_history = []  # 记录测试损失
        F1_score_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs.float(), batch_y.long())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss,vali_true,vali_pred = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_true,test_pred = self.vali(test_data, test_loader, criterion)

            F1_score = self.F1_vali(vali_true,vali_pred ,test_true,test_pred)

            train_loss_history.append(train_loss)
            vali_loss_history.append(vali_loss)
            test_loss_history.append(test_loss)
            F1_score_history.append(F1_score)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        loss_accuracy_f1_curve(train_loss_history,vali_loss_history,test_loss_history,F1_score_history)
        np.save("training_histories_classifier.npy", {
            "train_loss_history": train_loss_history,
            "vali_loss_history": vali_loss_history,
            "test_loss_history": test_loss_history,
            "F1_score_history": F1_score_history
        })
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        pred = []
        true = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x).detach().cpu()
                loss = criterion(outputs.float(), batch_y.long())
                total_loss.append(loss)
                true.append(batch_y.long())
                pred.append(outputs)
        total_loss = np.average(total_loss)
        self.model.train()
        pred = np.array(pred).reshape(-1)
        true = np.array(true).reshape(-1)
        return total_loss,true,pred

    def test(self):
        _, test_loader = self._get_data(flag='test')
        _, vali_loader = self._get_data(flag='valid')
        criterion = self._select_criterion()
        test_labels = []
        pred_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                pred_labels.append(torch.argmax(outputs, dim=-1).detach().cpu())
                test_labels.append(batch_y.detach().cpu())
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                pred_labels.append(torch.argmax(outputs, dim=-1).detach().cpu())
                test_labels.append(batch_y.detach().cpu())

        pred_labels = np.array(pred_labels).reshape(-1)
        test_labels = np.array(test_labels).reshape(-1)
        _ = score_my(test_labels, pred_labels, digits=4)
        confusion_maxtrix_graph(test_labels, pred_labels)

