import os
import time
import warnings
import numpy as np
import gc

from data_provider.Dataset import DP64QAM_Dataset
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,loss_accuracy_f1_curve,score_my,distribution_scatter,plot_energy_distribution,visualize

import torch
import torch.nn as nn
import torch.multiprocessing
from torch import optim
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        shuffle_flag = False
        drop_last = True
        data_set = DP64QAM_Dataset(self.args, flag, quickLoad=True, ratio=1)
        data_loader = DataLoader(data_set,
                                 batch_size=self.args.batch_size,
                                 shuffle=shuffle_flag,
                                 drop_last=drop_last)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        true = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(batch_x, outputs) # bs,16384,2
                total_loss.append(loss.detach().cpu())
                true.append(batch_y)
        total_loss = np.array(total_loss)
        true = np.array(true)
        self.model.train()
        return total_loss,true

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists("./graph_result"):
            os.makedirs("./graph_result")

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=self.args.delta)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_loss_history = []
        vali_loss_history = []
        test_loss_history = []
        F1_score_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} ".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss,vali_label = self.vali(vali_data, vali_loader, criterion)
            test_loss,test_label = self.vali(test_data, test_loader, criterion)
            F1_score = self.F1_vali(vali_loss,vali_label[:,0], test_loss,test_label[:,0],self.args.ratio*train_loss)
            
            vali_loss = np.average(vali_loss)
            test_loss = np.average(test_loss)

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

        np.save("metrics_histories.npy", {
            "train_loss_history": train_loss_history,
            "vali_loss_history": vali_loss_history,
            "test_loss_history": test_loss_history,
            "F1_score_history": F1_score_history
        })
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.threshold = train_loss_history[-1] * self.args.ratio
        # return self.model

    def test(self, setting, load_model=None):
        criterion = nn.MSELoss(reduce=False)
        valid_labels = []
        test_labels = []
        torch.cuda.empty_cache()
        gc.collect()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='valid')
        test_data, test_loader = self._get_data(flag='test')

        if load_model:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()

        attens_energy = []
        with torch.no_grad():
            for i, (batch_x,_) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                score = torch.mean(criterion(batch_x, outputs), dim=(1, 2)).detach().cpu().numpy()  # [bs,]
                attens_energy.append(score)
        train_energy = np.array(attens_energy).reshape(-1)
        threshold = self.args.ratio * np.mean(train_energy)
        print(f"threshold in test is : {threshold:.5f}")

        attens_energy = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                score = torch.mean(criterion(batch_x, outputs), dim=(1, 2)).detach().cpu().numpy()  # [bs,]
                attens_energy.append(score)
                valid_labels.append(batch_y)
        valid_energy = np.array(attens_energy).reshape(-1)

        attens_energy = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            outputs = self.model(batch_x)
            score = torch.mean(criterion(batch_x, outputs), dim=(1, 2)).detach().cpu().numpy()  # [bs,]
            attens_energy.append(score)
            test_labels.append(batch_y)
        test_energy = np.array(attens_energy).reshape(-1)

        test_labels = np.array(test_labels).reshape(-1)
        valid_labels = np.array(valid_labels).reshape(-1)

        valid_pred = (valid_energy > threshold).astype(int)  # test > threshold ,标记为异常
        valid_gt = valid_labels.astype(int)
        test_pred = (test_energy > threshold).astype(int)  # test > threshold ,标记为异常
        test_gt = test_labels.astype(int)

        pred = np.append(valid_pred, test_pred)
        gt = np.append(valid_gt, test_gt)
        energy = np.append(valid_energy,test_energy)

        visualize(vali_data,self.model, batch_size=self.args.batch_size, num=self.args.out_len, itr=2,save_name='vali')
        visualize(test_data,self.model, batch_size=self.args.batch_size, num=self.args.out_len, itr=2,save_name='test')

        score_my(gt,pred,4,True)
        distribution_scatter(energy,gt,pred,threshold)
        plot_energy_distribution(valid_energy,test_energy,threshold)


    def F1_vali(self,vali_loss, vali_label, test_loss, test_label, threshold):
        total_loss = np.append(vali_loss, test_loss, axis=0)
        label = np.append(vali_label, test_label, axis=0)
        y_pred = (np.array(total_loss).flatten() > threshold).astype(int)  # test > threshold ,标记为异常
        y_true = np.array(label).flatten().astype(int)
        return score_my(y_true, y_pred)
