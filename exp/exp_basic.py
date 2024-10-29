import torch
from models import DPQAM64_recon,DPQAM64_classifier,LSTM_recon,LSTM_classifier,Recon,Classifier
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DPQAM64_recon':DPQAM64_recon,
            'DPQAM64_classifier':DPQAM64_classifier,
            'LSTM_recon':LSTM_recon,
            'LSTM_classifier':LSTM_classifier,
            'recon':Recon,
            'classifier':Classifier,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    def _build_model(self):
        raise NotImplementedError
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    def _get_data(self,flag):
        pass
    def vali(self):
        pass
    def train(self):
        pass
    def test(self):
        pass
