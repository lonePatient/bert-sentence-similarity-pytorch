#encoding:utf-8
import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
class Accuracy(object):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

class ClassReport(object):
    def __init__(self,target_names = None):
        self.target_names = target_names

    def __call__(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        classify_report = classification_report(y_true, y_pred,target_names=self.target_names)
        print('\n\nclassify_report:\n', classify_report)

class F1Score(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = None
        self.targets = None

    def _compute(self,output,target):
        f1 =f1_score(y_true = target, y_pred=output)
        return f1

    def result(self):
        f1 = self._compute(self.outputs, self.targets)
        print("F1 Score: %.4f"%(f1))

    def update(self,output,target):
        _, y_pred = torch.max(output.data, 1)
        y_pred = y_pred.cpu().numpy()
        y_true = target.cpu().numpy()
        if self.outputs is None:
            self.outputs = y_pred
            self.targets = y_true
        else:
            self.outputs = np.concatenate((self.outputs,y_pred),axis =0)
            self.targets = np.concatenate((self.targets, y_true), axis=0)