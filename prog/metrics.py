""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

class SmoothMeter:
    """Computes and stores the average and current value"""
    def __init__(self, length=50):
        self.length = length
        self.reset()

    def reset(self):
        self.val_list = []
        self.val_registor = []
        self.val = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val_registor.append(val)
        # self.val_list = self.val_list + [val] * n
        self.val_list = self.val_list + [val]
        self.val_list = self.val_list[-self.length:]
        self.val = val
        self.sum = sum(self.val_list)
        self.avg = self.sum / len(self.val_list)






class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]