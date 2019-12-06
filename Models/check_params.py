import models

const = models.TreeLSTMClassifier(50, 50, 50, 'constituency')
dep = models.TreeLSTMClassifier(50, 50, 50, 'dependency')
hybrid = models.TreeLSTMClassifier(50, 50, 50, 'hybrid')
bidir = models.BidirectionalLSTM(50, 50, 50)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(const))
print(count_parameters(dep))
print(count_parameters(hybrid))
print(count_parameters(bidir))
