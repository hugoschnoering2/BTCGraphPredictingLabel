
import os
import torch


class EarlyStopping:

    def __init__(self, patience: int, delta: float = 0., path=None):

        self.patience = patience
        self.delta = delta
        self.path = path

        self.counter = 0
        self.min_loss = None

    def __call__(self, loss, model=None):

        if self.min_loss is None:
            self.min_loss = loss
            if self.path and model:
                torch.save(model.state_dict(), os.path.join(self.path, "model.pt"))
                return "save"
        elif loss > self.min_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.min_loss = loss
            self.counter = 0
            if self.path and model:
                torch.save(model.state_dict(), os.path.join(self.path, "model.pt"))
                return "save"

        return False


def compute_class_weights(dataset):
    labels = [graph.label for graph in dataset]
    values_count = [0 for _ in range(max(*labels) + 1)]
    for label in labels:
        values_count[label] += 1
    total_count = sum(values_count)
    total_classes = len(values_count)
    class_weights = [total_count / (values_count[i] * total_classes) for i in range(total_classes)]
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights
