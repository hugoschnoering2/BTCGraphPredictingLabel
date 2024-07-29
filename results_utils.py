
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class ResultsReport:

    def __init__(self, path: str = None):

        self.training_loss = dict()
        self.validation_loss = dict()
        self.validation_accuracy = dict()
        self.validation_balanced_accuracy = dict()
        self.validation_macro_f1 = dict()
        self.lrs = dict()

        self.path = path

    def add_scores(self, epoch: int, train_loss: float, val_loss: float, val_y_true: list, val_y_pred: list, lr: float):
        self.training_loss[epoch] = train_loss
        self.validation_loss[epoch] = val_loss
        self.validation_accuracy[epoch] = accuracy_score(y_true=val_y_true, y_pred=val_y_pred)
        self.validation_balanced_accuracy[epoch] = balanced_accuracy_score(y_true=val_y_true, y_pred=val_y_pred)
        self.validation_macro_f1[epoch] = f1_score(y_true=val_y_true, y_pred=val_y_pred, average="macro",
                                                   zero_division=0.)
        self.lrs[epoch] = lr

    def to_csv(self):
        if self.path:
            df = pd.concat([
                pd.Series(self.training_loss, name="training_loss"),
                pd.Series(self.validation_loss, name="validation_loss"),
                pd.Series(self.validation_accuracy, name="validation_accuracy"),
                pd.Series(self.validation_balanced_accuracy, name="validation_balanced_accuracy"),
                pd.Series(self.validation_macro_f1, name="validation_macro_f1"),
                pd.Series(self.lrs, name="learning_rates"),
            ], axis=1)
            df.to_csv(self.path)



