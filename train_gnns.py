
import os
import yaml
import random
import logging
import datetime
import importlib

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.utils import reorder_batch
from data.sampler import create_buffer
from data.dataset import create_dataset
from data.preprocessing import Preprocessing

from training_utils import EarlyStopping, compute_class_weights
from results_utils import ResultsReport


print("Loading the configuration.")
if os.path.exists("conf.yaml"):
    config = yaml.load(open("conf.yaml", "r"), Loader=yaml.FullLoader)
else:
    raise Exception("The configuration file 'conf.yaml' is missing")


seed = config["global_seed"]
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# creating the folder where the results will be saved
results_path = os.path.join("results", config["model"]["name"])
if not os.path.exists(results_path):
    os.mkdir(results_path)
results_path = os.path.join(results_path, str(datetime.datetime.utcnow()))
os.mkdir(results_path)
with open(os.path.join(results_path, "config.yaml"), "w") as file:
    yaml.dump(config, file, default_flow_style=False, sort_keys=False)


# creating the logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s", filemode="a",
                    datefmt="%Y-%m-%d %H:%M:%S", filename=os.path.join(results_path, "report.log"))
console = logging.StreamHandler()
logging.getLogger("").addHandler(console)
formatter = logging.Formatter("%(asctime)s: %(message)s")
console.setFormatter(formatter)
console.setLevel(logging.INFO)


logging.info("creating the buffer")
create_buffer(config=config)

logging.info("fitting the pre-processor")
preprocessor = Preprocessing()
n_batch = np.random.randint(config["buffer"]["size"])
logging.info(f"loading the train batch number {n_batch}")
create_dataset(folder=os.path.join(config["buffer"]["folder"], f"train_{n_batch}"), config=config,
               preprocessing=preprocessor, fit_preprocessing=True)


logging.info("constructing the train / val datasets")
n_batch = np.random.randint(config["buffer"]["size"])
logging.info(f"loading the train batch number {n_batch}")
train_dataset = create_dataset(folder=os.path.join(config["buffer"]["folder"], f"train_{n_batch}"), config=config,
                               preprocessing=preprocessor,
                               min_num_samples_per_label=config["learning"]["min_num_samples"],
                               max_num_samples_per_label=config["learning"]["max_num_samples"])
num_features = train_dataset[0].x.shape[1]
n_batch = np.random.randint(config["buffer"]["size"])
logging.info(f"loading the valid batch number {n_batch}")
val_dataset = create_dataset(folder=os.path.join(config["buffer"]["folder"], f"val_{n_batch}"), config=config,
                             preprocessing=preprocessor,
                             min_num_samples_per_label=config["learning"]["min_num_samples"],
                             max_num_samples_per_label=config["learning"]["max_num_samples"])


logging.info("building the train / val dataloaders")
train_dataloader = DataLoader(train_dataset, batch_size=config["learning"]["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["learning"]["batch_size"], shuffle=False, drop_last=False)


logging.info("creating the model / optimizer / lr scheduler")
models_module = importlib.import_module("torch_geometric.nn.models.basic_gnn")
model_class = getattr(models_module, config["model"]["name"])
model = model_class(in_channels=num_features, out_channels=len(config["categories"]),
                    num_layers=len(config["sampling"]["num_neighbors"]) + 1, **config["model"]["args"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning"]["init_learning_rate"])
previous_learning_rate = config["learning"]["init_learning_rate"]
lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, **config["learning"]["lr_scheduler"])
early_stopping = EarlyStopping(path=results_path, **config["learning"]["early_stopping"])
class_weights = compute_class_weights(train_dataset)
class_weights /= class_weights.mean()


report = ResultsReport(path=os.path.join(results_path, "report.csv"))


try:

    logging.info("training")

    for epoch in tqdm(range(config["learning"]["max_num_epochs"]), leave=False):

        train_loss = torch.tensor(0.)
        train_size = 0
        model.train()

        for batch in train_dataloader:

            optimizer.zero_grad()

            ordered_batch = reorder_batch(batch, batch.batch)
            num_nodes_sampled = list(ordered_batch.num_nodes_sampled.numpy())
            out = model(x=ordered_batch.x, edge_index=ordered_batch.edge_index,
                        batch_size=config["learning"]["batch_size"],
                        num_sampled_nodes_per_hop=num_nodes_sampled,
                        num_sampled_edges_per_hop=list(2*np.array(num_nodes_sampled)))
            out = out[:config["learning"]["batch_size"]]

            y = ordered_batch.label
            loss = F.cross_entropy(out, y, weight=class_weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach() * torch.tensor(out.shape[0])
            train_size += out.shape[0]

        train_loss /= torch.tensor(train_size)

        model.eval()

        val_loss = torch.tensor(0.)
        val_size = 0
        val_y_pred = []
        val_y_true = []

        with torch.no_grad():

            for batch in val_dataloader:

                ordered_batch = reorder_batch(batch, batch.batch)
                num_nodes_sampled = list(ordered_batch.num_nodes_sampled.numpy())
                out = model(x=ordered_batch.x, edge_index=ordered_batch.edge_index,
                            batch_size=config["learning"]["batch_size"],  num_sampled_nodes_per_hop=num_nodes_sampled,
                            num_sampled_edges_per_hop=list(2*np.array(num_nodes_sampled)))
                out = out[:config["learning"]["batch_size"]]

                y = ordered_batch.label
                loss = F.cross_entropy(out, y, weight=class_weights)
                val_loss += loss.detach() * torch.tensor(out.shape[0])
                val_size += out.shape[0]
                _, predicted_labels = torch.max(out, dim=1)
                val_y_pred.extend(list(predicted_labels.numpy()))
                val_y_true.extend(list(y.numpy()))

        val_loss /= torch.tensor(val_size)

        report.add_scores(epoch=epoch, train_loss=float(train_loss.detach().numpy()),
                          val_loss=float(val_loss.detach().numpy()),
                          val_y_pred=val_y_pred, val_y_true=val_y_true,
                          lr=previous_learning_rate)
        report.to_csv()

        lr_scheduler.step(val_loss)
        resp = early_stopping(val_loss.detach().item(), model=model)
        if resp == "save":
            logging.debug("saving the model")
        elif resp == True:
            break

        current_learning_rate = optimizer.param_groups[0]["lr"]

        if previous_learning_rate != current_learning_rate:
            logging.debug(f"reducing the learning rate {previous_learning_rate} -> {current_learning_rate}")
            previous_learning_rate = current_learning_rate

        logging.debug(f"epoch {epoch} - train loss: {float(train_loss.detach().numpy()):.5f}, "
                      f"val loss: {float(val_loss.detach().numpy()):.5f}, "
                      f"val F1 macro score: {report.validation_macro_f1.get(epoch):.5f}")

        if ((config["learning"]["resampling_period"] is not None)
                and (epoch > 0)
                and (epoch % config["learning"]["resampling_period"] == 0)):

            if config["learning"]["reload_best"]:
                model.load_state_dict(torch.load(os.path.join(early_stopping.path, "model.pt")))

            n_batch = np.random.randint(config["buffer"]["size"])
            logging.info(f"loading the train batch number {n_batch}")
            train_dataset = create_dataset(folder=os.path.join(config["buffer"]["folder"], f"train_{n_batch}"),
                                           config=config, preprocessing=preprocessor,
                                           min_num_samples_per_label=config["learning"]["min_num_samples"],
                                           max_num_samples_per_label=config["learning"]["max_num_samples"])
            train_dataloader = DataLoader(train_dataset, batch_size=config["learning"]["batch_size"], shuffle=True,
                                          drop_last=True)

            n_batch = np.random.randint(config["buffer"]["size"])
            logging.info(f"loading the valid batch number {n_batch}")
            val_dataset = create_dataset(folder=os.path.join(config["buffer"]["folder"], f"val_{n_batch}"),
                                         config=config, preprocessing=preprocessor,
                                         min_num_samples_per_label=config["learning"]["min_num_samples"],
                                         max_num_samples_per_label=config["learning"]["max_num_samples"])
            val_dataloader = DataLoader(train_dataset, batch_size=config["learning"]["batch_size"], shuffle=False,
                                        drop_last=False)

except KeyboardInterrupt:
    pass
except Exception as e:
    raise e


logging.info("testing")

model = model_class(in_channels=num_features, out_channels=len(config["categories"]),
                    num_layers=len(config["sampling"]["num_neighbors"]) + 1, **config["model"]["args"])
model.load_state_dict(torch.load(os.path.join(early_stopping.path, "model.pt"), weights_only=True))
model.eval()


test_y_true = dict()
test_probs_pred = dict()
try:
    for n_batch in tqdm(range(config["buffer"]["size"])):
        test_dataset = create_dataset(folder=os.path.join(config["buffer"]["folder"], f"test_{n_batch}"), config=config,
                                      preprocessing=preprocessor)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
        with torch.no_grad():
            for batch in test_dataloader:
                ordered_batch = reorder_batch(batch, batch.batch)
                num_nodes_sampled = list(ordered_batch.num_nodes_sampled.numpy())
                out = model(x=ordered_batch.x, edge_index=ordered_batch.edge_index, batch_size=256,
                            num_sampled_nodes_per_hop=num_nodes_sampled,
                            num_sampled_edges_per_hop=list(2*np.array(num_nodes_sampled)))
                log_probs = out[:256]
                probs = F.softmax(log_probs, dim=1).detach().numpy()
                ys = ordered_batch.label.detach().numpy()
                seeds = ordered_batch.seed.detach().numpy()
                for p, y, seed in zip(probs, ys, seeds):
                    test_y_true[int(seed)] = y
                    test_probs_pred[int(seed)] = p + test_probs_pred.get(int(seed), np.zeros(len(config["categories"])))
except KeyboardInterrupt:
    pass
except Exception as e:
    raise e

test_y_true_list = []
test_y_pred_list = []
for seed, label in test_y_true.items():
    test_y_true_list.append(label)
    probs = test_probs_pred[seed]
    y_pred = np.argmax(probs)
    test_y_pred_list.append(y_pred)
test_accuracy = accuracy_score(test_y_true_list, test_y_pred_list)
test_f1 = f1_score(test_y_true_list, test_y_pred_list, average="macro", zero_division=0.)
logging.info(f"test accuracy {test_accuracy}, test macro f1 {test_f1}")

report = classification_report(test_y_true_list, test_y_pred_list)
print(report)


