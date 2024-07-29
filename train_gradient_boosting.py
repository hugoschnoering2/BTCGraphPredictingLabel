
import os
import yaml
import pickle
import datetime
import pandas as pd
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

from data.postgresql import PostgresqlDataService as DataService, Condition
from data.dataset import NodeFeaturesExtractor
from data.utils import train_test_split
from data.preprocessing import Preprocessing


print("Loading the configuration.")
if os.path.exists("conf.yaml"):
    config = yaml.load(open("conf.yaml", "r"), Loader=yaml.FullLoader)
else:
    raise Exception("The configuration file 'conf.yaml' is missing")


print("Fetching the alias of the labeled nodes.")
labelled_nodes = DataService(**config["db"]).fetch(
    table="node_features", columns=["alias", "label"],
    conditions=[Condition("label", "IN", config["categories"])], limit=None)
labelled_nodes = {int(row["alias"]): row["label"] for row in labelled_nodes}
train_nodes, val_nodes, test_nodes = train_test_split(list(labelled_nodes.keys()),
                                                      **config["learning"]["train_test_split"])


train_nodes += val_nodes  # no validation set as we will not optimize hyperparameters


extractor = NodeFeaturesExtractor(db=config["db"], max_num_nodes=10000)
preprocessor = Preprocessing()


print("Train - extracting the node features.")
X_train, y_train = extractor.extract(nodes=train_nodes,  label_values=config["categories"])
print("Train - fitting pre-processing")
preprocessor.fit(X_train)
print("Train - transforming the node features.")
X_train = preprocessor.transform(X_train)
y_train = pd.Series(y_train).loc[X_train.index]


print("Test - extracting the node features.")
X_test, y_test = extractor.extract(nodes=test_nodes,  label_values=config["categories"])
print("Test - transforming the node features")
X_test = preprocessor.transform(X_test)
y_test = pd.Series(y_test).loc[X_test.index]


print("Training the Gradient Boosting classifier.")
clf = GradientBoostingClassifier()
clf.fit(X_train.values, y_train.values)


print("Predicting the labels in the test set.")
y_pred = clf.predict(X_test.values)
report = classification_report(y_test, y_pred, output_dict=True)


print("Saving the model.")
folder = "results"
if not os.path.exists(folder):
    os.mkdir(folder)
folder = os.path.join(folder, "GB")
if not os.path.exists(folder):
    os.mkdir(folder)
folder = os.path.join(folder, str(datetime.datetime.now()))
os.mkdir(folder)
with open(os.path.join(folder, "conf.yaml"), "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
with open(os.path.join(folder, "report.yaml"), "w") as f:
    yaml.dump(report, f, default_flow_style=False, sort_keys=False)
with open(os.path.join(folder, "model.pkl"), "wb") as f:
    pickle.dump(clf, f)


print("Computing the feature importances.")
feature_importances = clf.feature_importances_
cols = preprocessor.columns
features_df = (pd.DataFrame({'Feature': cols, 'Importance': feature_importances})
               .sort_values(by='Importance', ascending=False).head(10))
fig = go.Figure(go.Bar(x=features_df["Feature"], y=features_df["Importance"], marker=dict(color="royalblue")))
fig.update_layout(yaxis_title="Importance", template="simple_white", height=500, width=1000,
                  font=dict( family="Computer Modern, serif"))
fig.write_image(os.path.join(folder, "feature_importances.jpg"), format="jpeg")
