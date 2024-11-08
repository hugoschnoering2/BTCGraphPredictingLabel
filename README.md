

# Bitcoin Temporal Graph Analysis

## Prerequisites

Before using this repository, follow the steps below:

### 1. Download the Dataset
Download the temporal graph dataset from the following link:  
[BitcoinTemporalGraph Dataset](https://figshare.com/articles/dataset/BitcoinTemporalGraph/26305093)

### 2. Decompress and Load the Dataset into PostgreSQL

- **Decompress the Archive**  
  Run the following command to decompress the dataset:
  ```bash
  pigz -p 10 -dc dataset.tar.gz | tar -xvf -
  ```

- **Restore the Tables into PostgreSQL**  
  Use the `pg_restore` utility to restore the tables into an existing PostgreSQL database:
  ```bash
  pg_restore -j number_jobs -Fd -O -U database_username -d database_name dataset
  ```

  **Important**: Ensure your database has sufficient storage:
  - `node_features` table: ~40GB
  - `transaction_edges` table: ~80GB (including indexes)

---

## Step 2: Install Python Dependencies

Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Step 3: Create Configuration File

Create a `conf.yaml` file using `example_conf.yaml` as a template. Ensure the following fields are correctly configured:

```yaml
db:
  endpoint: "PostgreSQL endpoint, ex: 127.0.0.1"
  user: "PostgreSQL username, ex: postgres"
  port: "PostgreSQL port, ex: 5432"
  db: "PostgreSQL database name, ex: postgres"
  password: "PostgreSQL database password"
```

This configuration is required to connect to the PostgreSQL database where the dataset is stored.

---

## Step 4: Train Models

- **Train Gradient Boosting Model**  
  Run the following script to train a gradient boosting model:
  ```bash
  python train_gradient_boosting.py
  ```

- **Train Graph Neural Network Model**  
  Run the following script to train a graph neural network model:
  ```bash
  python train_gnn.py
  ```


## Important Notes

- The first iteration of `train_gnn.py` may take a significant amount of time because neighborhood sampling is performed. Once completed, the sampled neighborhoods will be saved, and subsequent runs will reuse these neighborhoods, skipping this step.

- Ensure PostgreSQL is configured for optimal performance. For example, we recommend setting the following parameters:
  - `shared_buffers`: 1,048,576 (default: 16,384)
  - `work_mem`: 16,384 (default: 4,096)
  - `maintenance_work_mem`: 1,048,576 (default: 65,536)
  - `wal_buffers`: 2,048 (default: -1)
  - `max_parallel_workers_per_gather`: 4 (default: 2)

  Matching or exceeding these configurations will help achieve better performance.

## Configuration Parameters in `conf.yaml`

This section describes the additional parameters in the `conf.yaml` file:

### `categories`
- **Description**: A list of labels that will be used during training.

---

### `model`
- **name**: The name of the GNN model to use. Options include:
  - `GCN`
  - `GraphSage`
  - `GIN`
  - `GAT`
- **args**: Arguments passed to the model, which is a subclass of the PyTorch Geometric `BasicGNN`.  
  Reference: [torch_geometric.nn.models.BasicGNN](https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/nn/models/basic_gnn.html)

---

### `learning`

- **`train_test_split`**:
  - `prop_val`: Proportion of the dataset for validation.
  - `prop_test`: Proportion of the dataset for testing.
  - `max_nodes`: Maximum number of nodes in each split.
  - `seed`: Seed value for random splitting.

- **`init_learning_rate`**: Initial learning rate.
- **`max_num_epochs`**: Maximum number of training epochs.
- **`batch_size`**: Batch size used during training.
- **`min_num_samples`**: Minimum number of samples per label in each split.
- **`max_num_samples`**: Maximum number of samples per label in each split.

- **`resampling_period`**: Number of epochs between neighborhood resampling.
- **`reload_best`**: If `true`, reloads the best model from previous neighborhoods; otherwise, continues with the current neighborhoods.

- **`lr_scheduler`**: Parameters for the learning rate scheduler (`ReduceLROnPlateau`).  
  Reference: [torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau)

- **`early_stopping`**:
  - `patience`: Number of epochs with no improvement before stopping training.

---

### `sampling`

- **`num_neighbors`**: A list defining the number of neighbors to sample at each depth relative to the seed node.
- **`n_jobs`**: Number of parallel jobs to use during sampling.

---

### `buffer`

- **`folder`**: Name of the folder where sampled neighborhoods will be saved.
- **`size`**: Number of neighborhoods to sample for each dataset instance.

---

### `global_seed`
- **Description**: Global seed value for ensuring reproducibility.
