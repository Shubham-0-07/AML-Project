import pandas as pd
import numpy as np
import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration 
OUTPUT_PATH = '../outputs/'
DATA_PATH = '../outputs/'
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'visualizations'), exist_ok=True)


# Load and Prepare Data for Graph 
def load_and_prepare_graph_data():
    print("Loading feature-engineered data for GNN...")
    file_path = os.path.join(DATA_PATH, "feature_engineered_data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} was not found. Please run the feature engineering script first.")
    
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    
    # Create Node Mapping
    all_accounts = pd.concat([df['sender_account'], df['receiver_account']]).unique()
    account_map = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)
    print(f"Total unique accounts (nodes): {num_nodes}")

    # Create Node Features
    node_features_df = df.groupby('account_id').agg(
        avg_sent=('amount', 'mean'),
        max_sent=('amount', 'max'),
        total_sent=('amount', 'sum'),
        txn_count=('transaction_id', 'count')
    ).reset_index()

    # Merge with static customer info
    cust_info = df[['account_id', 'annual_income', 'is_on_watchlist']].drop_duplicates()
    node_features_df = pd.merge(node_features_df, cust_info, on='account_id', how='left')
    
    # Create a complete feature matrix for all nodes
    x = np.zeros((num_nodes, node_features_df.shape[1] - 1)) # -1 for account_id
    feature_cols = [col for col in node_features_df.columns if col != 'account_id']
    
    for _, row in node_features_df.iterrows():
        if row['account_id'] in account_map:
            node_idx = account_map[row['account_id']]
            x[node_idx] = row[feature_cols].values

    # Scale node features
    scaler = StandardScaler()
    x = scaler.fit_transform(np.nan_to_num(x))

    # Create Edge Index and Edge Features 
    senders = [account_map[acc] for acc in df['sender_account']]
    receivers = [account_map[acc] for acc in df['receiver_account']]
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    
    # Add transaction amount and type as edge features
    edge_attr_cols = ['amount', 'transaction_type', 'channel']
    # One-hot encode categorical edge features
    df_edge_attr = pd.get_dummies(df[edge_attr_cols], columns=['transaction_type', 'channel'])
    edge_attr_scaler = MinMaxScaler()
    edge_attr = edge_attr_scaler.fit_transform(df_edge_attr)

    # Create Node Labels (y) and Masks 
    suspicious_accounts = df[df['is_suspicious'] == 1]['account_id'].unique()
    y = torch.zeros(num_nodes, dtype=torch.long)
    for acc in suspicious_accounts:
        if acc in account_map:
            y[account_map[acc]] = 1

    # Temporal Split for Nodes 
    first_seen = df.groupby('account_id')['timestamp'].min().reset_index()
    first_seen_sorted = first_seen.sort_values('timestamp')
    split_point = int(len(first_seen_sorted) * 0.8)
    train_accounts_set = set(first_seen_sorted.iloc[:split_point]['account_id'])
    test_accounts_set = set(first_seen_sorted.iloc[split_point:]['account_id'])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for acc, i in account_map.items():
        if acc in train_accounts_set:
            train_mask[i] = True
        elif acc in test_accounts_set:
            test_mask[i] = True

    print(f"Nodes in training mask: {train_mask.sum().item()}")
    print(f"Nodes in testing mask: {test_mask.sum().item()}")
    if test_mask.sum().item() == 0:
        print("\nWarning: Temporal split resulted in empty test set. Falling back to random split.")
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * 0.8)
        train_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[indices[:train_size]], test_mask[indices[train_size:]] = True, True

    graph_data = Data(x=torch.tensor(x, dtype=torch.float),
                      edge_index=edge_index,
                      edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                      y=y,
                      train_mask=train_mask,
                      test_mask=test_mask)
    
    print("Graph data object with edge features created successfully.")
    return graph_data

# GNN Model Definition #
class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.edge_lin = torch.nn.Linear(num_edge_features, num_node_features)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

#  Training and Evaluation  #
def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def plot_confusion_matrix(cm, model_name):
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Benign', 'Predicted Suspicious'],
                yticklabels=['Actual Benign', 'Actual Suspicious'])
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
    # Save the plot
    plot_path = os.path.join(OUTPUT_PATH, 'visualizations', f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {plot_path}")
    plt.close()

def evaluate_gnn_model(y_true, y_pred_proba, model_name):
    print(f"\n Evaluating {model_name} ")
    y_pred_binary = y_pred_proba.argmax(axis=1)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
    auc_pr = average_precision_score(y_true, y_pred_proba[:, 1])
    
    if len(np.unique(y_true)) < 2:
        tn, fp, fn, tp = 0, 0, 0, 0
        if np.unique(y_true)[0] == 1: tp = len(y_true)
        else: tn = len(y_true)
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Plot confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    plot_confusion_matrix(cm, model_name)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}, AUC-PR: {auc_pr:.4f}, FPR: {fpr:.4f}")
    
    metrics = {
        "model_name": model_name, "precision": precision, "recall": recall, "f1_score": f1,
        "auc_roc": auc_roc, "auc_pr": auc_pr, "false_positive_rate": fpr,
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn)
    }
    metrics_file = os.path.join(OUTPUT_PATH, f"{model_name.lower().replace(' ', '_')}_metrics.json")
    with open(metrics_file, 'w') as f: json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

#  Main Function  #
def main():
    data = load_and_prepare_graph_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = GNNModel(num_node_features=data.num_node_features,
                     num_edge_features=data.num_edge_features,
                     hidden_channels=64, 
                     num_classes=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    num_positives = data.y[data.train_mask].sum()
    if num_positives == 0:
        criterion = torch.nn.NLLLoss()
    else:
        num_negatives = len(data.y[data.train_mask]) - num_positives
        class_weight = torch.tensor([1.0, num_negatives / num_positives], device=device)
        criterion = torch.nn.NLLLoss(weight=class_weight)

    print("\n Training GNN Model ")
    losses = []
    for epoch in range(1, 501):
        loss = train(model, data, optimizer, criterion)
        losses.append(loss)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('GraphSAGE Model Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(OUTPUT_PATH, 'visualizations', 'graphsage_training_loss_curve.png')
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Training loss curve saved to {loss_plot_path}")
    plt.close()

    model.eval()
    with torch.no_grad():
        out = model(data)
        
    test_true = data.y[data.test_mask].cpu().numpy()
    test_pred_proba = torch.exp(out[data.test_mask]).cpu().numpy()
    
    evaluate_gnn_model(test_true, test_pred_proba, "GraphSAGE GNN ")
    
    print("\n GNN Modeling Script Finished ")

if __name__ == "__main__":
    main()
