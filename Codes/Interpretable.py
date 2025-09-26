import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
# -------------------------------
# 1. Load the trained model and tokenizer
# -------------------------------
model_path = "loading_finetuned_model"  # Update if needed
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------
# 2. Dataset class and prediction function
# -------------------------------
class StoryPointDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

def predict_batch(model, tokenizer, texts):
    dataset = StoryPointDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=16)
    predictions = []
    label_mapping = {0: "Small", 1: "Medium", 2: "Large", 3: "Huge"}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    # Return predicted labels as strings (for saving to CSV)
    return [label_mapping[pred] for pred in predictions]

# -------------------------------
# 3. Process CSV files in a folder
# -------------------------------
def process_csv_folder(folder_path):
    all_texts = []
    all_labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"Skipping {file_name}: Missing 'text' or 'label' column.")
                continue
            
            # Convert labels to int if they are not already
            df['label'] = df['label'].astype(int)
            
            predictions = predict_batch(model, tokenizer, df['text'].tolist())
            df['predicted_label'] = predictions

            # Save the results
            output_path = os.path.join(predict_folder_path, f"predicted_{file_name}")
            df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")

            # Collect data for clustering
            all_texts.extend(df['text'].tolist())
            all_labels.extend(df['label'].tolist())
    return all_texts, all_labels

# -------------------------------
# 4. Extract BERT [CLS] token embeddings
# -------------------------------
def extract_bert_embeddings(model, tokenizer, texts, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding)
    return np.vstack(embeddings)

# -------------------------------
# 5. Apply K-Means clustering
# -------------------------------
def cluster_bert_embeddings(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)
    return cluster_assignments, kmeans

# -------------------------------
# 6. Evaluate clustering performance
# -------------------------------
def evaluate_clustering(true_labels, cluster_assignments):
    ari_score = adjusted_rand_score(true_labels, cluster_assignments)
    print(f"Cluster-Language Alignment Score (ARI): {ari_score:.4f}")
    return ari_score

# -------------------------------
# 7. 2D Visualization of Clusters using PCA
# -------------------------------
def visualize_clusters(embeddings, cluster_assignments, true_labels, label_names):
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Determine majority label for each cluster
    cluster_labels = {}
    for cluster in np.unique(cluster_assignments):
        indices = np.where(cluster_assignments == cluster)[0]
        cluster_true_labels = [true_labels[i] for i in indices]
        majority_label = max(set(cluster_true_labels), key=cluster_true_labels.count)
        try:
            majority_label = int(majority_label)
        except ValueError:
            majority_label = 0
        if majority_label >= len(label_names):
            majority_label = len(label_names) - 1
        cluster_labels[cluster] = label_names[majority_label]

    # Create a color mapping for each cluster using the viridis colormap
    unique_clusters = np.unique(cluster_assignments)
    num_clusters = len(unique_clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    color_dict = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    cluster_colors = [color_dict[c] for c in cluster_assignments]

    # Prepare figure with a 3x2 grid
    fig = plt.figure(figsize=(12, 15))
    
    # Main cluster plot (spanning two columns)
    ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax_main.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_colors, s=50)
    ax_main.set_title('BERT Embeddings Clustered via K-Means')
    ax_main.set_xlabel("PCA Component 1")
    ax_main.set_ylabel("PCA Component 2")
    
    # Annotate cluster centers with majority label
    for cluster in unique_clusters:
        indices = np.where(cluster_assignments == cluster)[0]
        cluster_center = np.mean(reduced_embeddings[indices, :], axis=0)
        ax_main.text(cluster_center[0], cluster_center[1], f'{cluster_labels[cluster]}',
                     fontsize=12, fontweight='bold', color='black',
                     horizontalalignment='center', verticalalignment='center')
    
    # Create legend patches for each cluster
    legend_patches = [mpatches.Patch(color=color_dict[cluster], label=f'Small,Medium,Large,Huge')
                      for cluster in unique_clusters]
    ax_main.legend(handles=legend_patches, title='Label Name', loc='lower left')
    
    # Four subplots for each true label
    label_colors = ["red", "green", "blue", "orange"]
    true_labels_array = np.array(true_labels)
    subplot_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for i, pos in enumerate(subplot_positions):
        ax_label = plt.subplot2grid((3, 2), pos)
        label_indices = np.where(true_labels_array == i)[0]
        label_points = reduced_embeddings[label_indices]
        ax_label.scatter(label_points[:, 0], label_points[:, 1], c=label_colors[i], s=50)
        ax_label.set_title(f"Label {i} - {label_names[i]}")
        ax_label.set_xlabel("PCA Component 1")
        ax_label.set_ylabel("PCA Component 2")
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# 9. Debug: Print Cluster Composition
# -------------------------------
def debug_cluster_composition(cluster_assignments, true_labels, label_names):
    """
    For each cluster, prints the count of each true label.
    This helps in understanding if clusters (e.g., Medium and Small) are overlapping.
    """
    unique_clusters = np.unique(cluster_assignments)
    print("Cluster Composition:")
    for cluster in unique_clusters:
        indices = np.where(cluster_assignments == cluster)[0]
        cluster_true_labels = [true_labels[i] for i in indices]
        # Count occurrences for each label 0, 1, 2, 3
        counts = {i: cluster_true_labels.count(i) for i in range(len(label_names))}
        print(f"Cluster {cluster}: {counts}")

# -------------------------------
# 10. Main Script Execution
# -------------------------------
test_folder = "test-part"  # Update this path as needed
predict_folder_path = "perdicted-folder" 
all_texts, all_labels = process_csv_folder(test_folder)

# Extract BERT embeddings from texts
bert_embeddings = extract_bert_embeddings(model, tokenizer, all_texts, device)

# Cluster embeddings using K-Means
cluster_assignments, kmeans_model = cluster_bert_embeddings(bert_embeddings, num_clusters=4)

# Optionally, evaluate clustering performance
evaluate_clustering(all_labels, cluster_assignments)

# Define label names corresponding to numeric labels 0,1,2,3
label_names = ["Small", "Medium", "Large", "Huge"]

# Visualize clusters in 2D with subplots
visualize_clusters(bert_embeddings, cluster_assignments, all_labels, label_names)

# Debug: Print cluster composition to check the overlap between Medium and Small
debug_cluster_composition(cluster_assignments, all_labels, label_names)