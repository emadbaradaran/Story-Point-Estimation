import torch
import torch.nn.functional as F
import torch.nn
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Load datasets and increase quality and size
def load_datasets_from_folder(folder_path):
    """Load datasets from a folder containing multiple CSV files."""
    dataframes = []
    try:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df = df.dropna(subset=['title', 'description', 'storypoint'])

                def classify_story_points(sp):
                    if sp <= 5:
                        return 0  # Small
                    elif sp <= 15:
                        return 1  # Medium
                    elif sp <= 40:
                        return 2  # Large
                    else:
                        return 3  # Huge

                df['label'] = df['storypoint'].apply(classify_story_points)
                df['text'] = df['title'] + ". " + df['description']
                dataframes.append(df[['text', 'label']])
        return pd.concat(dataframes, ignore_index=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

# Define constants
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256 # Increased max length for better context
BATCH_SIZE = 16
EPOCHS = 5  # Increased epochs for better convergence
LEARNING_RATE = 1e-5  # Fine-tuned learning rate
NUM_CLASSES = 4  # Small, Medium, Large, Huge
PATIENCE = 2  # Early stopping patience
N_SPLITS = 5  # Number of folds for StratifiedKFold

# Dataset class
class StoryPointDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Ensure texts and labels are list-like for indexing
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if isinstance(labels, pd.Series) else labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
# Model training function
def train_model(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

# Model evaluation function
def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    # Compute Precision, Recall, F1-score
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    # Compute ROC AUC Score (for multi-class, "ovr" strategy)
    roc_auc = roc_auc_score(
        F.one_hot(torch.tensor(all_labels), num_classes=NUM_CLASSES).numpy(),
        F.one_hot(torch.tensor(all_preds), num_classes=NUM_CLASSES).numpy(),
        average="macro",
        multi_class="ovr",
    )
    return total_loss / len(dataloader), accuracy, precision, recall, f1, roc_auc, all_preds, all_labels

# Main script
if __name__ == "__main__":
    # Load data
    folder_path = "dataset_path"  # Replace with actual folder path
    data = load_datasets_from_folder(folder_path)
    
    # Extract features and labels
    texts, labels = data['text'], data['label']
    
    # Oversample minority classes
    ros = RandomOverSampler(random_state=42)
    texts_np = np.array(texts).reshape(-1, 1)
    labels_np = np.array(labels)
    texts_resampled, labels_resampled = ros.fit_resample(texts_np, labels_np)
    texts_resampled = texts_resampled.flatten()

    # Validate oversampling results
    unique_labels, counts = np.unique(labels_resampled, return_counts=True)
    print("Label distribution after oversampling:", dict(zip(unique_labels, counts)))
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Create results directories
    cv_results_folder = "Results/cross_validation"
    os.makedirs(cv_results_folder, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Lists to store metrics across folds
    fold_train_losses = []
    fold_train_accuracies = []
    fold_test_losses = []
    fold_test_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_roc_aucs = []
    
    # Cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_resampled, labels_resampled)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        # Split data for this fold
        fold_train_texts = texts_resampled[train_idx]
        fold_train_labels = labels_resampled[train_idx]
        fold_test_texts = texts_resampled[test_idx]
        fold_test_labels = labels_resampled[test_idx]
        
        # Create datasets
        fold_train_dataset = StoryPointDataset(fold_train_texts, fold_train_labels, tokenizer)
        fold_test_dataset = StoryPointDataset(fold_test_texts, fold_test_labels, tokenizer)
        
        # Create dataloaders
        fold_train_loader = DataLoader(fold_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        fold_test_loader = DataLoader(fold_test_dataset, batch_size=BATCH_SIZE)
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
        model.bert.config.hidden_dropout_prob = 0.2  # Dropout for regularization
        model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, 
            num_training_steps=len(fold_train_loader) * EPOCHS
        )
        
        # Weighted loss function based on class distribution
        class_weights = torch.tensor([1.0, 2.0, 5.0, 10.0]).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        fold_losses = []
        fold_accuracies = []
        
        for epoch in range(EPOCHS):
            # Train the model
            train_loss, train_acc = train_model(model, fold_train_loader, optimizer, scheduler, loss_fn, device)
            fold_losses.append(train_loss)
            fold_accuracies.append(train_acc)
            
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Save epoch results
            fold_epoch_dir = os.path.join(cv_results_folder, f"fold_{fold+1}_epoch_{epoch+1}")
            os.makedirs(fold_epoch_dir, exist_ok=True)
            model.save_pretrained(fold_epoch_dir)
        
        # Store training metrics for this fold
        fold_train_losses.append(fold_losses)
        fold_train_accuracies.append(fold_accuracies)
        
        # Evaluate on test set
        test_loss, test_acc, precision, recall, f1, roc_auc, test_preds, test_true = evaluate_model(
            model, fold_test_loader, loss_fn, device
        )
        
        # Store evaluation metrics
        fold_test_losses.append(test_loss)
        fold_test_accuracies.append(test_acc)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_roc_aucs.append(roc_auc)
        
        # Print fold results
        print(f"Fold {fold+1} Test Results:")
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Create confusion matrix
        cm = confusion_matrix(test_true, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Small", "Medium", "Large", "Huge"],
                    yticklabels=["Small", "Medium", "Large", "Huge"])
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(cv_results_folder, f"fold_{fold+1}_confusion_matrix.png"))
        plt.close()
        
        # Save detailed classification report
        report = classification_report(
            test_true, test_preds,
            target_names=["Small", "Medium", "Large", "Huge"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(cv_results_folder, f"fold_{fold+1}_classification_report.csv"))
        
        # Save final fold model
        fold_model_dir = os.path.join(cv_results_folder, f"fold_{fold+1}_final_model")
        os.makedirs(fold_model_dir, exist_ok=True)
        model.save_pretrained(fold_model_dir)
        tokenizer.save_pretrained(fold_model_dir)
    
    # Calculate average metrics
    avg_test_acc = np.mean(fold_test_accuracies)
    std_test_acc = np.std(fold_test_accuracies)
    avg_precision = np.mean(fold_precisions)
    std_precision = np.std(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    std_recall = np.std(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    avg_roc_auc = np.mean(fold_roc_aucs)
    std_roc_auc = np.std(fold_roc_aucs)
    
    # Print and save cross-validation summary
    print("\n=== Cross-Validation Summary ===")
    print(f"Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
    
    # Save summary to file
    with open(os.path.join(cv_results_folder, "cv_summary.txt"), "w") as f:
        f.write("=== Cross-Validation Summary ===\n")
        f.write(f"Number of Folds: {N_SPLITS}\n")
        f.write(f"Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}\n")
        f.write(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}\n")
        f.write(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}\n")
        f.write(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"ROC AUC: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}\n")
    
    # Plot cross-validation metrics
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Test Accuracy across folds
    plt.subplot(2, 2, 1)
    plt.bar(range(1, N_SPLITS+1), fold_test_accuracies, color='skyblue')
    plt.axhline(y=avg_test_acc, color='r', linestyle='-', label=f'Mean: {avg_test_acc:.4f}')
    plt.fill_between([0.5, N_SPLITS+0.5], 
                     [avg_test_acc-std_test_acc, avg_test_acc-std_test_acc],
                     [avg_test_acc+std_test_acc, avg_test_acc+std_test_acc],
                     alpha=0.2, color='r')
    plt.title('Test Accuracy by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, N_SPLITS+1))
    plt.legend()
    
    # Plot 2: F1 Score across folds
    plt.subplot(2, 2, 2)
    plt.bar(range(1, N_SPLITS+1), fold_f1s, color='lightgreen')
    plt.axhline(y=avg_f1, color='r', linestyle='-', label=f'Mean: {avg_f1:.4f}')
    plt.fill_between([0.5, N_SPLITS+0.5], 
                     [avg_f1-std_f1, avg_f1-std_f1],
                     [avg_f1+std_f1, avg_f1+std_f1],
                     alpha=0.2, color='r')
    plt.title('F1 Score by Fold')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, N_SPLITS+1))
    plt.legend()
    
    # Plot 3: Precision across folds
    plt.subplot(2, 2, 3)
    plt.bar(range(1, N_SPLITS+1), fold_precisions, color='coral')
    plt.axhline(y=avg_precision, color='r', linestyle='-', label=f'Mean: {avg_precision:.4f}')
    plt.fill_between([0.5, N_SPLITS+0.5], 
                     [avg_precision-std_precision, avg_precision-std_precision],
                     [avg_precision+std_precision, avg_precision+std_precision],
                     alpha=0.2, color='r')
    plt.title('Precision by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Precision')
    plt.xticks(range(1, N_SPLITS+1))
    plt.legend()
    
    # Plot 4: Recall across folds
    plt.subplot(2, 2, 4)
    plt.bar(range(1, N_SPLITS+1), fold_recalls, color='lightblue')
    plt.axhline(y=avg_recall, color='r', linestyle='-', label=f'Mean: {avg_recall:.4f}')
    plt.fill_between([0.5, N_SPLITS+0.5], 
                     [avg_recall-std_recall, avg_recall-std_recall],
                     [avg_recall+std_recall, avg_recall+std_recall],
                     alpha=0.2, color='r')
    plt.title('Recall by Fold')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.xticks(range(1, N_SPLITS+1))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(cv_results_folder, "cv_metrics_summary.png"))
    plt.close()
    
    # Plot learning curves (training loss and accuracy) for each fold
    plt.figure(figsize=(15, 10))
    
    # Plot training loss curves
    plt.subplot(2, 1, 1)
    for fold, losses in enumerate(fold_train_losses):
        plt.plot(range(1, EPOCHS+1), losses, marker='o', label=f'Fold {fold+1}')
    plt.title('Training Loss by Epoch for Each Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot training accuracy curves
    plt.subplot(2, 1, 2)
    for fold, accs in enumerate(fold_train_accuracies):
        plt.plot(range(1, EPOCHS+1), accs, marker='o', label=f'Fold {fold+1}')
    plt.title('Training Accuracy by Epoch for Each Fold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(cv_results_folder, "learning_curves.png"))
    plt.close()
    
    # Create a final model trained on all data if needed
    print("\nTraining final model on all data...")
    
    # Create datasets from all available data
    full_dataset = StoryPointDataset(texts_resampled, labels_resampled, tokenizer)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize final model
    final_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    final_model.bert.config.hidden_dropout_prob = 0.2
    final_model.to(device)
    
    # Initialize optimizer and scheduler for final model
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE)
    final_scheduler = get_scheduler(
        "linear", optimizer=final_optimizer, num_warmup_steps=0, 
        num_training_steps=len(full_loader) * EPOCHS
    )
    
    # Train final model
    final_losses = []
    final_accs = []
    
    for epoch in range(EPOCHS):
        # Train
        loss, acc = train_model(final_model, full_loader, final_optimizer, final_scheduler, loss_fn, device)
        final_losses.append(loss)
        final_accs.append(acc)
        
        print(f"Final Model - Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Save final model
    final_model_dir = "Results/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    final_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Final model saved to {final_model_dir}")
    
    # Plot final model training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), final_losses, marker='o', color='blue')
    plt.title('Final Model - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), final_accs, marker='o', color='green')
    plt.title('Final Model - Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("Results", "final_model_training.png"))
    plt.close()
