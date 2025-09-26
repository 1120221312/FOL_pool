import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import time
import warnings
from dataset import load_and_prepare_data, validate_data_types
from model import create_model, get_optimizer, get_scheduler


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate various evaluation metrics with robust error handling"""
    metrics = {}
    
    try:
        # Convert to numpy arrays
        y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_prob = y_prob.cpu().numpy() if torch.is_tensor(y_prob) else y_prob
        
        # Handle binary labels for metrics (convert -1 to 0 for sklearn)
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        
        # AUC with robust error handling
        try:
            if len(np.unique(y_true_binary)) > 1:  # Check if both classes present
                metrics['auc'] = roc_auc_score(y_true_binary, y_prob)
            else:
                metrics['auc'] = 0.5  # Default AUC when only one class present
        except ValueError as e:
            warnings.warn(f"AUC calculation failed: {e}")
            metrics['auc'] = 0.5
        
        # Precision and recall with zero_division handling
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
            
    except Exception as e:
        warnings.warn(f"Error calculating metrics: {e}")
        # Return default metrics
        metrics = {'accuracy': 0.0, 'auc': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    return metrics


def train_epoch(model, train_loader, optimizer, device, clip_grad_norm=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    for batch_embeddings, batch_labels in train_loader:
        # Move to device and ensure float32
        batch_embeddings = batch_embeddings.to(device).float()
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = model.compute_loss(outputs, batch_labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long() * 2 - 1  # Convert to {-1, 1}
            
            all_outputs.extend(probs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(np.array(all_targets), 
                               (np.array(all_outputs) > 0.5).astype(int) * 2 - 1,
                               np.array(all_outputs))
    
    return avg_loss, metrics


def evaluate(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            # Move to device and ensure float32
            batch_embeddings = batch_embeddings.to(device).float()
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_embeddings)
            loss = model.compute_loss(outputs, batch_labels)
            
            total_loss += loss.item()
            
            # Store predictions
            probs = torch.sigmoid(outputs)
            all_outputs.extend(probs.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    metrics = calculate_metrics(np.array(all_targets),
                               (np.array(all_outputs) > 0.5).astype(int) * 2 - 1,
                               np.array(all_outputs))
    
    return avg_loss, metrics


def train_nnpu_model(data_path, num_epochs=50, learning_rate=1e-3, 
                     model_type='mlp', device=None, verbose=True):
    """
    Main training function for NNPU learning
    
    Args:
        data_path: Path to the data file
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_type: Type of model ('mlp' or 'linear')
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print progress
    """
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading and preparing data...")
    train_loader, test_loader, prior = load_and_prepare_data(
        data_path, 
        test_size=0.2, 
        random_state=42,
        prior_cap=0.8,  # Cap prior to prevent numerical instability
        embedding_dim=768
    )
    
    # Validate data types
    validate_data_types(train_loader)
    
    # Get embedding dimension from data
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = create_model(
        input_dim=input_dim,
        model_type=model_type,
        prior=prior,
        device=device
    )
    
    # Create optimizer and scheduler with appropriate settings for text embeddings
    optimizer = get_optimizer(model, learning_rate=learning_rate, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=5)
    
    # Training loop
    best_test_auc = 0.0
    train_losses = []
    test_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, clip_grad_norm=1.0
        )
        
        # Evaluate
        test_loss, test_metrics = evaluate(model, test_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Store losses
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Track best model
        if test_metrics['auc'] > best_test_auc:
            best_test_auc = test_metrics['auc']
        
        # Print progress
        if verbose and (epoch + 1) % 5 == 0:
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Test AUC: {test_metrics['auc']:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
    
    # Final evaluation with fixed f-string formatting
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Test AUC: {best_test_auc:.4f}")
    
    # Final comprehensive evaluation
    final_train_loss, final_train_metrics = evaluate(model, train_loader, device)
    final_test_loss, final_test_metrics = evaluate(model, test_loader, device)
    
    print("\nFinal Results:")
    print(f"Train - Loss: {final_train_loss:.4f}, "
          f"AUC: {final_train_metrics['auc']:.4f}, "
          f"Accuracy: {final_train_metrics['accuracy']:.4f}")
    print(f"Test  - Loss: {final_test_loss:.4f}, "
          f"AUC: {final_test_metrics['auc']:.4f}, "
          f"Accuracy: {final_test_metrics['accuracy']:.4f}")
    
    return model, {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_auc': best_test_auc,
        'final_train_metrics': final_train_metrics,
        'final_test_metrics': final_test_metrics
    }


def main():
    """Main function to run NNPU training"""
    
    # Configuration
    config = {
        'data_path': '/home/runner/work/FOL_pool/FOL_pool/formatted_merged_10_output_dispute_merged_output.pkl',
        'num_epochs': 50,
        'learning_rate': 1e-3,  # Appropriate for text embeddings
        'model_type': 'mlp',
        'device': None,  # Auto-detect
        'verbose': True
    }
    
    print("NNPU Learning Training")
    print("=" * 40)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 40)
    
    try:
        # Train the model
        model, results = train_nnpu_model(**config)
        
        print("\nTraining successful!")
        return model, results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, results = main()