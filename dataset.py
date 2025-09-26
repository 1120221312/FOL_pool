import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import warnings


class PUDataset(Dataset):
    """Dataset for Positive-Unlabeled learning with proper float32 handling"""
    
    def __init__(self, embeddings, labels):
        # Ensure float32 data type for consistency
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def load_and_prepare_data(data_path, test_size=0.2, random_state=42, 
                         prior_cap=0.8, embedding_dim=768):
    """
    Load and prepare data for NNPU learning with proper data type handling
    
    Args:
        data_path: Path to the pickle file containing the data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        prior_cap: Maximum allowed prior probability to prevent numerical instability
        embedding_dim: Expected embedding dimension
    
    Returns:
        train_loader, test_loader, prior_probability
    """
    try:
        # Load data from pickle file
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data with keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")
        
        # Extract embeddings and labels based on data structure
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Legal case data structure - create embeddings from case types
            print("Processing legal case data structure...")
            all_case_types = []
            all_labels = []
            
            # Flatten the hierarchical structure
            for case_group in data:
                for case_type, cases in case_group.items():
                    if isinstance(cases, dict):
                        for case_id, case_data in cases.items():
                            all_case_types.append(case_type)
                            # Create positive/negative labels based on case type frequency
                            # More frequent case types get positive labels
                            all_labels.append(1 if len(cases) > 5 else -1)
            
            print(f"Found {len(all_case_types)} cases across {len([k for d in data for k in d.keys()])} case types")
            
            # Create embeddings using TF-IDF on case type names (simplified approach)
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Use character n-grams for Chinese text
            vectorizer = TfidfVectorizer(
                max_features=embedding_dim,
                analyzer='char',
                ngram_range=(1, 3),
                dtype=np.float32
            )
            
            embeddings = vectorizer.fit_transform(all_case_types).toarray().astype(np.float32)
            labels = np.array(all_labels)
            
            print(f"Created embeddings with shape: {embeddings.shape}")
            
        elif isinstance(data, dict):
            # Assume data has embeddings and labels keys
            if 'embeddings' in data and 'labels' in data:
                embeddings = data['embeddings']
                labels = data['labels']
            else:
                # If not, try to extract from the data structure
                keys = list(data.keys())
                embeddings = data[keys[0]]  # Assume first key is embeddings
                labels = data[keys[1]] if len(keys) > 1 else np.ones(len(embeddings))
        else:
            # If data is a list or array, create dummy embeddings
            embeddings = np.random.randn(len(data), embedding_dim).astype(np.float32)
            labels = np.ones(len(data))
        
        # Convert to numpy arrays and ensure float32 for embeddings
        embeddings = np.array(embeddings, dtype=np.float32)
        labels = np.array(labels)
        
        print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        print(f"Labels shape: {labels.shape}, unique values: {np.unique(labels)}")
        
        # Validate embedding dimension
        if embeddings.shape[1] != embedding_dim:
            warnings.warn(f"Expected embedding dimension {embedding_dim}, got {embeddings.shape[1]}")
        
        # Create binary labels for PU learning
        # Assume positive class is 1, negative/unlabeled is 0 or -1
        positive_mask = labels > 0
        negative_mask = labels <= 0
        
        # Create PU labels: 1 for positive, -1 for unlabeled
        pu_labels = np.where(positive_mask, 1, -1)
        
        # Calculate and validate prior probability
        n_positive = np.sum(positive_mask)
        n_total = len(labels)
        prior = n_positive / n_total
        
        # Cap prior probability to prevent numerical instability
        if prior > prior_cap:
            warnings.warn(f"Prior probability {prior:.3f} is too high, capping at {prior_cap}")
            prior = prior_cap
        
        print(f"Prior probability: {prior:.3f}")
        print(f"Positive samples: {n_positive}, Total samples: {n_total}")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, pu_labels, test_size=test_size, random_state=random_state,
            stratify=pu_labels
        )
        
        # Create datasets
        train_dataset = PUDataset(X_train, y_train)
        test_dataset = PUDataset(X_test, y_test)
        
        # Create data loaders with appropriate batch size for embedding dimension
        batch_size = min(64, len(X_train) // 10)  # Adaptive batch size
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, prior
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback: create synthetic data for testing
        print("Creating synthetic data for testing...")
        return create_synthetic_data(embedding_dim, prior_cap)


def create_synthetic_data(embedding_dim=768, prior_cap=0.8, n_samples=1000):
    """Create synthetic data for testing when real data is not available"""
    
    # Create synthetic embeddings with float32 dtype
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Create synthetic labels with reasonable prior
    n_positive = int(n_samples * 0.3)  # 30% positive samples
    labels = np.concatenate([
        np.ones(n_positive),
        -np.ones(n_samples - n_positive)
    ])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    embeddings = embeddings[indices]
    labels = labels[indices]
    
    prior = n_positive / n_samples
    
    print(f"Created synthetic data: {n_samples} samples, {embedding_dim} dimensions")
    print(f"Prior probability: {prior:.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and loaders
    train_dataset = PUDataset(X_train, y_train)
    test_dataset = PUDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, prior


def validate_data_types(data_loader):
    """Validate that data types are consistent (float32 for embeddings)"""
    for batch_embeddings, batch_labels in data_loader:
        assert batch_embeddings.dtype == torch.float32, f"Expected float32, got {batch_embeddings.dtype}"
        assert batch_labels.dtype == torch.long, f"Expected long, got {batch_labels.dtype}"
        break  # Only check first batch
    print("Data type validation passed: embeddings are float32, labels are long")