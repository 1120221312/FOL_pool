import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NNPULoss(nn.Module):
    """Non-Negative Positive-Unlabeled Learning Loss"""
    
    def __init__(self, prior, loss_func='sigmoid', beta=0.0, gamma=1.0):
        """
        Args:
            prior: Prior probability of positive class
            loss_func: Loss function type ('sigmoid' or 'logistic')
            beta: Coefficient for negative risk
            gamma: Coefficient for non-negative risk
        """
        super(NNPULoss, self).__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma
        self.loss_func = loss_func
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model predictions (logits)
            targets: Labels (1 for positive, -1 for unlabeled)
        """
        # Ensure float32 consistency
        outputs = outputs.float()
        
        # Separate positive and unlabeled samples
        positive_mask = (targets == 1)
        unlabeled_mask = (targets == -1)
        
        if self.loss_func == 'sigmoid':
            # Sigmoid loss
            loss_func = lambda x: torch.sigmoid(-x)
        else:
            # Logistic loss  
            loss_func = lambda x: torch.log(1 + torch.exp(-x))
        
        # Positive risk
        if torch.sum(positive_mask) > 0:
            positive_risk = torch.mean(loss_func(outputs[positive_mask]))
        else:
            positive_risk = torch.tensor(0.0, device=outputs.device)
        
        # Negative risk (estimated from unlabeled samples)
        if torch.sum(unlabeled_mask) > 0:
            unlabeled_risk = torch.mean(loss_func(-outputs[unlabeled_mask]))
            negative_risk = unlabeled_risk - self.prior * positive_risk
        else:
            negative_risk = torch.tensor(0.0, device=outputs.device)
        
        # Non-negative risk
        if negative_risk < -self.beta:
            return -self.gamma * negative_risk
        else:
            return self.prior * positive_risk + negative_risk


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for binary classification with proper initialization"""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256], dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Proper initialization for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        return self.network(x).squeeze(-1)


class LinearClassifier(nn.Module):
    """Simple linear classifier for comparison"""
    
    def __init__(self, input_dim=768):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        return self.linear(x).squeeze(-1)


class NNPUModel(nn.Module):
    """Main NNPU learning model wrapper"""
    
    def __init__(self, input_dim=768, model_type='mlp', prior=0.3, 
                 hidden_dims=[512, 256], dropout=0.3):
        super(NNPUModel, self).__init__()
        
        self.prior = prior
        self.model_type = model_type
        
        # Create the classifier
        if model_type == 'mlp':
            self.classifier = MLPClassifier(input_dim, hidden_dims, dropout)
        elif model_type == 'linear':
            self.classifier = LinearClassifier(input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create the loss function
        self.loss_fn = NNPULoss(prior=prior, beta=0.0, gamma=1.0)
        
    def forward(self, x):
        return self.classifier(x)
    
    def compute_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)
    
    def predict_proba(self, x):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs
    
    def predict(self, x, threshold=0.5):
        """Get binary predictions"""
        probs = self.predict_proba(x)
        return (probs > threshold).long()


def create_model(input_dim=768, model_type='mlp', prior=0.3, device='cpu'):
    """Factory function to create NNPU models with proper device handling"""
    
    model = NNPUModel(
        input_dim=input_dim,
        model_type=model_type,
        prior=prior,
        hidden_dims=[512, 256],  # Appropriate for text embeddings
        dropout=0.3
    )
    
    # Move to device and ensure float32
    model = model.to(device).float()
    
    print(f"Created {model_type} model with input_dim={input_dim}, prior={prior:.3f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def get_optimizer(model, learning_rate=1e-3, weight_decay=1e-5):
    """Get optimizer with appropriate settings for text embeddings"""
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def get_scheduler(optimizer, num_epochs=50, warmup_epochs=5):
    """Get learning rate scheduler with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 ** ((epoch - warmup_epochs) // 10)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)