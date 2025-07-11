# Flow Speed Integration for Transformer Models

This document outlines the integration of a dynamic "flow speed" parameter into our transformer architecture. The flow speed parameter controls the rate of information flow through transformer blocks, acting as a scaling factor for the attention and feed-forward operations.

## 1. Conceptual Overview

The flow speed parameter serves as a scalar multiplier that controls the magnitude of updates in each transformer block:

```
x = x + flow_speed * self.attention(self.norm1(x))
x = x + flow_speed * self.feedforward(self.norm2(x))
```

This can be understood as controlling the step size in a gradient flow dynamic, where larger values accelerate convergence but may reduce stability, while smaller values provide smoother updates but slower convergence.

## 2. Implementation Components

### 2.1 TransformerBlock Modifications

```python
class TransformerBlock(nn.Module):
    # Existing initialization...
    
    def forward(self, x: torch.Tensor, flow_speed = None) -> torch.Tensor:
        """Apply transformer block to input."""
        # Attention with flow speed
        if flow_speed is None:
            flow_speed = torch.ones(x.size(0), device=x.device, requires_grad=False)
        
        # Reshape to batch dimension
        flow_speed = torch.reshape(flow_speed, (x.size(0), 1, 1))
        
        # Apply attention with flow speed scaling
        x = x + flow_speed * self.attention(self.norm1(x))
        
        # Feed-forward with flow speed scaling
        x = x + flow_speed * self.feedforward(self.norm2(x))
        
        return x
```

### 2.2 Flow Speed Predictor

#### 2.2.1 Dummy Implementation

For initial testing and integration, we'll start with a dummy flow predictor that simply returns a constant value of 1:

```python
class DummyFlowPredictor(nn.Module):
    """
    A simple flow predictor that returns a constant value of 1.
    Useful for testing the integration without worrying about prediction logic.
    """
    
    def __init__(self, per_layer=False, num_layers=None):
        super().__init__()
        self.per_layer = per_layer
        self.num_layers = num_layers
    
    def forward(self, targets=None, inputs=None):
        """
        Return constant flow speed of 1.
        
        Args:
            targets: Ignored, here for interface compatibility
            inputs: Ignored, here for interface compatibility
            
        Returns:
            Tensor of ones, shape depends on per_layer setting
        """
        if inputs is not None:
            batch_size = inputs.size(0)
        elif targets is not None and isinstance(targets, torch.Tensor):
            batch_size = targets.size(0)
        else:
            batch_size = 1
            
        device = next(self.parameters()).device if self.parameters() else (
            inputs.device if inputs is not None else torch.device('cpu')
        )
            
        if self.per_layer and self.num_layers is not None:
            # Return [batch_size, num_layers] tensor of ones
            return torch.ones(batch_size, self.num_layers, device=device)
        else:
            # Return [batch_size] tensor of ones
            return torch.ones(batch_size, device=device)
```

#### 2.2.2 Integration Pattern

The flow predictor integration follows this general pattern:

```python
# In GMMTransformer.__init__
if use_flow_predictor:
    # Start with dummy predictor for testing
    self.flow_predictor = DummyFlowPredictor(
        per_layer=False  # Set to True for per-layer flow speeds
    )

# In GMMTransformer.forward
if hasattr(self, 'flow_predictor') and targets is not None:
    # Get flow speeds from predictor
    # Could be [batch_size] or [batch_size, num_layers]
    flow_speed = self.flow_predictor(targets, x)
else:
    # Default flow speed if no predictor or targets
    flow_speed = torch.ones(batch_size, device=x.device)

# In TransformerBlock.forward
# Handle flow_speed appropriately based on shape
if flow_speed.dim() > 1:  # Per-layer flow speeds
    # Use appropriate layer index
    block_flow_speed = flow_speed[:, layer_idx]
else:  # Global flow speed
    block_flow_speed = flow_speed
```

The implementation can be customized based on your data structure and requirements. The key is that it should accept targets (and optionally inputs) and return appropriate flow speed values with gradients for end-to-end training.

### 2.3 GMMTransformer Integration

```python
class GMMTransformer(nn.Module):
    # Existing initialization...
    
    def forward(self, x, targets=None, flow_speed=None):
        """
        Process input through transformer model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            targets: Optional targets for dynamic flow speed prediction
            flow_speed: Optional explicit flow speed parameter
            
        Returns:
            Contextualized embeddings [batch_size, seq_len, hidden_dim]
        """
        batch_size = x.shape[0]
        
        # Use flow_speed predictor if available and no explicit flow_speed provided
        if flow_speed is None and targets is not None and hasattr(self, 'flow_predictor'):
            flow_speed = self.flow_predictor(targets, x)
        elif flow_speed is None:
            flow_speed = torch.ones(batch_size, device=x.device)
        
        # Project input to model dimension
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, flow_speed)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        return x
```

### 2.4 ClusterPredictionModel Integration

```python
class ClusterPredictionModel(nn.Module):
    """
    Extension of GMMTransformer for predicting cluster centers/assignments.
    """
    
    # Existing initialization...
    
    def forward(self, x, targets=None, flow_speed=None):
        """
        Process input and predict cluster information.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            targets: Optional targets for dynamic flow speed prediction
            flow_speed: Optional explicit flow speed parameter
            
        Returns:
            Predictions tensor:
            - For centers: [batch_size, seq_len, input_dim]
            - For assignments: [batch_size, seq_len, num_clusters]
        """
        # Get contextualized embeddings from transformer
        # Pass both targets and flow_speed to transformer
        embeddings = self.transformer(x, targets=targets, flow_speed=flow_speed)
        
        # Apply prediction head
        predictions = self.head(embeddings)
        
        # For assignments, apply softmax to get probabilities
        if self.prediction_type == "assignments":
            predictions = torch.softmax(predictions, dim=-1)
        
        return predictions

## 3. Model Configuration and Initialization

### 3.1 Flow Predictor Initialization in the Model

```python
class GMMTransformer(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        ff_expansion=4,
        bias=True,
        norm_eps=1e-5,
        use_flash_attn=True,
        use_flow_predictor=False,
        flow_predictor_type="dummy",
        flow_predictor_dim=1,
        flow_predictor_hidden_dim=32,
        flow_predictor_per_layer=False,
        flow_predictor_init_value=1.0,
        **kwargs
    ) -> None:
        """
        Initialize transformer model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of model embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function for feed-forward networks
            ff_expansion: Expansion factor for feed-forward networks
            bias: Whether to use bias in layers
            norm_eps: Epsilon for layer normalization
            use_flash_attn: Whether to use flash attention when available
            use_flow_predictor: Whether to use dynamic flow speed prediction
            flow_predictor_type: Type of flow predictor ("dummy", "mlp", etc.)
            flow_predictor_dim: Input dimension for flow predictor
            flow_predictor_hidden_dim: Hidden dimension for flow predictor
            flow_predictor_per_layer: Whether to use per-layer flow speeds
            flow_predictor_init_value: Initial flow speed value
        """
        super().__init__()
        
        # Standard transformer components...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Input projection from raw coordinates to model dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Dropout after input projection
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_expansion=ff_expansion,
                bias=bias,
                activation=activation,
                norm_eps=norm_eps,
                use_flash_attn=use_flash_attn
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = LayerNorm(hidden_dim, bias=bias, eps=norm_eps)
        
        # Flow predictor (optional)
        self.use_flow_predictor = use_flow_predictor
        if use_flow_predictor:
            if flow_predictor_type == "dummy":
                self.flow_predictor = DummyFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None
                )
                logger.info(f"Initialized dummy flow speed predictor (per_layer={flow_predictor_per_layer})")
            elif flow_predictor_type == "mlp":
                # Would implement a trainable MLP-based predictor here
                # This would take target features and output flow speed(s)
                pass
            else:
                logger.warning(f"Unknown flow predictor type: {flow_predictor_type}, using dummy")
                self.flow_predictor = DummyFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None
                )
        
        # Initialize weights
        self._init_weights()
        
        # Log model size
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GMMTransformer created with {num_params:,} parameters")
```

### 3.2 Training Loop Modifications

```python
def _train_epoch(self):
    """Train the model for one epoch."""
    self.model.train()
    epoch_loss = 0.0
    
    # Set up progress bar
    pbar = tqdm(
        total=len(self.train_loader),
        disable=not self.config.training.show_progress_bar,
        desc=f"Epoch {self.current_epoch+1}"
    )
    
    # Check if model.forward accepts targets parameter
    from inspect import signature
    model_signature = signature(self.model.forward)
    accepts_targets = 'targets' in model_signature.parameters
    
    # Training loop
    for i, batch in enumerate(self.train_loader):
        # Standard batch format: (inputs, targets)
        inputs = batch[0]
        targets = batch[1]
            
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.config.device.use_mixed_precision and self.scaler is not None:
            with autocast(device_type=self.device.type):
                # Pass targets to model only if it accepts them
                if accepts_targets:
                    outputs = self.model(inputs, targets=targets)
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
            # Rest of mixed precision training code...
            
        else:
            # Standard forward and backward pass
            if accepts_targets:
                outputs = self.model(inputs, targets=targets)
            else:
                outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Rest of standard training code...
        
        # Rest of the training epoch code...
```

> **Note**: The previous implementation supported dictionary-based batch formats where `batch` could be a dictionary containing 'points' and other metadata. This approach has been simplified to focus on the standard tuple format `(inputs, targets)` for clarity. If dictionary-based batches are needed, the code would need to be expanded to handle extraction of parameters and conversion to tensor formats.
```

### 3.3 Validation Loop Modifications

```python
def _validate_epoch(self):
    """Validate the model on the validation set."""
    if self.val_loader is None:
        return 0.0, {}
        
    self.model.eval()
    
    # Check if model.forward accepts targets parameter
    from inspect import signature
    model_signature = signature(self.model.forward)
    accepts_targets = 'targets' in model_signature.parameters
    
    # Create metrics tracker
    compare_with_kmeans = self.config.validation.metrics.compare_with_kmeans
    tracker = MetricsTracker(
        metric_fns=self.metric_fns,
        compare_with_kmeans=compare_with_kmeans,
        device=self.device,
        include_loss=True
    )
    
    # Validation loop
    with torch.no_grad():
        for batch in self.val_loader:
            # Standard batch format: (inputs, targets)
            inputs = batch[0]
            targets = batch[1]
                
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with targets if accepted
            if accepts_targets:
                outputs = self.model(inputs, targets=targets)
            else:
                outputs = self.model(inputs)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            batch_size = inputs.size(0)
            
            # Update metrics
            tracker.update_loss(loss.item(), batch_size)
            tracker.update(outputs, targets, inputs)

## 4. Configuration Updates

Add these options to your model configuration system:

```python
class ModelConfig:
    # Existing fields...
    use_flow_predictor: bool = False  # Whether to use flow speed predictor
    flow_predictor_type: str = "dummy"  # Type of flow predictor: "dummy", "mlp", etc.
    flow_predictor_dim: int = 1  # Dimension of flow predictor input
    flow_predictor_hidden_dim: int = 32  # Hidden dimension of flow predictor
    flow_predictor_per_layer: bool = False  # Whether to use per-layer flow speeds
    flow_predictor_init_value: float = 1.0  # Initial flow speed value (for learned predictors)
```

## 5. Implementation Stages

1. **Stage 1: Modify TransformerBlock**
   - Update the forward method to accept and use flow_speed parameter
   - Remove scale and beta parameters
   - Set default flow_speed to ones tensor
   - Handle both scalar and per-layer flow speeds

2. **Stage 2: Update GMMTransformer**
   - Add flow_predictor as an optional component
   - Modify forward method to accept targets and flow_speed
   - Add flow prediction logic
   - Pass flow_speed to blocks with appropriate indexing

3. **Stage 3: Update ClusterPredictionModel**
   - Pass targets and flow_speed to the internal transformer
   - Maintain the same interface for consistency

4. **Stage 4: Configuration**
   - Add flow predictor configuration options to ModelConfig
   - Document new parameters

5. **Stage 5: Implement Custom FlowSpeedPredictor (Optional)**
   - Develop specific predictor implementations based on application needs
   - Consider per-layer and per-component (attention vs. feedforward) flow control

## 6. Expected Benefits

1. **Adaptive Processing**: Flow speed adapts to the specific characteristics of each input batch
2. **Learned Dynamics**: The predictor learns optimal dynamics based on SNR or other parameters
3. **Inference Flexibility**: Explicit flow_speed can be provided at inference time for manual control
4. **Improved Performance**: Adaptive flow control should improve convergence and stability
5. **End-to-End Training**: Flow control is learned as part of the main training process

## 7. Diagnostic and Debugging

To verify the flow speed predictor is working correctly:

1. **Check Flow Speed Values**: Log flow speed values during training to ensure they're reasonable
2. **Validation Analysis**: Analyze correlation between flow speed and model performance
3. **Ablation Studies**: Compare models with and without adaptive flow speed
4. **Parameter Sensitivity**: Test sensitivity to flow predictor hyperparameters

```python
# Example debugging code
if self.config.training.debug_flow_speed and hasattr(self.model, 'flow_predictor'):
    with torch.no_grad():
        flow_values = self.model.flow_predictor(targets).cpu().numpy()
        logger.debug(f"Flow speeds: min={flow_values.min():.4f}, max={flow_values.max():.4f}, mean={flow_values.mean():.4f}")
```

## 8. Future Extensions

1. **Separate Flow Speeds**: Different flow speeds for attention and feed-forward networks
2. **Block-Specific Control**: Individualized flow control for each transformer block
3. **Attention-Head Control**: Flow control at the attention head level
4. **Temperature Annealing**: Gradually reduce flow speed during training
5. **Multi-Scale Control**: Multiple flow parameters controlling different aspects of the dynamics

## 9. References

- Gradient Flow in Recurrent Nets (Hochreiter et al.)
- Attention Is All You Need (Vaswani et al.)
- Dynamical Systems Perspective on Deep Learning (E et al.)