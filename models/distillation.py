import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Loss function for knowledge distillation, combining soft targets from the teacher
    and hard targets from the ground truth.
    """
    def __init__(self, temperature=2.0, alpha=0.5):
        """
        Initialize the distillation loss.
        
        Args:
            temperature (float): Temperature parameter for softening the distributions.
            alpha (float): Weight for balancing soft and hard targets (0-1).
                           alpha=0 means only hard targets, alpha=1 means only soft targets.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_logits, teacher_logits, labels=None):
        """
        Compute the distillation loss.
        
        Args:
            student_logits (torch.Tensor): Logits from the student model [batch_size, seq_len, vocab_size].
            teacher_logits (torch.Tensor): Logits from the teacher model [batch_size, seq_len, vocab_size].
            labels (torch.Tensor, optional): Ground truth labels for hard targets [batch_size, seq_len].
            
        Returns:
            torch.Tensor: The computed distillation loss.
        """
        # If teacher_logits and student_logits have different sequence lengths, align them
        if teacher_logits.shape[1] != student_logits.shape[1]:
            min_seq_len = min(teacher_logits.shape[1], student_logits.shape[1])
            teacher_logits = teacher_logits[:, :min_seq_len, :]
            student_logits = student_logits[:, :min_seq_len, :]
            if labels is not None:
                labels = labels[:, :min_seq_len]
        
        # Compute soft targets loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob, dim=-1).mean()
        soft_targets_loss = soft_targets_loss * (self.temperature ** 2)
        
        # If no hard targets provided, only use soft targets
        if labels is None:
            return soft_targets_loss
        
        # Compute hard targets loss using cross-entropy
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,  # Mask padding tokens
        )
        
        # Combine soft and hard losses
        loss = self.alpha * soft_targets_loss + (1 - self.alpha) * hard_loss
        
        return loss


class FeatureDistillationLoss(nn.Module):
    """
    Loss function for intermediate feature distillation between 
    teacher and student models.
    """
    def __init__(self, normalize=True, student_dim=768, teacher_dim=1024):
        """
        Initialize the feature distillation loss.
        
        Args:
            normalize (bool): Whether to normalize the features before computing the loss.
            student_dim (int): Hidden dimension of student model features.
            teacher_dim (int): Hidden dimension of teacher model features.
        """
        super().__init__()
        self.normalize = normalize
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # Create projection layer if dimensions don't match
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
            # Initialize with identity-like mapping for better starting point
            nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        else:
            self.projection = None
    
    def forward(self, student_features, teacher_features):
        """
        Compute the feature distillation loss.
        
        Args:
            student_features (torch.Tensor or list or tuple): Features from the student model.
            teacher_features (torch.Tensor or list or tuple): Features from the teacher model.
            
        Returns:
            torch.Tensor: The computed feature distillation loss.
        """
        # Check if the features are lists or tuples (e.g., hidden states from different layers)
        if (isinstance(student_features, (list, tuple)) and 
            isinstance(teacher_features, (list, tuple))):
            total_loss = 0
            # Use only the features that are present in both student and teacher
            min_length = min(len(student_features), len(teacher_features))
            for i in range(min_length):
                s_feat = student_features[i]
                t_feat = teacher_features[i]
                # Skip None values
                if s_feat is None or t_feat is None:
                    continue
                total_loss += self._compute_loss(s_feat, t_feat)
            return total_loss / min_length if min_length > 0 else torch.tensor(0.0, device=student_features[0].device)
        else:
            return self._compute_loss(student_features, teacher_features)
    
    def _compute_loss(self, student_feat, teacher_feat):
        """Helper method to compute the loss between two feature tensors."""
        # Ensure both inputs are tensors
        if not isinstance(student_feat, torch.Tensor) or not isinstance(teacher_feat, torch.Tensor):
            return torch.tensor(0.0, device=teacher_feat.device if isinstance(teacher_feat, torch.Tensor) else 
                                           (student_feat.device if isinstance(student_feat, torch.Tensor) else 'cpu'))
        
        # Get the shapes
        student_shape = student_feat.shape
        teacher_shape = teacher_feat.shape
        
        # Debug print
        print(f"Student feature shape: {student_shape}, Teacher feature shape: {teacher_shape}")
        
        # Handle the case where both sequence length and hidden dimensions differ
        if len(student_shape) == len(teacher_shape) == 3:  # [batch_size, seq_len, hidden_dim]
            # First handle sequence length difference
            if student_shape[1] != teacher_shape[1]:
                print(f"Adjusting sequence length: {student_shape[1]} -> {teacher_shape[1]}")
                # Interpolate to match sequence length
                student_feat = F.interpolate(
                    student_feat.transpose(1, 2).contiguous(),  # [batch, hidden_dim, seq_len]
                    size=teacher_shape[1],
                    mode='linear'
                ).transpose(1, 2).contiguous()  # [batch, seq_len, hidden_dim]
                print(f"After sequence adjustment: {student_feat.shape}")
            
            # Then handle hidden dimension difference
            if student_shape[2] != teacher_shape[2] or student_feat.shape[2] != teacher_shape[2]:
                print(f"Projecting hidden dimension: {student_feat.shape[2]} -> {teacher_shape[2]}")
                # Create or update projection layer if needed
                if self.projection is None or self.projection.in_features != student_feat.shape[2] or self.projection.out_features != teacher_shape[2]:
                    self.projection = nn.Linear(student_feat.shape[2], teacher_shape[2], bias=False).to(student_feat.device)
                    nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
                elif self.projection.weight.device != student_feat.device:
                    # Ensure the projection layer is on the same device as student_feat
                    self.projection = self.projection.to(student_feat.device)
                
                # Apply projection - safer approach to avoid reshape issues
                batch_size, seq_len = student_feat.shape[:2]
                # Flatten batch and sequence dimensions
                flat_student = student_feat.reshape(-1, student_feat.shape[-1])
                # Project
                flat_projected = self.projection(flat_student)
                # Reshape back
                student_feat = flat_projected.reshape(batch_size, seq_len, -1)
                print(f"After projection: {student_feat.shape}")
        
        # Handle 2D tensors (e.g., pooled outputs) with different hidden dimensions
        elif len(student_shape) == len(teacher_shape) == 2 and student_shape[1] != teacher_shape[1]:
            print(f"Projecting 2D tensor: {student_shape[1]} -> {teacher_shape[1]}")
            if self.projection is None or self.projection.in_features != student_shape[1] or self.projection.out_features != teacher_shape[1]:
                self.projection = nn.Linear(student_shape[1], teacher_shape[1], bias=False).to(student_feat.device)
                nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
            elif self.projection.weight.device != student_feat.device:
                # Ensure the projection layer is on the same device as student_feat
                self.projection = self.projection.to(student_feat.device)
            student_feat = self.projection(student_feat)
            print(f"After 2D projection: {student_feat.shape}")
        
        # Handle 4D tensors (e.g., image features) with different spatial dimensions
        elif len(student_shape) == len(teacher_shape) == 4:
            # Handle spatial dimensions (height, width)
            if student_shape[2:] != teacher_shape[2:]:
                print(f"Adjusting spatial dimensions: {student_shape[2:]} -> {teacher_shape[2:]}")
                student_feat = F.interpolate(
                    student_feat,
                    size=teacher_shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                print(f"After spatial adjustment: {student_feat.shape}")
            
            # Handle channel dimension
            if student_shape[1] != teacher_shape[1]:
                print(f"Projecting channel dimension: {student_shape[1]} -> {teacher_shape[1]}")
                if self.projection is None or self.projection.in_features != student_shape[1] or self.projection.out_features != teacher_shape[1]:
                    self.projection = nn.Conv2d(student_shape[1], teacher_shape[1], kernel_size=1, bias=False).to(student_feat.device)
                    nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
                elif self.projection.weight.device != student_feat.device:
                    # Ensure the projection layer is on the same device as student_feat
                    self.projection = self.projection.to(student_feat.device)
                student_feat = self.projection(student_feat)
                print(f"After channel projection: {student_feat.shape}")
        
        # Normalize features if required
        if self.normalize:
            student_feat = F.normalize(student_feat, p=2, dim=-1)
            teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        
        # Final shape check before computing loss
        print(f"Final shapes - Student: {student_feat.shape}, Teacher: {teacher_feat.shape}")
        
        # Compute MSE loss
        try:
            loss = F.mse_loss(student_feat, teacher_feat)
            return loss
        except Exception as e:
            print(f"Error computing MSE loss: {e}")
            # If we still have shape mismatch, log detailed information and return a dummy loss
            print(f"Shape mismatch still exists. Student: {student_feat.shape}, Teacher: {teacher_feat.shape}")
            print("Returning dummy loss to avoid training failure")
            return torch.tensor(0.0, device=student_feat.device)


class AttentionDistillationLoss(nn.Module):
    """
    Loss function for distilling attention maps from the teacher to the student.
    """
    def __init__(self):
        """
        Initialize the attention distillation loss.
        """
        super().__init__()
        
    def forward(self, student_attentions, teacher_attentions):
        """
        Compute the attention distillation loss.
        
        Args:
            student_attentions (list): List of attention maps from the student model.
            teacher_attentions (list): List of attention maps from the teacher model.
            
        Returns:
            torch.Tensor: The computed attention distillation loss.
        """
        total_loss = 0
        count = 0
        
        # If the number of layers is different, select a subset
        num_student_layers = len(student_attentions)
        num_teacher_layers = len(teacher_attentions)
        
        if num_student_layers < num_teacher_layers:
            # Select teacher layers spaced evenly
            teacher_indices = torch.linspace(0, num_teacher_layers - 1, num_student_layers).long()
            selected_teacher_attentions = [teacher_attentions[i] for i in teacher_indices]
            paired_attentions = zip(student_attentions, selected_teacher_attentions)
        else:
            # Use all teacher layers and match with corresponding student layers
            student_indices = torch.linspace(0, num_student_layers - 1, num_teacher_layers).long()
            selected_student_attentions = [student_attentions[i] for i in student_indices]
            paired_attentions = zip(selected_student_attentions, teacher_attentions)
        
        # Compute KL divergence between attention maps
        for student_attn, teacher_attn in paired_attentions:
            # Ensure same number of attention heads by averaging if needed
            if student_attn.shape[1] != teacher_attn.shape[1]:
                if student_attn.shape[1] < teacher_attn.shape[1]:
                    # Average teacher attention heads
                    teacher_attn = teacher_attn.view(
                        teacher_attn.shape[0],
                        -1,
                        student_attn.shape[1],
                        teacher_attn.shape[2],
                        teacher_attn.shape[3]
                    ).mean(dim=1)
                else:
                    # This case is less common but included for completeness
                    student_attn = student_attn.view(
                        student_attn.shape[0],
                        -1,
                        teacher_attn.shape[1],
                        student_attn.shape[2],
                        student_attn.shape[3]
                    ).mean(dim=1)
            
            # Ensure same sequence length by interpolation if needed
            if student_attn.shape[2] != teacher_attn.shape[2] or student_attn.shape[3] != teacher_attn.shape[3]:
                student_attn = F.interpolate(
                    student_attn.flatten(0, 1),
                    size=(teacher_attn.shape[2], teacher_attn.shape[3]),
                    mode='bilinear'
                ).view(student_attn.shape[0], student_attn.shape[1], teacher_attn.shape[2], teacher_attn.shape[3])
            
            # Normalize attention maps to form probability distributions
            student_attn = F.softmax(student_attn, dim=-1)
            teacher_attn = F.softmax(teacher_attn, dim=-1)
            
            # Compute KL divergence
            loss = F.kl_div(
                student_attn.log(),
                teacher_attn,
                reduction='batchmean',
                log_target=False
            )
            
            total_loss += loss
            count += 1
        
        return total_loss / count if count > 0 else 0


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss function for knowledge distillation, combining logits distillation,
    feature distillation, and optionally attention distillation.
    """
    def __init__(
        self,
        temperature=2.0,
        alpha=0.5,
        lambda_logits=1.0,
        lambda_feature=0.5,
        lambda_attn=0.5,
        use_feature_distillation=True,
        use_attn_distillation=True,
        student_dim=768,
        teacher_dim=1024,
    ):
        """
        Initialize the combined distillation loss.
        
        Args:
            temperature (float): Temperature for softening logits.
            alpha (float): Weight for balancing soft and hard targets in logits distillation.
            lambda_logits (float): Weight for logits distillation loss.
            lambda_feature (float): Weight for feature distillation loss.
            lambda_attn (float): Weight for attention distillation loss.
            use_feature_distillation (bool): Whether to use feature distillation.
            use_attn_distillation (bool): Whether to use attention distillation.
            student_dim (int): Hidden dimension of student model features.
            teacher_dim (int): Hidden dimension of teacher model features.
        """
        super().__init__()
        self.logits_loss = DistillationLoss(temperature=temperature, alpha=alpha)
        self.feature_loss = FeatureDistillationLoss(
            normalize=True, 
            student_dim=student_dim, 
            teacher_dim=teacher_dim
        ) if use_feature_distillation else None
        self.attn_loss = AttentionDistillationLoss() if use_attn_distillation else None
        
        self.lambda_logits = lambda_logits
        self.lambda_feature = lambda_feature
        self.lambda_attn = lambda_attn
        
        # Save configuration for logging
        self.config = {
            "temperature": temperature,
            "alpha": alpha,
            "lambda_logits": lambda_logits,
            "lambda_feature": lambda_feature,
            "lambda_attn": lambda_attn,
            "use_feature_distillation": use_feature_distillation,
            "use_attn_distillation": use_attn_distillation,
            "student_dim": student_dim,
            "teacher_dim": teacher_dim,
        }
        
    def forward(
        self,
        student_outputs,
        teacher_outputs,
        labels=None,
    ):
        """
        Compute the combined distillation loss.
        
        Args:
            student_outputs: Outputs from the student model.
            teacher_outputs: Outputs from the teacher model.
            labels (torch.Tensor, optional): Ground truth labels.
            
        Returns:
            dict: Dictionary containing the total loss and individual component losses.
        """
        losses = {}
        
        # Logits distillation
        logits_loss = self.logits_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels
        )
        losses["logits_loss"] = logits_loss
        
        # Feature distillation (hidden states)
        if self.feature_loss is not None and hasattr(student_outputs, "hidden_states") and hasattr(teacher_outputs, "hidden_states"):
            feature_loss = self.feature_loss(
                student_outputs.hidden_states,
                teacher_outputs.hidden_states
            )
            losses["feature_loss"] = feature_loss
        else:
            feature_loss = 0
            losses["feature_loss"] = torch.tensor(0.0, device=logits_loss.device)
        
        # Attention distillation
        if self.attn_loss is not None and hasattr(student_outputs, "attentions") and hasattr(teacher_outputs, "attentions"):
            attn_loss = self.attn_loss(
                student_outputs.attentions,
                teacher_outputs.attentions
            )
            losses["attn_loss"] = attn_loss
        else:
            attn_loss = 0
            losses["attn_loss"] = torch.tensor(0.0, device=logits_loss.device)
        
        # Compute weighted sum for total loss
        total_loss = (
            self.lambda_logits * logits_loss +
            self.lambda_feature * feature_loss +
            self.lambda_attn * attn_loss
        )
        losses["total_loss"] = total_loss
        
        return losses
