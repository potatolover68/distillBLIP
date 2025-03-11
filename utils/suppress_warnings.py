"""
Utility module to suppress warnings in the project.
"""
import warnings
import logging
import os

def suppress_all_warnings():
    """
    Suppress all warnings in the project.
    """
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Suppress specific common warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Suppress PyTorch specific warnings
    warnings.filterwarnings("ignore", message=".*User provided device_type of 'cuda'.*")
    warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node.*")
    warnings.filterwarnings("ignore", message=".*Consider setting.*")
    warnings.filterwarnings("ignore", message=".*Set TORCH_LOGS.*")
    
    # Suppress transformers warnings
    warnings.filterwarnings("ignore", message=".*The model is automatically converting to fp16.*")
    
    # Set environment variables to suppress other warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Suppress logging below warning level
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
