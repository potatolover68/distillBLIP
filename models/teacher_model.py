import torch
from torch import nn
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIPTeacherModel(nn.Module):
    """
    Wrapper around the pretrained BLIP model to be used as the teacher
    in the knowledge distillation process.
    """
    def __init__(self, model_path="Salesforce/blip-image-captioning-large"):
        """
        Initialize the BLIP teacher model.
        
        Args:
            model_path (str): Path or name of the pretrained BLIP model.
        """
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        
        # Freeze the teacher model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """
        Forward pass through the BLIP model.
        
        Args:
            pixel_values (torch.Tensor): Batch of images.
            input_ids (torch.Tensor, optional): Input text tokens.
            attention_mask (torch.Tensor, optional): Attention mask for text.
            
        Returns:
            dict: Dictionary containing the output logits and other relevant information.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True
        )
        
        return outputs
    
    def prepare_inputs(self, images, text=None):
        """
        Prepare inputs for the BLIP model using the processor.
        
        Args:
            images: Raw images to process.
            text (str, optional): Conditional text for captioning.
            
        Returns:
            dict: Dictionary of processed inputs.
        """
        if text:
            inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(images=images, return_tensors="pt")
        
        return inputs
    
    def generate_captions(self, pixel_values, input_ids=None, **generate_kwargs):
        """
        Generate image captions.
        
        Args:
            pixel_values (torch.Tensor): Batch of processed images.
            input_ids (torch.Tensor, optional): Optional prompt tokens.
            **generate_kwargs: Additional arguments for the generation process.
            
        Returns:
            list: List of generated captions.
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                **generate_kwargs
            )
            
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return captions
