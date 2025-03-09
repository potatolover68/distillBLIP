import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
from transformers.models.blip.modeling_blip import BlipTextModel, BlipConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithCrossAttentions

class DistilledBLIPConfig:
    """Configuration for the distilled BLIP model."""
    def __init__(
        self,
        vision_model_name="google/vit-base-patch16-224-in21k",
        text_model_name="bert-base-uncased",
        vision_hidden_size=768,
        text_hidden_size=768,
        cross_attention_dim=768,
        num_visual_encoder_layers=6,  # Reduced from 12 in original ViT
        num_text_encoder_layers=6,    # Reduced from 12 in original BERT
        num_text_decoder_layers=6,    # Reduced from 12 in original BERT
        num_attention_heads=8,        # Reduced from 12 in original
        intermediate_size=2048,       # Reduced from 3072 in original
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        vocab_size=30522,             # BERT vocab size
        use_cache=True,
        pad_token_id=0,
        bos_token_id=101,
        eos_token_id=102,
        sep_token_id=102,
        decoder_start_token_id=101,
    ):
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_visual_encoder_layers = num_visual_encoder_layers
        self.num_text_encoder_layers = num_text_encoder_layers
        self.num_text_decoder_layers = num_text_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.decoder_start_token_id = decoder_start_token_id


class DistilledVisionEncoder(nn.Module):
    """A distilled version of the ViT model for vision encoding."""
    def __init__(self, config):
        super().__init__()
        # Create a smaller ViT model
        vit_config = ViTConfig.from_pretrained(config.vision_model_name)
        vit_config.num_hidden_layers = config.num_visual_encoder_layers
        vit_config.hidden_size = config.vision_hidden_size
        vit_config.intermediate_size = config.intermediate_size
        vit_config.num_attention_heads = config.num_attention_heads
        vit_config.hidden_dropout_prob = config.hidden_dropout_prob
        vit_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        
        self.vision_model = ViTModel(vit_config)
        
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DistilledTextDecoder(nn.Module):
    """A distilled version of the BERT model for text generation."""
    def __init__(self, config):
        super().__init__()
        # Create a smaller BERT-based text model for the decoder
        blip_config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_config.num_hidden_layers = config.num_text_decoder_layers
        blip_config.hidden_size = config.text_hidden_size
        blip_config.intermediate_size = config.intermediate_size
        blip_config.num_attention_heads = config.num_attention_heads
        blip_config.hidden_dropout_prob = config.hidden_dropout_prob
        blip_config.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        
        self.text_model = BlipTextModel(blip_config)
        
        # Add projection layer for vision features
        self.visual_projection = nn.Linear(
            config.vision_hidden_size, config.text_hidden_size
        )
        
        # Add cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.text_hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            for _ in range(config.num_text_decoder_layers)
        ])
        
        # Add layer norm
        self.cross_layer_norm = nn.ModuleList([
            nn.LayerNorm(config.text_hidden_size)
            for _ in range(config.num_text_decoder_layers)
        ])
        
        # Add feed-forward networks after cross-attention
        self.cross_feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.text_hidden_size, config.intermediate_size),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.text_hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
            for _ in range(config.num_text_decoder_layers)
        ])
        
        # Add layer norm for FFN
        self.ffn_layer_norm = nn.ModuleList([
            nn.LayerNorm(config.text_hidden_size)
            for _ in range(config.num_text_decoder_layers)
        ])
        
        # LM head
        self.lm_head = nn.Linear(config.text_hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.text_model.embeddings.word_embeddings.weight
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        text_features = text_outputs.last_hidden_state
        
        # Project visual features
        projected_visual_features = self.visual_projection(encoder_hidden_states)
        
        # Apply cross-attention between text and visual features
        hidden_states = text_features
        for i in range(len(self.cross_attention)):
            residual = hidden_states
            
            # Apply cross-attention
            hidden_states = self.cross_layer_norm[i](hidden_states)
            hidden_states, _ = self.cross_attention[i](
                query=hidden_states,
                key=projected_visual_features,
                value=projected_visual_features,
                key_padding_mask=~encoder_attention_mask.bool() if encoder_attention_mask is not None else None,
            )
            hidden_states = residual + hidden_states
            
            # Apply feed-forward network
            residual = hidden_states
            hidden_states = self.ffn_layer_norm[i](hidden_states)
            hidden_states = self.cross_feedforward[i](hidden_states)
            hidden_states = residual + hidden_states
        
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
            cross_attentions=None,
        )


class DistilledBLIPForConditionalGeneration(nn.Module):
    """Distilled BLIP model for image captioning."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize the vision and text components
        self.vision_encoder = DistilledVisionEncoder(config)
        self.text_decoder = DistilledTextDecoder(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Get vision encoder outputs
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        # Create image attention mask (assumes all image features are attended to)
        batch_size = image_embeds.shape[0]
        seq_length = image_embeds.shape[1]
        encoder_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=image_embeds.device)
        
        # Pass through text decoder with visual features
        decoder_outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        return decoder_outputs
    
    def generate(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
        max_length=30,
        num_beams=4,
        **generate_kwargs
    ):
        """
        Generate text conditioned on image inputs.
        
        Args:
            pixel_values (torch.Tensor): The pixel values of the images
            input_ids (torch.Tensor, optional): Optional input prompt IDs
            attention_mask (torch.Tensor, optional): Attention mask for input_ids
            max_length (int): Maximum length for generation
            num_beams (int): Number of beams for beam search
            **generate_kwargs: Additional kwargs for generation
            
        Returns:
            torch.Tensor: Generated text token IDs
        """
        # Get vision encoder outputs
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        # Create image attention mask (assumes all image features are attended to)
        batch_size = image_embeds.shape[0]
        seq_length = image_embeds.shape[1]
        encoder_attention_mask = torch.ones(
            (batch_size, seq_length), dtype=torch.long, device=image_embeds.device
        )
        
        # Set up for text generation
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": encoder_attention_mask,
        }
        
        # If no input_ids are provided, use BOS token
        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1),
                self.config.bos_token_id,
                dtype=torch.long,
                device=pixel_values.device,
            )
        
        # Simple generator function
        def _prepare_attention_mask_for_generation(attention_mask, input_shape, device):
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                expanded_attn_mask = _expand_mask(attention_mask, dtype=torch.float32)
                expanded_attn_mask = expanded_attn_mask.to(device)
                return expanded_attn_mask
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                return torch.ones((input_shape[0], 1, input_shape[1], input_shape[1]), device=device)
        
        generated_tokens = []
        cur_ids = input_ids
        
        # Simple greedy search (as a placeholder)
        # In a real implementation, you'd use the HF generate() functionality
        for _ in range(max_length):
            outputs = self.text_decoder(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=encoder_attention_mask,
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            generated_tokens.append(next_token.unsqueeze(1))
            
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=1
                )
            
            # Stop if all sequences have EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return torch.cat([input_ids, torch.cat(generated_tokens, dim=1)], dim=1)
