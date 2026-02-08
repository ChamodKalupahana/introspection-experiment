"""
Reusable model wrapper for HuggingFace models with steering vector support.

Usage:
    # Basic generation
    model = ModelWrapper("meta-llama/Meta-Llama-3.1-8B-Instruct")
    response = model.generate("Hello, how are you?")
    
    # With steering vector
    steering_vector = torch.randn(4096)  # Match hidden size
    model.set_steering_vector(steering_vector, layer=15, head=0)
    steered_response = model.generate("Hello, how are you?")
    model.clear_steering_vector()
"""

import torch
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class SteeringHook:
    """Hook that adds a steering vector to attention head outputs."""
    
    def __init__(self, steering_vector: torch.Tensor, head: Optional[int] = None, scale: float = 1.0):
        """
        Args:
            steering_vector: Vector to add to activations. Shape should be (hidden_size,) 
                           or (head_dim,) if targeting a specific head.
            head: If specified, only modify this attention head. If None, apply to all.
            scale: Multiplier for the steering vector.
        """
        self.steering_vector = steering_vector
        self.head = head
        self.scale = scale
        self.handle = None
    
    def __call__(self, module, input, output):
        """Hook function called after the layer's forward pass."""
        # output is typically a tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Add steering vector to hidden states
        # hidden_states shape: (batch, seq_len, hidden_size)
        steering = self.steering_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        
        if self.head is not None:
            # For head-specific steering, we'd need access to the attention internals
            # For now, we apply to the full hidden states
            # TODO: Implement head-specific steering if needed
            hidden_states = hidden_states + (steering * self.scale)
        else:
            hidden_states = hidden_states + (steering * self.scale)
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


class ModelWrapper:
    """Wrapper for HuggingFace models with steering vector support."""
    
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
            device_map: Device placement strategy ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        self.model_id = model_id
        
        print(f"Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print(f"Loading model from {model_id}...")
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Attention heads: {self.num_attention_heads}")
        
        self._steering_hooks: List[SteeringHook] = []
        self._hook_handles = []
    
    @property
    def num_layers(self) -> int:
        """Get the number of transformer layers."""
        return self.model.config.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.model.config.hidden_size
    
    @property
    def num_attention_heads(self) -> int:
        """Get the number of attention heads."""
        return self.model.config.num_attention_heads
    
    @property
    def head_dim(self) -> int:
        """Get the dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.model.parameters()).device
    
    def get_layer(self, layer_idx: int):
        """Get a specific transformer layer by index."""
        if hasattr(self.model, 'model'):
            # Llama-style models
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer'):
            # GPT-style models
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture: {type(self.model)}")
    
    def set_steering_vector(
        self,
        steering_vector: torch.Tensor,
        layer: int,
        head: Optional[int] = None,
        scale: float = 1.0,
    ) -> None:
        """
        Add a steering vector to be injected at a specific layer.
        
        Args:
            steering_vector: Vector to add to activations. Shape: (hidden_size,)
            layer: Layer index to inject the steering vector at
            head: Optional specific attention head to target (not yet implemented)
            scale: Multiplier for the steering vector strength
        """
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {self.num_layers})")
        
        if steering_vector.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Steering vector size {steering_vector.shape[-1]} does not match "
                f"model hidden size {self.hidden_size}"
            )
        
        # Create hook
        hook = SteeringHook(steering_vector, head=head, scale=scale)
        
        # Register hook on the layer
        target_layer = self.get_layer(layer)
        handle = target_layer.register_forward_hook(hook)
        
        self._steering_hooks.append(hook)
        self._hook_handles.append(handle)
        
        print(f"Steering vector set at layer {layer}" + (f", head {head}" if head else "") + f" (scale={scale})")
    
    def clear_steering_vectors(self) -> None:
        """Remove all steering vector hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._steering_hooks.clear()
        self._hook_handles.clear()
        print("All steering vectors cleared")
    
    def generate(
        self,
        prompt: Union[str, List[dict]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Either a string prompt or a list of chat messages
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy decoding)
            system_prompt: Optional system prompt (only used if prompt is a string)
        
        Returns:
            Generated text response
        """
        # Build messages list
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        
        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def get_activations(
        self,
        prompt: Union[str, List[dict]],
        layers: Optional[List[int]] = None,
    ) -> dict:
        """
        Get intermediate activations for a prompt.
        
        Args:
            prompt: Input prompt or messages
            layers: List of layer indices to capture (default: all)
        
        Returns:
            Dictionary mapping layer index to activation tensors
        """
        if layers is None:
            layers = list(range(self.num_layers))
        
        activations = {}
        handles = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[layer_idx] = output[0].detach().cpu()
                else:
                    activations[layer_idx] = output.detach().cpu()
            return hook
        
        # Register hooks
        for layer_idx in layers:
            layer = self.get_layer(layer_idx)
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)
        
        try:
            # Build messages
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
            
            # Tokenize and run forward pass
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
        
        return activations


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Model Wrapper Demo")
    print("=" * 60)
    
    # Load model
    model = ModelWrapper("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Test normal generation
    print("\n--- Normal Generation ---")
    response = model.generate("What is the capital of France? Answer in one word.")
    print(f"Response: {response}")
    
    # Test with steering vector (random vector for demo)
    print("\n--- With Steering Vector ---")
    steering = torch.randn(model.hidden_size) * 0.1
    model.set_steering_vector(steering, layer=15, scale=100.0)
    
    steered_response = model.generate("What is the capital of France? Answer in one word.")
    print(f"Steered Response: {steered_response}")
    
    model.clear_steering_vectors()
    
    # Test activation extraction
    print("\n--- Activation Extraction ---")
    activations = model.get_activations("Hello world", layers=[0, 15, 31])
    for layer_idx, act in activations.items():
        print(f"Layer {layer_idx}: shape = {act.shape}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
