"""
Simple experiment to load and test Meta-Llama-3.1-8B-Instruct from Hugging Face.

Prerequisites:
1. Request access to the model at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Login to Hugging Face CLI: huggingface-cli login
   (You'll need a HuggingFace access token from https://huggingface.co/settings/tokens)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model():
    """Load the Llama 3.1 8B Instruct model and tokenizer."""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print("Model loaded successfully!")
    return model, tokenizer


def test_generation(model, tokenizer):
    """Test the model with a simple prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2? Answer in one word."},
    ]
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    print("\nGenerating response...")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response (only the new tokens)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    print(f"Prompt: What is 2 + 2? Answer in one word.")
    print(f"Response: {response}")
    
    return response


def main():
    print("=" * 50)
    print("Llama 3.1 8B Instruct - Loading Test")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model, tokenizer = load_model()
    
    # Test generation
    response = test_generation(model, tokenizer)
    
    print("\n" + "=" * 50)
    print("✓ Model loaded and verified successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
