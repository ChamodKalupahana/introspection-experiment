"""
Script to compute a steering vector for 'dog' using contrastive activation pairs.

The steering vector is computed as:
    steering_vector = mean(positive_activations) - mean(negative_activations)

where positive prompts contain "dog" and negative prompts contain other animals.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from model_wrapper import ModelWrapper


def load_prompts(filepath: str) -> list:
    """Load prompt pairs from a JSONL file."""
    prompts = []
    with open(filepath, 'r') as f:
        for line in f:
            prompts.append(json.loads(line.strip()))
    return prompts


def compute_steering_vector(
    model: ModelWrapper,
    prompts: list,
    layer: int,
    position: str = "last",
) -> torch.Tensor:
    """
    Compute a steering vector from contrastive prompt pairs.
    
    Args:
        model: ModelWrapper instance
        prompts: List of dicts with 'positive' and 'negative' keys
        layer: Layer to extract activations from
        position: Which token position to use ('last', 'mean', or 'first')
    
    Returns:
        Steering vector tensor of shape (hidden_size,)
    """
    positive_activations = []
    negative_activations = []
    
    print(f"Computing activations for {len(prompts)} prompt pairs at layer {layer}...")
    
    for prompt_pair in tqdm(prompts):
        # Get positive activation
        pos_acts = model.get_activations(prompt_pair["positive"], layers=[layer])
        pos_hidden = pos_acts[layer]  # Shape: (1, seq_len, hidden_size)
        
        # Get negative activation
        neg_acts = model.get_activations(prompt_pair["negative"], layers=[layer])
        neg_hidden = neg_acts[layer]
        
        # Extract activation at specified position
        if position == "last":
            pos_vec = pos_hidden[0, -1, :]  # Last token
            neg_vec = neg_hidden[0, -1, :]
        elif position == "mean":
            pos_vec = pos_hidden[0].mean(dim=0)  # Mean over all tokens
            neg_vec = neg_hidden[0].mean(dim=0)
        elif position == "first":
            pos_vec = pos_hidden[0, 0, :]  # First token
            neg_vec = neg_hidden[0, 0, :]
        else:
            raise ValueError(f"Unknown position: {position}")
        
        positive_activations.append(pos_vec)
        negative_activations.append(neg_vec)
    
    # Stack and compute means
    positive_mean = torch.stack(positive_activations).mean(dim=0)
    negative_mean = torch.stack(negative_activations).mean(dim=0)
    
    # Compute steering vector as difference
    steering_vector = positive_mean - negative_mean
    
    return steering_vector


def test_steering_vector(
    model: ModelWrapper,
    steering_vector: torch.Tensor,
    layer: int,
    test_prompts: list,
    scales: list = [0.0, 1.0, 2.0, 5.0],
):
    """Test the steering vector with different scales."""
    print("\n" + "=" * 60)
    print("Testing steering vector effects")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        for scale in scales:
            model.clear_steering_vectors()
            
            if scale != 0:
                model.set_steering_vector(steering_vector, layer=layer, scale=scale)
            
            response = model.generate(prompt, max_new_tokens=50)
            print(f"Scale {scale:5.1f}: {response[:100]}...")
        
        model.clear_steering_vectors()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute a steering vector for 'dog'")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--prompts", default="data/dog_steering_prompts.jsonl",
                        help="Path to prompts JSONL file")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to extract steering vector from")
    parser.add_argument("--position", choices=["last", "mean", "first"], default="last",
                        help="Token position to use for activation extraction")
    parser.add_argument("--output", default="data/dog_steering_vector.pt",
                        help="Output path for steering vector")
    parser.add_argument("--test", action="store_true",
                        help="Test the steering vector after computing")
    args = parser.parse_args()
    
    # Load model
    print("=" * 60)
    print("Dog Steering Vector Computation")
    print("=" * 60)
    
    model = ModelWrapper(args.model)
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompt pairs from {args.prompts}")
    
    # Compute steering vector
    steering_vector = compute_steering_vector(
        model, 
        prompts, 
        layer=args.layer,
        position=args.position,
    )
    
    print(f"\nSteering vector computed:")
    print(f"  Shape: {steering_vector.shape}")
    print(f"  Norm: {steering_vector.norm().item():.4f}")
    print(f"  Mean: {steering_vector.mean().item():.6f}")
    print(f"  Std: {steering_vector.std().item():.6f}")
    
    # Save steering vector
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "steering_vector": steering_vector,
        "layer": args.layer,
        "position": args.position,
        "model_id": args.model,
        "num_prompts": len(prompts),
    }, output_path)
    
    print(f"\nSteering vector saved to {output_path}")
    
    # Test if requested
    if args.test:
        test_prompts = [
            "My favorite animal is a",
            "I saw a furry creature, it was a",
            "The pet store had many animals including",
        ]
        test_steering_vector(
            model, 
            steering_vector, 
            layer=args.layer,
            test_prompts=test_prompts,
            scales=[0.0, 1.0, 3.0, 5.0, 10.0],
        )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
