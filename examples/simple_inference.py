"""
Simple Inference Example (Single GPU / Debug)

Usage:
python examples/simple_inference.py --model-path PATH_TO_WEIGHTS_DIR --prompt "print(1+1)"

Note: Running this realistically requires loading the full model weights, which may consume a large amount of VRAM. This is for demonstration purposes.
"""
import argparse
import torch
from transformers import AutoTokenizer
from model.CustomQwen32B_hybrid import create_hybrid_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to pre-trained weights directory')
    parser.add_argument('--prompt', type=str, default='Write a Python function to add two numbers.')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # For demonstration only: Specify a small subset of layers to replace to reduce memory footprint
    model = create_hybrid_model(model_path=args.model_path, replace_layers=[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
