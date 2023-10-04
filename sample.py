import argparse
import pathlib
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from PIL import Image
from transformers import CLIPTokenizer

from minisd.models import pipeline, clip, vae, diffusion, samplers

def generate_image(prompt):
    device = 'mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'>>> Using device: {device}')
    
    tokenizer = CLIPTokenizer(vocab_file='data/tokenizer/vocab.json', merges_file='data/tokenizer/merges.txt')
    generator = torch.Generator(device=device)
    sampler = samplers.DDPMSampler(generator=generator, device=device)
    negative_prompt = ''
    pipe = pipeline.MiniSDPipeline(tokenizer=tokenizer, sampler=sampler)
    image = pipe(prompt, negative_prompt=negative_prompt)
    return plt.imshow(image)

def main():
    # parser = argparse.ArgumentParser(description='Generate images from a text prompt.')
    # parser.add_argument('--prompt', type=str, required=True, help='The text prompt for image generation.')
    
    # args = parser.parse_args()
    # prompt = args.prompt

    # print(f'Generating image for prompt: {prompt}')
    prompt = 'a photo of a happy dog'
    generate_image(prompt)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    sanitized_prompt = prompt.replace(' ', '_').replace('/', '').replace('\\', '')[:50]
    filename = f'{sanitized_prompt}_{timestamp}.png'
    path = pathlib.Path('data/output_data/')

    plt.savefig(path/filename)
    print(f'Image saved as {filename}')

if __name__ == '__main__':
    main()
