import argparse
import pathlib
import matplotlib.pyplot as plt
from datetime import datetime

def generate_image(prompt):
    pass

def main():
    parser = argparse.ArgumentParser(description='Generate images from a text prompt.')
    parser.add_argument('--prompt', type=str, required=True, help='The text prompt for image generation.')
    
    args = parser.parse_args()
    prompt = args.prompt

    print(f'Generating image for prompt: {prompt}')
    generate_image(prompt)
    
    timestamp = datetime.now().strftime('%Y%m%d')
    sanitized_prompt = prompt.replace(' ', '_').replace('/', '').replace('\\', '')[:50]
    filename = f'{sanitized_prompt}_{timestamp}.png'
    path = pathlib.Path('data/output_data/')

    plt.savefig(path/filename)
    print(f'Image saved as {filename}')

if __name__ == '__main__':
    main()
