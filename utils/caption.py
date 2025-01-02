from PIL import Image
from utils.inference import load_models, generate_caption_from_image

encoder, decoder, vocab = load_models()

def generate_caption(image_file):
    image = Image.open(image_file).convert("RGB")
    caption = generate_caption_from_image(encoder, decoder, vocab, image)
    return caption