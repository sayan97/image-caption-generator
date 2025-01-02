import pickle
import torch
from torchvision import transforms
from models.architectures import EncoderCNN, DecoderRNN
from utils.vocabulary import Vocabulary
from config import Config
import __main__
setattr(__main__, 'Vocabulary', Vocabulary)

# Preprocessing pipeline
preprocess = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Resize(224), 
    transforms.CenterCrop(224), 
    transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                         (0.229, 0.224, 0.225))])


def load_vocab():
    print(inspect.currentframe().f_globals) 
    vocab = Vocabulary()
    with open(Config.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_models():
    vocab = Vocabulary()
    with open(Config.VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    encoder = EncoderCNN()
    #encoder.load_state_dict(torch.load(Config.ENCODER_MODEL_PATH))
    encoder.eval()

    decoder = DecoderRNN(vocab_size=len(vocab), embed_size=Config.EMBED_SIZE, hidden_size=Config.HIDDEN_SIZE)
    decoder.load_state_dict(torch.load(Config.DECODER_MODEL_PATH,  map_location=torch.device('cpu'))['state_dict'])
    decoder.eval()

    return encoder, decoder, vocab

def generate_caption_from_image(encoder, decoder, vocab, image):
    image_tensor = preprocess(image).unsqueeze(0)
    features = encoder(image_tensor)
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<end>':
            break
        caption.append(word)
    return ' '.join(caption)
