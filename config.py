import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Paths to models
    ENCODER_MODEL_PATH = os.path.join(BASE_DIR,  os.path.join('models','encoder.pth'))
    DECODER_MODEL_PATH = os.path.join(BASE_DIR, os.path.join('models','decoder.pt'))

    # Path to vocabulary file
    VOCAB_PATH = os.path.join(BASE_DIR, os.path.join('models','vocab.pkl'))

    # Default embedding and hidden sizes
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    MAX_SEQ_LENGTH = 47

    # Image Configuration
    UPLOAD_FOLDER = os.path.join('static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}