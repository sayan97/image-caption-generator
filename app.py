from flask import Flask, render_template, request, flash, redirect, url_for
import os
from config import Config
from utils.caption import generate_caption
from utils.vocabulary import Vocabulary

app=Flask(__name__)
app.secret_key = "any random string"

# Configuration
#ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def get_caption():
    try:
        #loading input image
        image = request.files['image']
        if image.filename == '':
            flash('No image found!', 'danger')
            return redirect(url_for('index'))

        #checking extension of image
        extension = image.filename.rsplit('.', 1)[1].lower()
        if extension in Config.ALLOWED_EXTENSIONS:
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Save the image to the upload folder
            image_path = os.path.join(upload_folder, 'input_image' + '.' + extension)
            image.save(image_path)
            
            # Generate caption
            caption = generate_caption(image_path) 
            
            return render_template('result.html', caption=caption, input_image=image_path)

            
        else:
            flash('Upload image in the given format - png/jpg/jpeg', 'danger')
            return redirect(url_for('index'))

    except Exception:
        flash('Something went wrong', 'danger')
        return redirect(url_for('index'))


if __name__=='__main__':
    print(os.getcwd())
    app.run()

