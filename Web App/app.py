from flask import Flask, request, render_template, url_for, send_from_directory
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import pytesseract
import requests
import easyocr
import re
from collections import defaultdict

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DETECTION_FOLDER'] = 'static/detections'
app.config['CROPPED_FOLDER'] = 'static/cropped_images'

# Ensure necessary directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['DETECTION_FOLDER'], app.config['CROPPED_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

ROBOFLOW_API_KEY = "oP0MuGH0hDPwpzq7Zz2X"
ROBOFLOW_MODEL = "ecr-kddvp"
ROBOFLOW_VERSION = "1"
ROBOFLOW_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        result_image_path, extracted_text, cropped_images = detect_objects(filepath)

        # Debug print to check the values being returned
        print(f"Result Image Path: {result_image_path}")
        print(f"Extracted Text: {extracted_text}")
        print(f"Cropped Images: {cropped_images}")

        if result_image_path is not None:
            original_image_path = url_for('uploaded_file', filename=file.filename)
            units = extracted_text.get('Units', 'N/A')
            box_no = extracted_text.get('BoxNo', 'N/A')
            print(f"Units: {units}")  # Debug print
            print(f"Box Number: {box_no}")  # Debug print
            return render_template('result.html',
                                   original_image=original_image_path,
                                   result_image=result_image_path,
                                   units=units,
                                   box_no=box_no,
                                   cropped_images=cropped_images)
        else:
            return "Error processing image"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def detect_objects(image_path):
    with open(image_path, 'rb') as image_file:
        response = requests.post(ROBOFLOW_URL, files={'file': image_file}, data={'confidence': 0.5, 'overlap': 0.5})
        if response.status_code == 200:
            detections = response.json()
            print(f"API Response: {detections}")  # Debug print
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            extracted_text = {'Units': '', 'BoxNo': ''}
            cropped_images = []

            class_texts = defaultdict(list)

            for i, detection in enumerate(detections.get('predictions', [])):
                class_name = detection.get('class')
                print(f"Detected class: {class_name}")  # Debug print
                x, y, w, h = detection.get('x', 0), detection.get('y', 0), detection.get('width', 0), detection.get('height', 0)
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2
                bbox = [x1, y1, x2, y2]

                draw.rectangle(bbox, outline="red", width=5)
                draw.text((x1, y1), class_name, fill="red")

                cropped_image = image.crop(bbox)
                cropped_image = apply_image_enhancements(cropped_image)
                cropped_image_path = os.path.join(app.config['CROPPED_FOLDER'], f'cropped_{i}.png')
                cropped_image.save(cropped_image_path)
                cropped_images.append(url_for('static', filename=f'cropped_images/cropped_{i}.png'))

                extracted_text_for_class = extract_numbers_from_image(cropped_image_path)
                print(f"Extracted text for {class_name}: {extracted_text_for_class}")  # Debug print

                class_texts[class_name].append(extracted_text_for_class)

            for class_name, texts in class_texts.items():
                if texts:
                    aggregated_text = ' '.join(texts).strip()
                    print(f"Aggregated text for {class_name}: {aggregated_text}")  # Debug print
                    if (class_name == 'Units') or (class_name == 'units'):
                        extracted_text['Units'] = aggregated_text
                    elif (class_name == 'BoxNo') or (class_name == 'boxno'):
                        extracted_text['BoxNo'] = aggregated_text

            result_image_path = os.path.join(app.config['DETECTION_FOLDER'], f"result_{os.path.basename(image_path)}")
            image.save(result_image_path)

            print(f"Returning: {result_image_path}, {extracted_text}, {cropped_images}")  # Debug print

            return url_for('static', filename=f'detections/result_{os.path.basename(image_path)}'), extracted_text, cropped_images
        else:
            return None, None, []


def extract_numbers_from_image(image_path):
    print(f"Processing image: {image_path}")

    # Preprocess image
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale

    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast

    # Apply sharpening filter
    image = image.filter(ImageFilter.SHARPEN)

    # Attempt with EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    print(f"EasyOCR raw result: {result}")  # Debug print

    if result:
        full_text = ' '.join([detection[1] for detection in result])
        print(f"EasyOCR full text: {full_text}")  # Debug print
        if full_text:
            return post_process_text(full_text)

    # If EasyOCR fails, attempt with Pytesseract
    text = pytesseract.image_to_string(image, config='--psm 6')
    print(f"Pytesseract raw result: {text}")  # Debug print

    return post_process_text(text.strip())

def post_process_text(text):
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def apply_image_enhancements(image):
    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast

    # Apply sharpening filter
    image = image.filter(ImageFilter.SHARPEN)

    # Convert image to binary (black and white)
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)

    return image

if __name__ == "__main__":
    app.run(debug=True)
