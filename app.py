import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Convert RGB to CMYK
def rgb_to_cmyk(r, g, b, rgb_scale=255, cmyk_scale=100):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, cmyk_scale
    r = r / rgb_scale
    g = g / rgb_scale
    b = b / rgb_scale
    c = 1 - r
    m = 1 - g
    y = 1 - b
    k = min(c, m, y)
    if k == 1:
        return 0, 0, 0, cmyk_scale
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    return (c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale)

# Convert BGR image to CMYK
def bgr_to_cmyk(image, cmyk_scale=100):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    c = 1 - rgb_image[:, :, 0]
    m = 1 - rgb_image[:, :, 1]
    y = 1 - rgb_image[:, :, 2]
    k = np.minimum(np.minimum(c, m), y)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = (c - k) / (1 - k)
        m = (m - k) / (1 - k)
        y = (y - k) / (1 - k)
        c[np.isnan(c)] = 0
        m[np.isnan(m)] = 0
        y[np.isnan(y)] = 0
    c = (c * cmyk_scale).astype(np.float32)
    m = (m * cmyk_scale).astype(np.float32)
    y = (y * cmyk_scale).astype(np.float32)
    k = (k * cmyk_scale).astype(np.float32)
    cmyk_image = np.stack((c, m, y, k), axis=-1)
    return cmyk_image

# Calculate toner usage
def calculate_toner_usage(cmyk_image):
    C, M, Y, K = cmyk_image[:, :, 0], cmyk_image[:, :, 1], cmyk_image[:, :, 2], cmyk_image[:, :, 3]
    total_pixels = C.size
    c_percent = np.sum(C) / (100 * total_pixels) * 100
    m_percent = np.sum(M) / (100 * total_pixels) * 100
    y_percent = np.sum(Y) / (100 * total_pixels) * 100
    k_percent = np.sum(K) / (100 * total_pixels) * 100
    return c_percent, m_percent, y_percent, k_percent

# Calculate total print cost
def calculate_print_cost(cmyk_usage, cartridge_costs, paper_cost, labor_cost):
    c_percent, m_percent, y_percent, k_percent = cmyk_usage
    c_cost, m_cost, y_cost, k_cost = cartridge_costs
    toner_cost = (c_percent / 100 * c_cost +
                  m_percent / 100 * m_cost +
                  y_percent / 100 * y_cost +
                  k_percent / 100 * k_cost)
    total_cost = toner_cost + paper_cost + labor_cost
    return total_cost

def convert_pdf_to_image(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # Get the first page
        pix = page.get_pixmap()
        
        # Check the number of channels in the image
        if pix.n == 3:  # RGB
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        elif pix.n == 4:  # RGBA
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR
        else:
            raise ValueError("Unsupported number of channels: {}".format(pix.n))

        return img_bgr
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check file type and process accordingly
        if filename.lower().endswith('.pdf'):
            print(f"Processing PDF: {filename}")
            image = convert_pdf_to_image(file_path)
            if image is None:
                flash('Failed to convert PDF to image. Please ensure the PDF is valid.')
                return redirect(url_for('index'))
        else:
            image = cv2.imread(file_path)
        
        if image is None:
            flash('Invalid image file')
            return redirect(url_for('index'))

        # Get data from the form
        c_cost = float(request.form['c_cost'])
        m_cost = float(request.form['m_cost'])
        y_cost = float(request.form['y_cost'])
        k_cost = float(request.form['k_cost'])
        paper_cost = float(request.form['paper_cost'])
        labor_cost = float(request.form['labor_cost'])
        copies = int(request.form['copies'])

        # Calculate toner usage
        cmyk_image = bgr_to_cmyk(image)
        cmyk_usage = calculate_toner_usage(cmyk_image)
        total_cost = calculate_print_cost(cmyk_usage, (c_cost, m_cost, y_cost, k_cost), paper_cost, labor_cost)
        total_print_cost = total_cost * copies

        return render_template('result.html', cmyk_usage=cmyk_usage, total_cost=total_print_cost)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
