import os
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image
import tifffile
from utils import preprocess_image, overlay_mask_on_image  

from model.model import get_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load("model/water_segmentation_model.pth", map_location=device))
model.to(device).eval()

@app.route("/", methods=["GET", "POST"])
def index():
    mask_filename = None
    original_filename = None
    overlay_filename = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            ext = filename.split('.')[-1]
            unique_name = f"{uuid.uuid4()}.{ext}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(save_path)

            if ext.lower() in ['tif', 'tiff']:
                image = tifffile.imread(save_path).astype(np.float32)
                image = (image - image.min()) / (image.max() - image.min())

                if image.shape[0] == 12:
                    image = np.transpose(image, (1, 2, 0))

                input_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)

            else:
                image = preprocess_image(save_path).squeeze(0).permute(1, 2, 0).cpu().numpy()
                input_tensor = preprocess_image(save_path).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                predicted_mask = (output > 0.5).float().squeeze().cpu().numpy()

            mask_img = Image.fromarray((predicted_mask * 255).astype(np.uint8))
            mask_filename = f"mask_{unique_name}.png"
            mask_img.save(os.path.join(app.config['UPLOAD_FOLDER'], mask_filename))


            if ext.lower() in ['tif', 'tiff'] and image.shape[2] >= 4:
                r = image[..., 3]
                g = image[..., 2]
                b = image[..., 1]
                rgb_image = np.stack([r, g, b], axis=-1)
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = (image * 255).astype(np.uint8)

            original_filename = f"orig_{unique_name}.png"
            Image.fromarray(rgb_image).save(os.path.join(app.config['UPLOAD_FOLDER'], original_filename))

            overlay = overlay_mask_on_image(rgb_image, predicted_mask)
            overlay_filename = f"overlay_{unique_name}.png"
            Image.fromarray(overlay).save(os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename))

    return render_template("index.html",
                           original_path=original_filename,
                           mask_path=mask_filename,
                           overlay_path=overlay_filename)

if __name__ == "__main__":
    app.run(debug=True)
