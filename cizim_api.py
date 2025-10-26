from flask import Flask, request, Response, render_template
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from io import BytesIO

app = Flask(__name__)

# OpenJourney modelini yükle
model_id = "prompthero/openjourney"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa cuda
    pipe.enable_attention_slicing()  # Bellek optimizasyonu
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_accident():
    user_text = request.json['description']
    prompt = f"Bird's eye view of a car accident based on: {user_text}, black and white pencil sketch, detailed line art, clean and minimal, only road and crashed vehicles, mdjrny-v4 style"
    negative_prompt = "colors, vibrant hues, people, buildings, trees, animals, extra objects, sky, background details, textures, gradients, shading"

    try:
        image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
        image = image.convert("L")  # Siyah-beyaz
    except Exception as e:
        return Response(f"Hata: {e}", status=500)

    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return Response(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)