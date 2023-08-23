from flask import Flask, request
import torch
import matplotlib.pyplot as plt
from my_model import UNet
from my_model import DiffusionModel

IMAGE_SHAPE = (64, 64)
model = DiffusionModel()
unet = UNet(labels=True)
unet.load_state_dict(torch.load(
    ("/Users/danielweiner/School/CunyTechPrep/Hackathon/epoch_ 170.pth"), map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('epoch_170'))
unet.eval()  # Set the model to evaluation mode

app = Flask(__name__)


@app.route('/')
def index():
    return '''
        <form action="/submit" method="post">
            <label for="text">Enter your text:</label>
            <textarea id="text" name="text" required></textarea>
            <input type="submit" value="Submit">
        </form>
    '''


@app.route('/submit', methods=['POST'])
def submit():
    # user_text = request.form['text']
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    NUM_CLASSES = len(classes)
    NUM_DISPLAY_IMAGES = 5
    torch.manual_seed(torch.randint())
    plt.figure(figsize=(20, 20))
    f, ax = plt.subplots(NUM_CLASSES, NUM_DISPLAY_IMAGES, figsize=(160, 60))

    for c in range(NUM_CLASSES):
        imgs = torch.randn((NUM_DISPLAY_IMAGES, 3) + IMAGE_SHAPE).to(device)
        for i in reversed(range(diffusion_model.timesteps)):
            t = torch.full((1,), i, dtype=torch.long, device=device)
            labels = torch.tensor(
                [c] * NUM_DISPLAY_IMAGES).resize(NUM_DISPLAY_IMAGES, 1).float().to(device)
            imgs = diffusion_model.backward(
                x=imgs, t=t, model=unet.eval().to(device), labels=labels)
        for idx, img in enumerate(imgs):
            ax[c][idx].imshow(reverse_transform(img))
            ax[c][idx].set_title(f"Class: {classes[c]}", fontsize=100)
    plt.show()
    # Process the text with your stable diffusion model
    processed_text = user_text
    tensor_input = torch.tensor(processed_text)
    output = model(tensor_input)
    result = your_postprocessing_function(
        output)  # Replace with your postprocessing

    return f'Your result: {result}'


if __name__ == '__main__':
    app.run(debug=True)
