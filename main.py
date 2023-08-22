from flask import Flask, request
import torch
import torch_model

model = torch_model.DiffusionModel()
model.load_state_dict(torch.load('epoch: x'))
model.eval()  # Set the model to evaluation mode

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
    user_text = request.form['text']

    # Process the text with your stable diffusion model
    processed_text = user_text
    tensor_input = torch.tensor(processed_text)
    output = model(tensor_input)
    result = your_postprocessing_function(
        output)  # Replace with your postprocessing

    return f'Your result: {result}'


if __name__ == '__main__':
    app.run(debug=True)
