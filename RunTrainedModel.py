from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

class NeuralNetwork:
    def __init__(self):
        self.weights = np.array([[0.5], [0.5]])  # Example weights

    def predict(self, input_data):
        inputs = np.array(input_data)
        output = np.dot(inputs, self.weights)
        return output.tolist()

neural_network = NeuralNetwork()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    input_data = [float(x) for x in input_data.split(',')]
    prediction = neural_network.predict(input_data)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
