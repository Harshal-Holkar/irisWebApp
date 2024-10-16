from flask import Flask, request, render_template, redirect, url_for
import numpy as np

app = Flask(__name__)

# Define neural network parameters
input_hidden_weights = np.array([
    [1.55141789603358, -1.62410372855789, 3.82874043469150, 3.25392835379368],
    [-0.900570833190443, 2.57789046971807, -1.02559862052883, 0.530005700403718],
    [0.537532559643009, -0.708457152390437, 0.935177828438799, 1.13088376710114],
    [0.238116480554309, 0.624065005774269, 0.00742087645945280, 0.0244841645674166],
    [-0.599982585255804, 0.567602179936636, -1.04161840935648, -1.09683894062973],
    [0.152609622416511, -0.253417093878617, 0.339171295616111, 0.325789137819781],
    [0.451341709563544, 3.08148409622368, 0.312884953481823, 2.16027646039249],
    [0.817580814808963, 0.496922414570123, -2.20250223064258, -3.24352482531454],
    [0.268872466624237, -0.736094347461468, 0.107070274162553, -0.350624410664811]
])

hidden_bias = np.array([-1.09976767136448, 0.0721423138119484, -0.135181413069945, 
                        0.313814828515395, -0.226851324878775, 0.426899543928486, 
                        0.643819220507793, 2.96589151682121, 0.112889868040944])

hidden_output_weights = np.array([
    [-3.57863762324817, 0.343355639023627, 1.36419961527214, -2.92410785250222,
     -0.727746181690593, 0.263453457836691, 2.25348001933178, 0.128947866977109, 0.149991633345534],
    [4.28763646686304, 0.391220041197829, -1.82298558005867, 0.0257154192824061,
     0.778613136501140, -1.15932101754209, -1.04014598586726, 3.02483726013221, -0.0694125421708802],
    [-2.70062099951572, -0.975349534296665, 2.27228015350991, 0.915741390315765,
     1.16122826290894, 1.14176777833932, 1.27331694886843, -3.96214505332840, 0.749673160184164]
])

output_bias = np.array([1.76178507969452, -2.08275077315966, 0.847861143845128])

max_input = np.array([7.9, 4.4, 6.9, 2.5])
min_input = np.array([4.3, 2.2, 1.1, 0.1])
mean_inputs = np.array([5.84150943396226, 3.06981132075472, 3.73113207547170, 1.19056603773585])

def scale_inputs(inputs):
    scaled = np.zeros_like(inputs)
    for i in range(len(inputs)):
        delta = (1 - 0) / (max_input[i] - min_input[i])
        scaled[i] = 0 - delta * min_input[i] + delta * inputs[i]
    return scaled

def logistic(x):
    x = np.asarray(x)
    return np.where(x > 100.0, 1.0, np.where(x < -100.0, 0.0, 1.0 / (1.0 + np.exp(-x))))

def normalize(output):
    output = np.clip(output, 0, 100)
    total = np.sum(np.exp(output))
    return np.exp(output) / total if total > 0 else output

def compute_feed_forward(input_data):
    hidden_layer = np.dot(input_hidden_weights, input_data) + hidden_bias
    hidden_layer = logistic(hidden_layer)

    output_layer = np.dot(hidden_output_weights, hidden_layer) + output_bias
    output_layer = normalize(output_layer)
    return output_layer

def run_neural_net_classification(inputs):
    scaled_inputs = scale_inputs(inputs)
    probabilities = compute_feed_forward(scaled_inputs)
    return probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None    
    
    if request.method == 'POST':  
        inputs = [
            float(request.form.get('sepallength', -9999)),
            float(request.form.get('sepalwidth', -9999)),
            float(request.form.get('petallength', -9999)),
            float(request.form.get('petalwidth', -9999))
        ]
        
        # Replace -9999 inputs with mean values
        inputs = [mean if inp == -9999 else inp for inp, mean in zip(inputs, mean_inputs)]
        
        # Run the classification and get probabilities
        probabilities = run_neural_net_classification(inputs)

        predicted_class = np.argmax(probabilities)
        classes = ["SETOSA", "VERSICOL", "VIRGINIC"]
        
        # Assign prediction and confidence
        prediction = classes[predicted_class]
        confidence = probabilities[predicted_class]
        
        # Redirect to the same route to prevent form resubmission
        return redirect(url_for('index', prediction=prediction, confidence=confidence))
    
    # Retrieve predictions from query parameters if they exist
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
