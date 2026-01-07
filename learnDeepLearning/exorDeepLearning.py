import numpy as np

nb_neurons = 4
learning_rate = 0.5

input_examples = [np.array([0, 0]).reshape(2, 1), np.array([0, 1]).reshape(2, 1), np.array([1, 0]).reshape(2, 1), np.array([1, 1]).reshape(2, 1)]
output_truth = [0, 1, 1, 0]

weights_1 = np.random.uniform(-1, 1, (nb_neurons, 2))
biases_1 = np.random.uniform(-1, 1, (nb_neurons, 1))
weights_2 = np.random.uniform(-1, 1, (1, nb_neurons))
biases_2 = np.random.uniform(-1, 1, (1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


for _epoch in range(100000):
  delta_weights_1 = np.zeros_like(weights_1)
  delta_biases_1 = np.zeros_like(biases_1)
  delta_weights_2 = np.zeros_like(weights_2)
  delta_biases_2 = np.zeros_like(biases_2)
  for i, single_input in enumerate(input_examples):
      # Forward pass
      z1 = weights_1 @ single_input + biases_1
      layer_1 = sigmoid(z1)
      z2 = weights_2 @ layer_1 + biases_2
      output = sigmoid(z2)
      
      # Backpropagation
      delta_weights_2 += (output - output_truth[i]) * sigmoid_derivative(z2) * layer_1.T * learning_rate / len(input_examples)
      delta_biases_2 += (output - output_truth[i]) * sigmoid_derivative(z2) * learning_rate / len(input_examples)
      delta_weights_1 += ((output - output_truth[i]) * sigmoid_derivative(z2) * weights_2.T * sigmoid_derivative(z1)) @ single_input.T * learning_rate / len(input_examples)
      delta_biases_1 += ((output - output_truth[i]) * sigmoid_derivative(z2) * weights_2.T * sigmoid_derivative(z1)) * learning_rate / len(input_examples)

  weights_2 -= delta_weights_2
  biases_2 -= delta_biases_2
  weights_1 -= delta_weights_1
  biases_1 -= delta_biases_1

for single_input in input_examples:
    z1 = weights_1 @ single_input + biases_1
    layer_1 = sigmoid(z1)
    z2 = weights_2 @ layer_1 + biases_2
    output = sigmoid(z2)
    print(f"Input: {single_input.T} Output: {output}")
