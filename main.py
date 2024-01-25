import numpy as np 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset using scikit-learn
mnist = fetch_openml(name='mnist_784', version=1)
X, y = mnist.data.astype('float32'), mnist.target.astype('int')
X /= 255.0

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
input_size = 784
hidden_size = 128
output_size = 10

# Initialize the weights and biases
np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Activation Function
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training params
learning_rate = 0.01
epochs = 100
batch_size = 64

# Training loop
for epoch in range(epochs):
    # Mini Batch Training
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        hidden_input = np.dot(batch_x, weights_input_hidden) + bias_hidden
        hidden_output = relu(hidden_input)
        output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_probs = softmax(output_layer_input)

        #Calculate loss
        batch_y = np.array(batch_y).astype(int)
        one_hot_labels = np.eye(output_size)[batch_y]
        loss = -np.sum(one_hot_labels * np.log(predicted_probs + 1e-10)) / len(batch_x)

        #Backpropagation (Gradient Descent)
        grad_output = predicted_probs - one_hot_labels
        grad_hidden = np.dot(grad_output, weights_hidden_output.T) *(hidden_output > 0)

        #Update weights and biases
        weights_hidden_output -= learning_rate*np.dot(hidden_output.T, grad_output) / len(batch_x)
        bias_output -= learning_rate * np.sum(grad_output, axis=0, keepdims=True) /len(batch_x)
        weights_input_hidden -= learning_rate * np.dot(batch_x.T, grad_hidden) / len(batch_x)
        bias_hidden -= learning_rate*np.sum(grad_hidden, axis=0, keepdims=True) /len(batch_x)
    
    #Calculate accuracy on training set
    hidden_input = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_output = relu(hidden_input)
    output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_labels = np.argmax(output_layer_input, axis=1)
    accuracy = np.mean(predicted_labels == y_train)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

#Model evluation on test set
hidden_input = np.dot(x_test, weights_input_hidden) + bias_hidden
hidden_output = relu(hidden_input)
output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
predicted_labels = np.argmax(output_layer_input, axis=1)
test_accuracy = np.mean(predicted_labels == y_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
