# -*- coding: utf-8 -*-
"""NN_Assignment_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14jWw3WiemoXWVGlTKDSDlw6GQKnHDw9z
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        # Initialize weights and bias with random values
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # Compute the weighted sum and apply the sigmoid activation function
        weighted_sum = np.dot(input_data, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

    def train(self, input_data, labels, epochs=100):
        for epoch in range(epochs):
            for inputs, label in zip(input_data, labels):
                # Make a prediction
                prediction = self.predict(inputs)

                # Compute the gradient
                gradient = (label - prediction) * prediction * (1 - prediction) * inputs

                # Update weights and bias
                self.weights += self.learning_rate * gradient
                self.bias += self.learning_rate * (label - prediction) * prediction * (1 - prediction)

    def test(self, test_data):
        predictions = [round(self.predict(inputs)) for inputs in test_data]
        return predictions

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_array = np.array(img).flatten() / 255.0  # Flatten and normalize
            images.append(img_array)
    return images

def create_labels(num_samples, num_classes):
    labels = []
    for i in range(num_classes):
        labels.extend([i] * num_samples)
    return labels

def generate_images(output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    for digit in range(10):
        for i in range(num_samples):
            # Create a blank image with a white background
            image = Image.new("L", (20, 20), color="white")
            draw = ImageDraw.Draw(image)

            # Load a font
            font = ImageFont.load_default()

            # Place the digit in the center of the image
            text_width, text_height = draw.textsize(str(digit), font)
            x = (20 - text_width) / 2
            y = (20 - text_height) / 2
            draw.text((x, y), str(digit), font=font, fill="black")

            # Save the image
            image_filename = f"{output_dir}/{digit}_{i}.png"
            image.save(image_filename)

    print(f"Images generated and saved in the '{output_dir}' directory.")

def generate_test_images(test_output_dir, num_samples=5):
    os.makedirs(test_output_dir, exist_ok=True)

    for digit in range(10):
        for i in range(num_samples):
            # Create a blank image with a white background
            test_image = Image.new("L", (20, 20), color="white")
            test_draw = ImageDraw.Draw(test_image)

            # Load a font
            font = ImageFont.load_default()

            # Place the digit in the center of the image
            text_width, text_height = test_draw.textsize(str(digit), font)
            x = (20 - text_width) / 2
            y = (20 - text_height) / 2
            test_draw.text((x, y), str(digit), font=font, fill="black")

            # Save the test image
            test_image_filename = f"{test_output_dir}/test_{digit}_{i}.png"
            test_image.save(test_image_filename)

    print(f"Test images generated and saved in the '{test_output_dir}' directory.")

if __name__ == "__main__":
    # Task 1: Generate training images
    generate_images("handwritten_images", num_samples=10)

    # Task 2: Generate test images
    generate_test_images("test_images", num_samples=5)

    # Task 3: Create perceptron
    input_size = 20 * 20  # Size of the images
    perceptron = Perceptron(input_size)

    # Task 4: Train the perceptron
    train_images = load_images("handwritten_images")
    train_labels = create_labels(num_samples=10, num_classes=10)
    perceptron.train(train_images, train_labels, epochs=1000)

    # Task 5: Test the perceptron
    test_images = load_images("test_images")
    predictions = perceptron.test(test_images)

    # Print the predictions
    print("Predictions:", predictions)