class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))

        output = 1 if weighted_sum >= self.threshold else 0

        return output


def main():
    
    neuron_weights = [0.5, -0.5, 0.3]
    print(f" neuron_weights: { neuron_weights}")
    neuron_threshold = 0.3
    print(f"neuron_threshold : {neuron_threshold }")


    neuron = McCullochPittsNeuron(neuron_weights, neuron_threshold)


    input_values = [1 ,0, 1]


    output = neuron.activate(input_values)

    print(f"Input Values: {input_values}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
