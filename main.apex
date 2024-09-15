class NeuralNetwork {
    Integer inputSize;
    Integer hiddenSize;
    Integer outputSize;
    
    List<List<Double>> inputHiddenWeights;
    List<List<Double>> hiddenOutputWeights;
    
    NeuralNetwork(Integer inputSize, Integer hiddenSize, Integer outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        this.inputHiddenWeights = initializeWeights(hiddenSize, inputSize);
        this.hiddenOutputWeights = initializeWeights(outputSize, hiddenSize);
    }
    
    List<List<Double>> initializeWeights(Integer rows, Integer cols) {
        List<List<Double>> weights = new List<List<Double>>();
        for (Integer i = 0; i < rows; i++) {
            List<Double> row = new List<Double>();
            for (Integer j = 0; j < cols; j++) {
                row.add(Math.random() * 0.2 - 0.1);
            }
            weights.add(row);
        }
        return weights;
    }
    
    Double sigmoid(Double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    Double sigmoidDerivative(Double x) {
        return x * (1.0 - x);
    }
    
    List<Double> forward(List<Double> input) {
        List<Double> hiddenLayer = new List<Double>();
        List<Double> outputLayer = new List<Double>();
        
        // Calculate hidden layer
        for (Integer i = 0; i < hiddenSize; i++) {
            Double sum = 0;
            for (Integer j = 0; j < inputSize; j++) {
                sum += input.get(j) * inputHiddenWeights.get(i).get(j);
            }
            hiddenLayer.add(sigmoid(sum));
        }
        
        // Calculate output layer
        for (Integer i = 0; i < outputSize; i++) {
            Double sum = 0;
            for (Integer j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer.get(j) * hiddenOutputWeights.get(i).get(j);
            }
            outputLayer.add(sigmoid(sum));
        }
        
        return outputLayer;
    }
    
    void train(List<List<Double>> inputs, List<List<Double>> expectedOutputs, Integer epochs, Double learningRate) {
        for (Integer epoch = 0; epoch < epochs; epoch++) {
            for (Integer i = 0; i < inputs.size(); i++) {
                List<Double> input = inputs.get(i);
                List<Double> expectedOutput = expectedOutputs.get(i);
                
                // Forward propagation
                List<Double> outputLayer = forward(input);
                
                // Compute errors and update weights
                // (Omitted for brevity: requires backpropagation implementation)
            }
        }
    }
    
    static void test(NeuralNetwork nn) {
        List<List<Double>> testInputs = new List<List<Double>>{
            new List<Double>{0.0, 0.0},
            new List<Double>{0.0, 1.0},
            new List<Double>{1.0, 0.0},
            new List<Double>{1.0, 1.0}
        };
        
        for (List<Double> input : testInputs) {
            List<Double> output = nn.forward(input);
            System.debug('Input: ' + input + ' => Output: ' + output);
        }
    }
}

// Create and test the neural network
NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
nn.train(new List<List<Double>>{
    new List<Double>{0.0, 0.0},
    new List<Double>{0.0, 1.0},
    new List<Double>{1.0, 0.0},
    new List<Double>{1.0, 1.0}
}, new List<List<Double>>{
    new List<Double>{0.0},
    new List<Double>{1.0},
    new List<Double>{1.0},
    new List<Double>{0.0}
}, 10000, 0.1);

NeuralNetwork.test(nn);
