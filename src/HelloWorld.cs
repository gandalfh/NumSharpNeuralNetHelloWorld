using MNIST.IO;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharpNeuralNetHelloWorld
{
    class HelloWorld
    {
        public HelloWorld()
        {
            inputToHiddenWeights20x784 = np.random.uniform(-0.5, 0.5, (20, 784));

            inputToHiddenWeightsInitial20x784 = inputToHiddenWeights20x784.Clone();

            hiddenToOutputWeights10x20 = np.random.uniform(-0.5, 0.5, (10, 20));

            hiddenBiases20x1 = np.zeros((20, 1));

            outputBiases10x1 = np.zeros((10, 1));
        }

        //These are the core matrices that are evolved by the neural network code
        public NDArray inputToHiddenWeights20x784;
        public NDArray hiddenToOutputWeights10x20;
        public NDArray hiddenBiases20x1;
        public NDArray outputBiases10x1;

        //These are all here for debugging convenience as it is often nice to see what values were at each step in the ProcessNeuralNetwork function
        public NDArray inputToHiddenWeightsInitial20x784;
        public NDArray hiddenNeuronBiasPreAdjust20x1;
        public NDArray inputToHiddenWeightsPreAdjust20x784;
        public NDArray outputBiasPreAdjust10x1;
        public NDArray expectedHiddenDelta20X1;
        public NDArray inputToHiddenWeightAdjustment20x784;
        public NDArray currentOutputNeurons10x1;
        public NDArray hiddenToOutputWeightAdjustment10X20;
        public NDArray hiddenNeuronsBeforeSigmoid20x1;
        public NDArray outputNeuronsBeforeSigmoid10x1;
        public NDArray expectedOutputDelta10X1;
        public NDArray expectedOutput10x1;
        public NDArray hiddenToOutputWeightsPreAdjust10x20;
        public NDArray hiddenNeuronsSigmoidDifferential20X1;
        public NDArray oneMinusHiddenNeurons20x1;
        public NDArray hiddenToOutputWeightsXExpectedOutputDelta20x1;
        public NDArray currentHiddenNeurons20X1;

        public int detected;

        public float learnRate = 0.01f;

        public IEnumerator<TestCase> LoadFiles(string labelsZipFile, string imagesZipFile)
        {
            var imagesAndLabels = FileReaderMNIST.LoadImagesAndLables(labelsZipFile, imagesZipFile);
            return imagesAndLabels.GetEnumerator();
        }
        //This function feeds an image into the neural network and applies the backpropagation
        public void ProcessNeuralNetwork(TestCase testCase, bool storeIncrementalSteps)
        {

            //Start
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=0
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=0

            NDArray image784x1 = testCase.AsNDArray();

            //Hidden neuron calculations
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=18
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=18

            var hiddenPreSigmoid20x1 = hiddenBiases20x1 + np.matmul(inputToHiddenWeights20x784, image784x1);

            if (storeIncrementalSteps)
            {
                hiddenNeuronsBeforeSigmoid20x1 = hiddenPreSigmoid20x1.Clone();
            }

            //normalize with sigmoid
            currentHiddenNeurons20X1 = np.divide(1, (np.add(1, np.exp(-hiddenPreSigmoid20x1))));

            //Final output neuron calculation
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=78
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=78

            //map from inputs to output and add bias
            var outputPreSigmoid10x1 = outputBiases10x1 + np.matmul(hiddenToOutputWeights10x20, currentHiddenNeurons20X1);

            if (storeIncrementalSteps)
            {
                outputNeuronsBeforeSigmoid10x1 = outputPreSigmoid10x1.Clone();
            }

            //normalize with sigmoid
            currentOutputNeurons10x1 = 1 / (1 + np.exp(-outputPreSigmoid10x1));

            expectedOutput10x1 = testCase.AsLabelNDArray();

            //np.argmax(o) gives the index of the maximum number.  This is simply checking to see which output was switched on
            detected = np.argmax(currentOutputNeurons10x1);

            //Start back propagation by figuring out by how much the output was "wrong"
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=108
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=108

            expectedOutputDelta10X1 = currentOutputNeurons10x1 - expectedOutput10x1;

            //Calculate the hidden to output weights "wrongness" using the output's "wrongness"
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=138
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=138

            //Multiply difference from expected by the hidden neurons to make a matrix to adjust the hidden to output weights
            hiddenToOutputWeightAdjustment10X20 = np.matmul(expectedOutputDelta10X1, np.transpose(currentHiddenNeurons20X1));

            if (storeIncrementalSteps)
            {
                hiddenToOutputWeightsPreAdjust10x20 = hiddenToOutputWeights10x20.Clone();
            }

            //Adjust the hidden to output weights by the calculated "wrongness"
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=198
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=198"

            //adjust the hidden to output weights
            hiddenToOutputWeights10x20 += -learnRate * hiddenToOutputWeightAdjustment10X20;

            if (storeIncrementalSteps)
            {
                outputBiasPreAdjust10x1 = outputBiases10x1.Clone();
            }

            //Adjust the output biases
            outputBiases10x1 += -learnRate * expectedOutputDelta10X1;

            //Now move on to building the matrix to adjust the input to hidden weights
            //differential of sigmoid is sigmoid * (1 - sigmoid). Remember above where we used sigmoid function on currentHiddenNeurons?
            //Result is 20x1 matrix
            if (storeIncrementalSteps)
            {
                oneMinusHiddenNeurons20x1 = 1 - currentHiddenNeurons20X1;
            }

            //Calculate the sigmoid differential of the hidden neurons to use later
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=204
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=204

            hiddenNeuronsSigmoidDifferential20X1 = (currentHiddenNeurons20X1 * (1 - currentHiddenNeurons20X1));

            //Multiply the "wrongness" of the output by the adjusted hidden to output weights to get the weights "wrongness"
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=264
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=264

            //Combine expectedOutputDelta (10X1) with the hiddenToOutputWeights (10x20), resulting in a 20x1 matrix
            hiddenToOutputWeightsXExpectedOutputDelta20x1 = np.matmul(np.transpose(hiddenToOutputWeights10x20), expectedOutputDelta10X1);

            //Multiply the "wrongness" of the adjusted hidden to output weights by the hidden neuron sigmoid differential to get the "wrongness" of each hidden neuron
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=324
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=324"

            //Then combine in the differential of the sigmoid (20X1) X (20X1) = 20X1
            expectedHiddenDelta20X1 = hiddenToOutputWeightsXExpectedOutputDelta20x1 * hiddenNeuronsSigmoidDifferential20X1;

            //Multiply the "wrongness" of each hidden neuron by the original image to calculate the "wrongness" of the input to hidden matrix
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=384
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=384

            //Then multiply by the image to make a 20x784 matrix
            inputToHiddenWeightAdjustment20x784 = np.matmul(expectedHiddenDelta20X1, np.transpose(image784x1));

            if (storeIncrementalSteps)
            {
                inputToHiddenWeightsPreAdjust20x784 = inputToHiddenWeights20x784.Clone();
            }

            //Apply the "wrongness" of the input to hidden matrix to the input to hidden matrix
            //10 fps: https://www.youtube.com/watch?v=zpCFjNjuBaY&t=453
            //60 fps: https://www.youtube.com/watch?v=IQdxHrfdMwk&t=453

            //Finally we adjust the actual weights using our calculated adjustment matrix
            inputToHiddenWeights20x784 += -learnRate * inputToHiddenWeightAdjustment20x784;
            if (storeIncrementalSteps)
            {
                hiddenNeuronBiasPreAdjust20x1 = hiddenBiases20x1.Clone();
            }

            //And adjust the bias using a portion of how far off the hidden neurons were
            hiddenBiases20x1 += -learnRate * expectedHiddenDelta20X1;
        }
    }
}
