using NumSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace NumSharpNeuralNetHelloWorld
{
    class Program
    {
        static void Main()
        {
            HelloWorld helloWorld = new HelloWorld();

            int epoch = 0;
            LoadMatrices(helloWorld, out epoch);

            //var imageZipFile = @"MNISTData\train-images-idx3-ubyte.gz";
            //var labelZipFile = @"MNISTData\train-labels-idx1-ubyte.gz";

            var imageZipFile = @"MNISTData\t10k-images-idx3-ubyte.gz";
            var labelZipFile = @"MNISTData\t10k-labels-idx1-ubyte.gz";

            var testCaseEnumerator = helloWorld.LoadFiles(labelZipFile, imageZipFile);

            int processed = 0;

            var stopWatch = new Stopwatch();

            stopWatch.Start();

            var correct = 0;

            var incorrectDisplayed = 0;

            while (true)
            {
                if (!testCaseEnumerator.MoveNext())
                {
                    Console.WriteLine("Epoch: {0} completed", epoch);
                    SaveMatrices(helloWorld, epoch);
                    epoch++;
                    testCaseEnumerator = helloWorld.LoadFiles(labelZipFile, imageZipFile);
                    testCaseEnumerator.MoveNext();
                    processed = 0;
                    correct = 0;
                }

                var testCase = testCaseEnumerator.Current;

                rotate90Clockwise(28, testCase.Image); //Rotate the image so it matches the visualization video coordinate system

                helloWorld.ProcessNeuralNetwork(testCase, true);


                if (helloWorld.detected == testCase.Label)
                {
                    correct++;
                }
                else if (incorrectDisplayed++ < 1)
                {
                    PrintIncorrectImage(helloWorld, testCase);
                }

                processed++;

                if (stopWatch.ElapsedMilliseconds > 5000)
                {
                    var pctCorrect = ((float)correct / (float)processed)*100;
                    Console.WriteLine("Processed {0} images in epoch {1}, {2:F2}% correct", processed, epoch, pctCorrect);
                    
                    stopWatch.Restart();
                    incorrectDisplayed = 0;
                }
            }
        }

        private static void PrintIncorrectImage(HelloWorld helloWorld, MNIST.IO.TestCase testCase)
        {
            Console.WriteLine("Missed: " + testCase.Label + ", detected a " + helloWorld.detected);
            for (var i = 27; i >= 0; i--)
            {
                var line = "";
                var allSpaces = true;
                for (var j = 0; j < 28; j++)
                {
                    var value = testCase.Image[j, i];
                    if (value > 0)
                    {
                        //line += String.Format("{0:X2}", testCase.Image[j, i]);
                        line += MapByteToCharacter(testCase.Image[j, i]);
                        allSpaces = false;
                    }
                    else
                    {
                        line += " ";
                    }
                }

                var skipDisplayLine = (i == 0 || i == 27) && allSpaces;

                if (!skipDisplayLine)
                {
                    Console.WriteLine(line);
                }
            }
        }

        static char MapByteToCharacter(byte b)
        {
            string chars = ".,*0@";

            return chars[(int)((chars.Length-1) * (b / 255.0f))];
        } 
            

        static void LoadMatrices(HelloWorld helloWorld, out int epoch)
        {
            epoch = 0;
            if (File.Exists(@"Matrices\inputToHiddenWeightsInitial.npy"))
            {
                helloWorld.inputToHiddenWeightsInitial20x784 = np.load(@"Matrices\inputToHiddenWeightsInitial.npy");
                helloWorld.inputToHiddenWeights20x784 = np.load(@"Matrices\inputToHiddenWeights.npy");
                helloWorld.hiddenToOutputWeights10x20 = np.load(@"Matrices\hiddenToOutputWeights.npy");
                helloWorld.hiddenBiases20x1 = np.load(@"Matrices\hiddenBiases.npy");
                helloWorld.outputBiases10x1 = np.load(@"Matrices\outputBiases.npy");
                using (var file = File.OpenText(@"Matrices\epoch.txt"))
                {
                    var epochString = file.ReadToEnd();
                    epoch = int.Parse(epochString);
                }
           }
        }

        static void SaveMatrices(HelloWorld helloWorld, int epoch)
        {
            np.save(@"Matrices\inputToHiddenWeightsInitial", helloWorld.inputToHiddenWeightsInitial20x784);
            np.save(@"Matrices\inputToHiddenWeights", helloWorld.inputToHiddenWeights20x784);
            np.save(@"Matrices\hiddenToOutputWeights", helloWorld.hiddenToOutputWeights10x20);
            np.save(@"Matrices\hiddenBiases", helloWorld.hiddenBiases20x1);
            np.save(@"Matrices\outputBiases", helloWorld.outputBiases10x1);

            using (var file = File.CreateText(@"Matrices\epoch.txt"))
            {
                file.WriteLine(epoch.ToString());
            }

        }

        static void rotate90Clockwise(int N, byte[,] a)
        {
            for (int i = 0; i < N / 2; i++)
            {
                for (int j = i; j < N - i - 1; j++)
                {
                    var temp = a[i, j];
                    a[i, j] = a[N - 1 - j, i];
                    a[N - 1 - j, i] = a[N - 1 - i, N - 1 - j];
                    a[N - 1 - i, N - 1 - j] = a[j, N - 1 - i];
                    a[j, N - 1 - i] = temp;
                }
            }
        }
    }
}
