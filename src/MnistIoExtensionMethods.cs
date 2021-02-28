using MNIST.IO;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharpNeuralNetHelloWorld
{
    public static class MnistIoExtensionMethods
    {
        //This function converts the image data into a NumSharp NDArray of values from 0.0 to 1.0
        public static NDArray AsNDArray(this TestCase testCase)
        {
            var Image = testCase.Image;
            var result = np.array(Image);
            result = result.flat.reshape(Shape.Matrix(Image.GetLength(0) * Image.GetLength(1), 1))/256.0;

            return result;
        }



        //This function builds 10x1 NDArray filled with zeroes, with a 1 at the image index label (for example if the testcase represents a 0, then array[0] will be 1
        public static NDArray AsLabelNDArray(this TestCase testCase)
        {
            var res = new NDArray(NPTypeCode.Double, Shape.Matrix(10, 1));
            res[testCase.Label][0] = 1;
            return res;
        }
    }
}
