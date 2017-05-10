using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class TrainData
    {
        public readonly int[] Input;

        public readonly double[] Output;

        public TrainData(int[] input, double[] output)
        {
            Input = input;
            Output = output;
        }
    }
}
