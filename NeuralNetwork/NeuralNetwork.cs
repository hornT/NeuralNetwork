using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        // https://habrahabr.ru/post/313216/

        private Layer[] _layers;
        private Layer _inputLayer;
        private Layer _outputLayer;

        public void Train(TrainData[] data, int maxEpoch = 50000, int hiddenLayersCount = 1, double e = 0.7, double a = 0.3)
        {
            PrepareData(data, hiddenLayersCount);

            double mse = 0;
            for (int i = 0; i < maxEpoch; i++)
            {
                foreach (TrainData dataItem in data)
                {
                    Calculate(dataItem.Input);
                    _outputLayer.CalcSigma(dataItem.Output);

                    // Calc sigma and update weights for other layers
                    for (int j = _layers.Length - 2; j >= 0; j--)
                    {
                        _layers[j].UpdateWeights(e, a);
                    }
                }
            }

            // calc mse
            foreach (TrainData dataItem in data)
            {
                Calculate(dataItem.Input);
                mse += _outputLayer.GetMse(dataItem.Output);
            }

            mse /= data.Length;

            Console.WriteLine($"mse: {mse}");
        }

        public double[] Calculate(double[] input)
        {
            _inputLayer.SetValues(input);

            for (int i = 1; i < _layers.Length; i++)
                _layers[i].CalculateValues();

            return _outputLayer.GetValues();
        }

        private void PrepareData(TrainData[] data, int hiddenLayersCount)
        {
            int inputCount = data[0].Input.Length;
            int outputCount = data[0].Output.Length;
            int hiddenCount = Math.Max(3, inputCount / 2);

            List<Layer> layers = new List<Layer>();

            // Output layer
            _outputLayer = new Layer(outputCount);
            layers.Add(_outputLayer);
            
            // Hidden layers
            Layer currLayer = _outputLayer;
            for (int i = 0; i < hiddenLayersCount; i++)
            {
                currLayer = new Layer(hiddenCount, currLayer);
                layers.Add(currLayer);
            }

            // Input layer
            _inputLayer = new Layer(inputCount, currLayer);
            layers.Add(_inputLayer);

            layers.Reverse();
            _layers = layers.ToArray();
        }
    }
}
