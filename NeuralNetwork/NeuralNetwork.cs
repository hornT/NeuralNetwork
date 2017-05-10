using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        // https://habrahabr.ru/post/313216/
        //private int inputCount;
        //private int hiddenCount;
        //private int outputCount;

        private Layer[] _layers;
        private Layer _inputLayer;
        private Layer _outputLayer;

        public void Train(TrainData[] data, int maxEpoch = 50000, int hiddenLayersCount = 2, double e = 0.7, double a = 0.3)
        {
            PrepareData(data, hiddenLayersCount);

            double mse = 0;
            //TrainData dataItem = data.FirstOrDefault();
            for (int i = 0; i < maxEpoch; i++)
            {
                foreach (TrainData dataItem in data)
                {
                    Calculate(dataItem.Input);
                    mse = _outputLayer.GetMse(dataItem.Output);
                    _outputLayer.CalcSigma(dataItem.Output);

                    // Calc sigma for other layers
                    for (int j = _layers.Length - 2; j >= 0; j--)
                    {
                        _layers[j].CalcSigma();
                        //_layers[j].UpdateWeights(e, a);
                    }
                    // Calc weights for other layers
                    for (int j = _layers.Length - 2; j >= 0; j--)
                    {
                        //_layers[j].CalcSigma();
                        _layers[j].UpdateWeights(e, a);
                    }
                }
            }

            Console.WriteLine($"mse: {mse}");
        }

        public double[] Calculate(int[] input)
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

        private double Mse()
        {
            return 0;
        }
    }
}
