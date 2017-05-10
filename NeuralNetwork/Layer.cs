using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Layer
    {
        Neuron[] Neurons { get; }

        public Layer(int neuronCount)
        {
            Neurons = Enumerable.Range(0, neuronCount).Select(x => new Neuron()).ToArray();
        }

        public Layer(int neuronCount, Layer nextLayer)
        {
            Neurons = Enumerable.Range(0, neuronCount).Select(x => new Neuron(nextLayer.Neurons)).ToArray();

            // TODO optimize
            var allWeights = Neurons.SelectMany(x => x.OutputWeights);
            foreach (var neuron in nextLayer.Neurons)
            {
                var weights = allWeights.Where(x => x.RightNeuron.Equals(neuron)).ToArray();
                neuron.SetInputWeights(weights);
            }
        }

        public void SetValues(int[] values)
        {
            // todo only input layer
            if (values == null || values.Length != Neurons.Length)
                throw new ArgumentException("values == null || values.Length != Neurons.Length");

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].SetValue(values[i]);
            }
        }

        public double[] GetValues()
        {
            // todo only output layer
            return Neurons.Select(x => x.Value).ToArray();
        }

        public void CalculateValues()
        {
            foreach (Neuron neuron in Neurons)
                neuron.CalculateValue();
        }

        public double GetMse(double[] idealValues)
        {
            // todo only output layer
            if (idealValues == null || idealValues.Length != Neurons.Length)
                throw new ArgumentException("idealValues == null || idealValues.Length != Neurons.Length");

            double result = 0;
            for (int i = 0; i < idealValues.Length; i++)
            {
                result += Math.Pow(idealValues[0] - Neurons[i].Value, 2);
            }

            return result / idealValues.Length;
        }

        public void CalcSigma(double[] idealValues)
        {
            // todo only output layer
            if (idealValues == null || idealValues.Length != Neurons.Length)
                throw new ArgumentException("idealValues == null || idealValues.Length != Neurons.Length");

            for (int i = 0; i < idealValues.Length; i++)
            {
                Neurons[i].CaclSigma(idealValues[i]);
            }
        }

        public void CalcSigma()
        {
            // input and hidden layers
            foreach (var neuron in Neurons)
            {
                neuron.CaclSigma();
            }
        }

        public void UpdateWeights(double e, double a)
        {
            // input and hidden layers
            foreach (var neuron in Neurons)
            {
                neuron.UpdateWeights(e, a);
            }
        }
    }
}
