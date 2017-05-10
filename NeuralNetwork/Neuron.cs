using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Neuron
    {
        private NeuronWeight[] _inputWeights;

        public NeuronWeight[] OutputWeights { get; private set; }

        public double Value { get; private set; }

        public double Sigma { get; private set; }

        public Neuron()
        {
        }

        public Neuron(Neuron[] neuronsInNextLayer)
        {
            OutputWeights = neuronsInNextLayer.Select(x => new NeuronWeight(this, x)).ToArray();
        }

        public void SetInputWeights(NeuronWeight[] inputWeights)
        {
            _inputWeights = inputWeights;
        }

        public void SetValue(double value)
        {
            Value = value;
        }

        public void CalculateValue()
        {
            // Check input layer
            if (_inputWeights == null)
                return;

            Value = _inputWeights.Select(x => x.GetValue()).Sum().Normalize();
        }

        public void CaclSigma(double idealValue)
        {
            // only output
            Sigma = (idealValue - Value) * Helper.NormalizeD(Value);
        }

        public void CaclSigma()
        {
            // input and hidden layers
            Sigma = Helper.NormalizeD(Value) * OutputWeights.Select(x => x.GetSigma()).Sum();
        }

        public void UpdateWeights(double e, double a)
        {
            foreach (var weight in OutputWeights)
            {
                weight.Update(Value, e, a);
            }
        }
    }
}
