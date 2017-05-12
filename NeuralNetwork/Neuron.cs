using System.Linq;

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
            Value = 0;
            foreach (var val in _inputWeights)
            {
                Value += val.GetValue();
            }

            Value = Value.Normalize();
        }

        public void CaclSigma(double idealValue)
        {
            // only output
            Sigma = (idealValue - Value) * Helper.NormalizeD(Value);
        }

        public void CaclSigma()
        {
            // input and hidden layers
            Sigma = 0;
            foreach (var weight in OutputWeights)
                Sigma += weight.GetSigma();

            Sigma *= Helper.NormalizeD(Value);
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
