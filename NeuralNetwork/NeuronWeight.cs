using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class NeuronWeight
    {
        private static Random rnd = new Random();

        double _weight;

        double _delta;

        public Neuron LeftNeuron { get; }

        public Neuron RightNeuron { get; }

        public NeuronWeight(Neuron leftNeuron, Neuron rightNeuron)
        {
            LeftNeuron = leftNeuron;
            RightNeuron = rightNeuron;
            _weight = rnd.GetNextDouble(-1, 1);
        }

        public double GetValue()
        {
            return LeftNeuron.Value * _weight;
        }

        public double GetSigma()
        {
            return RightNeuron.Sigma * _weight;
        }

        public void Update(double inputValue, double e, double a)
        {
            double grad = inputValue * RightNeuron.Sigma;
            _delta = e * grad + a * _delta;
            _weight += _delta;
        }
    }
}
