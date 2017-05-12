using System;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Layer
    {
        Neuron[] Neurons { get; }

        private int _procCount;

        private int[] _leftBorders;
        private int[] _rightBorders;

        public Action CalculateValues { get; private set; }

        public Action<double, double> UpdateWeights { get; private set; }

        public Action<double[]> SetValues { get; private set; }

        public Layer(int neuronCount)
        {
            Neurons = Enumerable.Range(0, neuronCount).Select(x => new Neuron()).ToArray();

            InitPortions();
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

            InitPortions();
        }

        private void InitPortions()
        {
            _procCount = Environment.ProcessorCount;
            if (Neurons.Length <= _procCount)
            {
                CalculateValues = CalculateValuesSmall;
                UpdateWeights = UpdateWeightsSmall;
                SetValues = SetValuesSmall;
            }
            else
            {
                _leftBorders = new int[_procCount];
                _rightBorders = new int[_procCount];

                int portion = Neurons.Length / _procCount;
                for (int i = 0; i < _procCount; i++)
                {
                    _leftBorders[i] = portion * i;
                    _rightBorders[i] = Math.Min(portion * (i + 1), Neurons.Length);
                }

                CalculateValues = CalculateValuesBig;
                UpdateWeights = UpdateWeightsBig;
                SetValues = SetValuesBig;
            }            
        }

        public void SetValuesSmall(double[] values)
        {
            // todo only input layer

            Parallel.For(0, Neurons.Length, i =>
            {
                Neurons[i].SetValue(values[i]);
            });
        }

        public void SetValuesBig(double[] values)
        {
            // todo only input layer

            Parallel.For(0, _procCount, i =>
            {
                for (int j = _leftBorders[i]; j < _rightBorders[i]; j++)
                {
                    Neurons[j].SetValue(values[j]);
                }
            });
        }

        public double[] GetValues()
        {
            // todo only output layer
            return Neurons.Select(x => x.Value).ToArray();
        }

        private void CalculateValuesSmall()
        {
            Parallel.ForEach(Neurons, neuron => {
                neuron.CalculateValue();
            });
        }

        private void CalculateValuesBig()
        {
            Parallel.For(0, _procCount, i =>
            {
                for (int j = _leftBorders[i]; j < _rightBorders[i]; j++)
                {
                    Neurons[j].CalculateValue();
                }
            });
        }

        public double GetMse(int[] idealValues)
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

        public void CalcSigma(int[] idealValues)
        {
            // todo only output layer
            if (idealValues == null || idealValues.Length != Neurons.Length)
                throw new ArgumentException("idealValues == null || idealValues.Length != Neurons.Length");

            Parallel.For(0, idealValues.Length, i =>
            {
                Neurons[i].CaclSigma(idealValues[i]);
            });
        }

        private void UpdateWeightsSmall(double e, double a)
        {
            Parallel.ForEach(Neurons, neuron => {
                neuron.CaclSigma();
                neuron.UpdateWeights(e, a);
            });
        }

        private void UpdateWeightsBig(double e, double a)
        {
            Parallel.For(0, _procCount, i =>
            {
                for (int j = _leftBorders[i]; j < _rightBorders[i]; j++)
                {
                    Neurons[j].CaclSigma();
                    Neurons[j].UpdateWeights(e, a);
                }
            });
        }
    }
}
