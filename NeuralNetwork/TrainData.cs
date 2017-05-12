namespace NeuralNetwork
{
    public class TrainData
    {
        public readonly double[] Input;

        public readonly int[] Output;

        public TrainData(double[] input, int[] output)
        {
            Input = input;
            Output = output;
        }
    }
}
