using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;
using System.IO;
using System.Globalization;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var nnXor = new NeuralNetwork.NeuralNetwork();
            Console.WriteLine("Xor:");
            nnXor.Train(new[]
            {
                new TrainData(new []{0, 0.0}, new []{0}),
                new TrainData(new []{0, 1.0}, new []{1}),
                new TrainData(new []{1, 0.0}, new []{1}),
                new TrainData(new []{1, 1.0}, new []{0})
            });

            Console.WriteLine($"0, 0: {nnXor.Calculate(new[] { 0.0, 0 })[0]}");
            Console.WriteLine($"0, 1: {nnXor.Calculate(new[] { 0.0, 1 })[0]}");
            Console.WriteLine($"1, 0: {nnXor.Calculate(new[] { 1.0, 0 })[0]}");
            Console.WriteLine($"1, 1: {nnXor.Calculate(new[] { 1.0, 1 })[0]}");

            Console.WriteLine("img:");
            var nnImg = new NeuralNetwork.NeuralNetwork();
            List<TrainData>[] trainData = new List<TrainData>[10];
            Parallel.For(0, 10, i =>
            {
                trainData[i] = GetTrainDataFromFile(i);
            });
            nnImg.Train(trainData.SelectMany(x => x).ToArray(), maxEpoch: 2);

            Console.ReadKey();
        }

        private static List<TrainData> GetTrainDataFromFile(int index)
        {
            List<TrainData> result = new List<TrainData>();

            var dataText = File.ReadAllText($"d:\\temp\\mnist\\src\\digits\\{index}.json");
            dataText = dataText.Remove(0, 11); // { "data": [
            dataText = dataText.Trim(']', '}');
            var digits = dataText.Split(',').Select(x => double.Parse(x, NumberStyles.Any, CultureInfo.InvariantCulture)).ToArray();

            int[] output = new int[10];
            output[index] = 1;

            for (int i = 0; i < digits.Length; i += 784)
            {
                double[] tmp = new double[784];
                Array.Copy(digits, i, tmp, 0, 784);
                result.Add(new TrainData(tmp, output));
            }

            return result;
        }

        private static async Task<List<TrainData>> GetTrainDataFromFileAsync(int index)
        {
            List<TrainData> result = new List<TrainData>();

            var dataText = await ReadTextAsync($"d:\\temp\\mnist\\src\\digits\\{index}.json");
            dataText = dataText.Remove(0, 11); // { "data": [
            dataText = dataText.Trim(']', '}');
            var digits = dataText.Split(',').Select(x => double.Parse(x, NumberStyles.Any, CultureInfo.InvariantCulture)).ToArray();

            int[] output = new int[10];
            output[index] = 1;

            for (int i = 0; i < digits.Length; i += 784)
            {
                double[] tmp = new double[784];
                Array.Copy(digits, i, tmp, 0, 784);
                result.Add(new TrainData(tmp, output));
            }

            return result;
        }

        // https://msdn.microsoft.com/ru-ru/library/jj155757(v=vs.110).aspx
        private static async Task<string> ReadTextAsync(string filePath)
        {
            using (FileStream sourceStream = new FileStream(filePath,
                FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 4096, useAsync: true))
            {
                StringBuilder sb = new StringBuilder();

                byte[] buffer = new byte[0x1000];
                int numRead;
                while ((numRead = await sourceStream.ReadAsync(buffer, 0, buffer.Length)) != 0)
                {
                    string text = Encoding.Default.GetString(buffer, 0, numRead);
                    sb.Append(text);
                }

                return sb.ToString();
            }
        }
    }
}
