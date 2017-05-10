using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var nn = new NeuralNetwork.NeuralNetwork();
            nn.Train(new []
            {
                new TrainData(new []{0, 0}, new []{0.0}),
                new TrainData(new []{0, 1}, new []{1.0}),
                new TrainData(new []{1, 0}, new []{1.0}),
                new TrainData(new []{1, 1}, new []{0.0})
            });

            //var result = nn.Calculate(new[] { 0, 0 });
            Console.WriteLine($"0, 0: {nn.Calculate(new[] { 0, 0 })[0]}");
            Console.WriteLine($"0, 1: {nn.Calculate(new[] { 0, 1 })[0]}");
            Console.WriteLine($"1, 0: {nn.Calculate(new[] { 1, 0 })[0]}");
            Console.WriteLine($"1, 1: {nn.Calculate(new[] { 1, 1 })[0]}");

            Console.ReadKey();
        }
    }
}
