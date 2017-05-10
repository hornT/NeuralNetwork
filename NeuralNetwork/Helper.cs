using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public static class Helper
    {
        public static double GetNextDouble(this Random random, double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        public static double Normalize(this double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double NormalizeD(double val)
        {
            return (1 - val) * val;
        }
    }
}
