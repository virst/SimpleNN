using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    public class Layer
    {
        public int size;
        public double[] neurons;
        public double[] biases;
        public double[][] weights;

        public Layer(int size, int nextSize)
        {
            this.size = size;
            neurons = new double[size];
            biases = new double[size];
            weights = new double[size][];
            for (int i = 0; i < weights.Length; i++) weights[i] = new double[nextSize];
        }
    }
}
