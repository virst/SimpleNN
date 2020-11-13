using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNN
{
    public class NeuralNetwork
    {
        private double learningRate;
        private Layer[] layers;
        private Func<Double, Double> activation;
        private Func<Double, Double> derivative;

        Random rnd = new Random();

        public NeuralNetwork(double learningRate, Func<Double, Double> activation, Func<Double, Double> derivative, params int[] sizes)
        {
            this.learningRate = learningRate;
            this.activation = activation;
            this.derivative = derivative;
            layers = new Layer[sizes.Length];
            for (int i = 0; i < sizes.Length; i++)
            {
                int nextSize = 0;
                if (i < sizes.Length - 1) nextSize = sizes[i + 1];
                layers[i] = new Layer(sizes[i], nextSize);
                for (int j = 0; j < sizes[i]; j++)
                {
                    layers[i].biases[j] = rnd.NextDouble() * 2.0 - 1.0;
                    for (int k = 0; k < nextSize; k++)
                    {
                        layers[i].weights[j][k] = rnd.NextDouble() * 2.0 - 1.0;
                    }
                }
            }
        }

        public double[] feedForward(double[] inputs)
        {
            //System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.Length);
            inputs.CopyTo(layers[0].neurons, 0);
            for (int i = 1; i < layers.Length; i++)
            {
                Layer l = layers[i - 1];
                Layer l1 = layers[i];
                for (int j = 0; j < l1.size; j++)
                {
                    l1.neurons[j] = 0;
                    for (int k = 0; k < l.size; k++)
                    {
                        l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                    }
                    l1.neurons[j] += l1.biases[j];
                    l1.neurons[j] = activation.Invoke(l1.neurons[j]);
                }
            }
            return layers[layers.Length - 1].neurons;
        }

        public void backpropagation(double[] targets)
        {
            double[] errors = new double[layers[layers.Length - 1].size];
            for (int i = 0; i < layers[layers.Length - 1].size; i++)
            {
                errors[i] = targets[i] - layers[layers.Length - 1].neurons[i];
            }
            for (int k = layers.Length - 2; k >= 0; k--)
            {
                Layer l = layers[k];
                Layer l1 = layers[k + 1];
                double[] errorsNext = new double[l.size];
                double[] gradients = new double[l1.size];
                for (int i = 0; i < l1.size; i++)
                {
                    gradients[i] = errors[i] * derivative.Invoke(layers[k + 1].neurons[i]);
                    gradients[i] *= learningRate;
                }
                double[,] deltas = new double[l1.size,l.size];
                for (int i = 0; i < l1.size; i++)
                {
                    for (int j = 0; j < l.size; j++)
                    {
                        deltas[i,j] = gradients[i] * l.neurons[j];
                    }
                }
                for (int i = 0; i < l.size; i++)
                {
                    errorsNext[i] = 0;
                    for (int j = 0; j < l1.size; j++)
                    {
                        errorsNext[i] += l.weights[i][j] * errors[j];
                    }
                }
                errors = new double[l.size];
                //System.arraycopy(errorsNext, 0, errors, 0, l.size);
                errorsNext.CopyTo(errors, 0);
                double[][] weightsNew = new double[l.weights.Length][]; for (int i = 0; i < weightsNew.Length; i++) weightsNew[i] = new double[l.weights[0].Length];
                for (int i = 0; i < l1.size; i++)
                {
                    for (int j = 0; j < l.size; j++)
                    {
                        weightsNew[j][i] = l.weights[j][i] + deltas[i,j];
                    }
                }
                l.weights = weightsNew;
                for (int i = 0; i < l1.size; i++)
                {
                    l1.biases[i] += gradients[i];
                }
            }
        }
    }
}
