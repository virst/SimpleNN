using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SimpleNN
{
    public partial class Form1 : Form
    {
        private List<Point> points = new List<Point>();
        Random rnd = new Random();
        Bitmap img;
        Bitmap pimg;
        int w;
        int h;
        public Form1()
        {
            InitializeComponent();
            Form1_Resize(null, null);
            timer1.Enabled = true;
        }

        private NeuralNetwork nn;

        private void Form1_Load(object sender, EventArgs e)
        {
            Func<Double, Double> sigmoid = x => 1 / (1 + Math.Exp(-x));
            Func<Double, Double> dsigmoid = y => y * (1 - y);
            //nn = new NeuralNetwork(0.01, sigmoid, dsigmoid, 2, 5, 5, 2);
            nn = new NeuralNetwork(0.01, sigmoid, dsigmoid, 2, 15, 15, 2);
        }



        private void panel1_Paint()
        {
            var g = Graphics.FromImage(img);

            if (points.Count > 0)
            {
                for (int k = 0; k < 10000; k++)
                {
                    Point p = points[rnd.Next(0, points.Count)];
                    double nx = (double)p.x / w - 0.5;
                    double ny = (double)p.y / h - 0.5;
                    nn.feedForward(new double[] { nx, ny });
                    double[] targets = new double[2];
                    if (p.type == 0) targets[0] = 1;
                    else targets[1] = 1;
                    nn.backpropagation(targets);
                }
            }


            for (int i = 0; i < w / 8; i++)
            {
                for (int j = 0; j < h / 8; j++)
                {
                    double nx = (double)i / w * 8 - 0.5;
                    double ny = (double)j / h * 8 - 0.5;
                    double[] outputs = nn.feedForward(new double[] { nx, ny });
                    double green = Math.Max(0, Math.Min(1, outputs[0] - outputs[1] + 0.5));
                    double blue = 1 - green;
                    green = 0.3 + green * 0.5;
                    blue = 0.5 + blue * 0.5;
                    Color color = Color.FromArgb((255 << 24) | (100 << 16) | ((int)(green * 255) << 8) | (int)(blue * 255));
                    //color = Color.FromArgb(255, color);
                    pimg.SetPixel(i, j, color);
                }
            }

            g.DrawImage(pimg, new Rectangle(0, 0, w, h));

            foreach (var p in points)
                g.FillEllipse(new SolidBrush(p.type == 0 ? Color.Green : Color.Blue), new Rectangle(p.x - 10, p.y - 10, 20, 20));
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            panel1_Paint();
            panel1.Refresh();
        }



        private void panel1_MouseClick(object sender, MouseEventArgs e)
        {
            int type = 0;
            if (e.Button == MouseButtons.Right) type = 1;
            points.Add(new Point(e.X, e.Y, type));
            panel1_Paint();
            panel1.Refresh();
        }

        private void Form1_Resize(object sender, EventArgs e)
        {
            w = panel1.Width;
            h = panel1.Height;
            img = new Bitmap(w, h);
            pimg = new Bitmap(w / 8, h / 8);
            panel1.Image = img;
        }
    }
}
