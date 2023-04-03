using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class nnDynamicManager : MonoBehaviour
{

    int Inodes;
    int Hnodes;
    int Onodes;

    Matrix weights_ih;
    Matrix weights_ho;
    Matrix bias_h;
    Matrix bias_o;

    List<Matrix> weights;
    List<Matrix> bias;

    Matrix[] Layers;
    Matrix[] LayersDelta;

    Matrix Hidden;
    Matrix Output;

    Matrix HiddenDelta;
    Matrix OutputDelta;

    float learningRate = 0.02f;


    public nnDynamicManager(int[] layerMap)
    {
        weights = new List<Matrix>();
        bias = new List<Matrix>();
        Layers = new Matrix[layerMap.Length-1];
        LayersDelta = new Matrix[layerMap.Length-1];

        for (int i = 0; i < layerMap.Length-1; i++)
        {

            weights.Add(new Matrix(layerMap[i + 1], layerMap[i]));

        }

        for (int i = 1; i < layerMap.Length; i++)
        {

            bias.Add(new Matrix(layerMap[i], 1));
        }
        foreach (var layer in weights)
        {
            layer.Randomize();
        }
        foreach (var layer in bias)
        {
            layer.Randomize();
        }
        //weights_ih = new Matrix(Hlayer, Ilayer);
        //weights_ho = new Matrix(Olayer, Hlayer);
        //bias_h = new Matrix(Hlayer, 1);
        //bias_o = new Matrix(Olayer, 1);
        //weights_ih.Randomize();
        //weights_ho.Randomize();
        //bias_h.Randomize();
        //bias_o.Randomize();

    }


    public float[] FeedForward(float[] inputs)
    {
        for (int i = 0; i < weights.Count; i++)
        {
            if (i == 0)
            {
                if (weights[i].matrix.Length == inputs.Length)
                {
                    Layers[i] = Matrix.ElementWiseMultiply(weights[i], Matrix.FromArray(inputs));
                }
                else
                {
                    Layers[i] = Matrix.DotMultiply(weights[i], Matrix.FromArray(inputs));
                }
            }
            else
            {
                Layers[i] = Matrix.DotMultiply(weights[i], Layers[i-1]);
            }

            Layers[i].ElementWiseAdd(bias[i]);
            Layers[i].SigmoidMatrix();
            LayersDelta[i] = Layers[i];

        }

        return Matrix.toArray(Layers[Layers.Length - 1]);
    }


    public void TrainNN(Matrix inputs, Matrix answer)
    {


        Matrix outputErrors = Matrix.ElementWiseSubtract(answer, LayersDelta[LayersDelta.Length - 1]);


        Matrix gradientsO = Matrix.DeltaSigmoidMatrix(LayersDelta[LayersDelta.Length - 1]);
        gradientsO.ElementWiseMultiply(outputErrors);
        gradientsO.ScaleMultiply(learningRate);

        Matrix transposeO = Matrix.Transpose(Layers[Layers.Length-2]);
        Matrix weightsdeltaO = Matrix.DotMultiply(gradientsO, transposeO);

        weights[weights.Count - 1].ElementWiseAdd(weightsdeltaO);
        bias[bias.Count - 1].ElementWiseAdd(gradientsO);
        //=================================================================

        Matrix weights_ho_transpose;
        Matrix hidden_errors = outputErrors;

        Matrix gradients;
        Matrix transpose;
        Matrix weightsdelta;

        for (int i = Layers.Length - 1; i > 0; i--)
        {

            weights_ho_transpose = Matrix.Transpose(weights[i]);
            hidden_errors = Matrix.DotMultiply(weights_ho_transpose, hidden_errors);

            gradients = Matrix.DeltaSigmoidMatrix(Layers[i - 1]);
            gradients = Matrix.DotMultiply(gradients, hidden_errors);
            gradients.ScaleMultiply(learningRate);
            Debug.Log("i: "+i);
            if(i ==1)
            {

                transpose = Matrix.Transpose(inputs);
            }
            else
            {

                transpose = Matrix.Transpose(Layers[i - 2]);
            }
            weightsdelta = Matrix.DotMultiply(gradients, transpose);

            weights[i-1].ElementWiseAdd(weightsdelta);
            bias[i - 1].ElementWiseAdd(gradients);

        }













       
                //minus the ouput from the answer
                //Matrix outputErrors = Matrix.ElementWiseSubtract(answer, OutputDelta);
                /*
                Matrix gradients = Matrix.DeltaSigmoidMatrix(OutputDelta);
                gradients.ElementWiseMultiply(outputErrors);
                gradients.ScaleMultiply(learningRate);

                Matrix hidden_transpose = Matrix.Transpose(Hidden);
                Matrix weights_ho_delta = Matrix.DotMultiply(gradients, hidden_transpose);

                weights_ho.ElementWiseAdd(weights_ho_delta);
                bias_o.ElementWiseAdd(gradients);

                Matrix weights_ho_transpose = Matrix.Transpose(weights_ho);
                Matrix hidden_errors = Matrix.DotMultiply(weights_ho_transpose, outputErrors);

                Matrix hidden_gradient = Matrix.DeltaSigmoidMatrix(Hidden);
                hidden_gradient = Matrix.DotMultiply(hidden_gradient, hidden_errors);
                hidden_gradient.ScaleMultiply(learningRate);

                Matrix inputs_traspose = Matrix.Transpose(inputs);
                Matrix weights_ih_delta = Matrix.DotMultiply(hidden_gradient, inputs_traspose);

                weights_ih.ElementWiseAdd(weights_ih_delta);
                bias_h.ElementWiseAdd(hidden_gradient);
        
        */



    }

}
