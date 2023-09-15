using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class Settings : MonoBehaviour
{
    public int length = 0;
    public List<Activation> activations;
    public List<int> Node_Count;
    public NeuralNetwork nn;
    public Reinforcement rl;
    public bool Dropout;
    public List<float> DropoutP;
    public int epoch;
    public int batch_size;
    public float learning_rate;
    public int N;
    public float alpha;
    public float discount_rate;
    public void S()
    {
        nn = GetComponent<NeuralNetwork>();
        LA[] layers = new LA[length];
        for (int i = 0; i < length; i++)
        {
            if (i != length - 1)
                layers[i] = new LA(Node_Count[i], getActivation(i), DropoutP.Count != 0 ? DropoutP[i] : 0);
            else
                layers[i] = new LA(Node_Count[i], getLoss(i));
        }
        rl.SetStructure(layers);
    }
    public enum Activation
    {
        Identity, Binary_Step, ReLU, LeakyReLU, PReLU, ELU, Sigmoid, Softplus, Softmax, TanH, SiLU, Gaussian,MSE,BCE,CCE,MAPE
    };
    public Functions.ActivationDelegate getActivation(int a)
    {
        if (activations[a].GetHashCode() == 0)
            return nn.functions.Identity;
        else if (activations[a].GetHashCode() == 1)
            return nn.functions.BinaryStep;
        else if(activations[a].GetHashCode() == 2)
            return nn.functions.ReLU;
        else if (activations[a].GetHashCode() == 3)
            return nn.functions.LeakyReLU;
        else if (activations[a].GetHashCode() == 4)
            return nn.functions.PReLU;
        else if (activations[a].GetHashCode() == 5)
            return nn.functions.ELU;
        else if (activations[a].GetHashCode() == 6)
            return nn.functions.Sigmoid;
        else if (activations[a].GetHashCode() == 7)
            return nn.functions.Softplus;
        else if (activations[a].GetHashCode() == 8)
            return nn.functions.Softmax;
        else if (activations[a].GetHashCode() == 9)
            return nn.functions.TanH;
        else if (activations[a].GetHashCode() == 10)
            return nn.functions.SiLU;
        else if (activations[a].GetHashCode() == 11)
            return nn.functions.Gaussian;
        else
            return null;
    }
    public Functions.LossDelegate getLoss(int a)
    {
        if (activations[a].GetHashCode() == 12)
            return nn.functions.MSE;
        else if (activations[a].GetHashCode() == 13)
            return nn.functions.BCE;
        else if (activations[a].GetHashCode() == 14)
            return nn.functions.CCE;
        else if (activations[a].GetHashCode() == 15)
            return nn.functions.MAPE;
        else
            return null;
    }
}
