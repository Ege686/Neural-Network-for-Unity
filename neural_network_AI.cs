using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class neural_network_AI : MonoBehaviour
{
    public List<List<float>> inputs = new List<List<float>>();
    public List<List<List<float>>> weights = new List<List<List<float>>>();
    public List<float> bias = new List<float>();
    public List<List<float>> target_output = new List<List<float>>();
    public List<List<float>> temporary_predicts = new List<List<float>>();
    public List<List<float>> temporary_predicts_sigmoid = new List<List<float>>();
    public List<float> loss_values = new List<float>();
    public List<List<List<float>>> hidden_layers = new List<List<List<float>>>();
    public List<List<List<float>>> delta_values = new List<List<List<float>>>();
    public float lr = 0.5f;
    List<List<float>> input = new List<List<float>>();
    List<List<float>> aot = new List<List<float>>();
    List<List<float>> aott = new List<List<float>>();
    public bool predicted;
    public float alt_sınır;
    public float üst_sınır;

    List<string> activation_function = new List<string>();

    public float sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

    public float sigmoid_deriv(float z)
    {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    public float sigmoid_derivz(float z)
    {
        return z * (1 - z);
    }
    public float Relu_deriv(float z)
    {
        float a = z;
        if (z < 0)
            a=0;
        if (z >=0)
            a=1;
        return a;
    }
    public float loss_function(float x, int sample,int stage, int outputCounter, List<float> output,float delta_value)
    {
        delta_values[sample][stage+1][outputCounter]= delta(sample, outputCounter,stage, output);
        return delta_values[sample][stage + 1][outputCounter] * x;
    }
    public float loss_function2(float x, int sample, int stage, int outputCounter,  List<float> output, List<float> output2, List<List<float>> weights,float delta_value, List<float> delta_value2)
    {
        delta_values[sample][stage + 1][outputCounter] = delta2(sample, outputCounter,stage, output, output2, weights,delta_value2);
        return delta_values[sample][stage + 1][outputCounter] * x;
    }
    public float delta(int sample,int outputCounter,int stage,List<float> output)
    {
        float Delta; 
        Delta = (output[outputCounter] - target_output[sample][outputCounter]) * sigmoid_derivz(output[outputCounter]);
        return Delta;
    }
    public float delta2(int sample, int outputCounter, int stage,List<float> output, List<float> output2, List<List<float>> weights,List<float> delta_value)
    {
        float Delta = 0;
        for (int i = 0; i < output.Count; i++)
        {
            Delta += delta_value[i] * weights[outputCounter][i];
        }
        return Delta * Relu_deriv(output2[outputCounter]);
    }
    public void sum(int iii,int iiii)
    {
        for (int iiiii = 0; iiiii < hidden_layers[iii][iiii+1].Count; iiiii++)
        {
            float sum = 0;
            for (int i = 0; i < hidden_layers[iii][iiii].Count; i++)
            {
                float x = hidden_layers[iii][iiii][i] * weights[iiii][i][iiiii];
                sum += x;
            }
            sum += bias[iiii];
            if (activation_function[iiii]=="relu")
            {/*
                if(sum<0)
                    hidden_layers[iii][iiii + 1][iiiii] = 0;
                else*/
                    hidden_layers[iii][iiii + 1][iiiii] = sum;
            }
            if (activation_function[iiii] == "sigmoid")
                hidden_layers[iii][iiii + 1][iiiii] = sigmoid(sum);

        }
    }

    public void procces(int Epochs, List<List<float>> inputa)
    {
        inputs = inputa;
        for (int i = 0; i < inputs.Count; i++)
        {
            hidden_layers[i][0] = inputs[i];
        }
        delta_values = hidden_layers;
        predicted = true;
        for (int epochs = 0; epochs < Epochs; epochs++)
        {
            for (int sample = 0; sample < inputs.Count; sample++)
            {

                for (int stage = 0; stage < weights.Count; stage++)
                {
                    sum(sample, stage);
                }
                for (int back_stage = weights.Count - 1; back_stage > -1; back_stage--)
                {
                    if (activation_function[back_stage]=="sigmoid")
                    {
                        for (int inputCounter = 0; inputCounter < hidden_layers[sample][back_stage].Count; inputCounter++)
                        {
                            for (int outputCounter = 0; outputCounter < hidden_layers[sample][back_stage + 1].Count; outputCounter++)
                            {
                                weights[back_stage][inputCounter][outputCounter] -= (lr * loss_function(hidden_layers[sample][back_stage][inputCounter], sample,back_stage, outputCounter, hidden_layers[sample][back_stage + 1], delta_values[sample][back_stage + 1][outputCounter]));
                                bias[back_stage] -= (lr * loss_function(1, sample, back_stage, outputCounter, hidden_layers[sample][back_stage + 1], delta_values[sample][back_stage + 1][outputCounter]))/ hidden_layers[sample][back_stage].Count;
                            }
                        }
                    }
                    if (activation_function[back_stage] == "relu")
                    {
                        for (int inputCounter = 0; inputCounter < hidden_layers[sample][back_stage].Count; inputCounter++)
                        {
                            for (int outputCounter = 0; outputCounter < hidden_layers[sample][back_stage + 1].Count; outputCounter++)
                            {
                                weights[back_stage][inputCounter][outputCounter] -= (lr * loss_function2(hidden_layers[sample][back_stage][inputCounter], sample,back_stage, outputCounter, hidden_layers[sample][back_stage + 2], hidden_layers[sample][back_stage + 1], weights[back_stage + 1], delta_values[sample][back_stage + 1][outputCounter], delta_values[sample][back_stage + 2]));
                                bias[back_stage] -= (lr * loss_function2(1, sample,back_stage, outputCounter, hidden_layers[sample][back_stage + 2], hidden_layers[sample][back_stage + 1], weights[back_stage + 1], delta_values[sample][back_stage + 1][outputCounter], delta_values[sample][back_stage + 2])) / hidden_layers[sample][back_stage].Count;
                            }
                        }
                    }
                }
            }
        }
    }

    public void predict(List<List<float>> input)
    {
        for (int i = 0; i < input.Count; i++)
        {
            hidden_layers[i][0] = inputs[i];
        }
        for (int sample = 0; sample < hidden_layers.Count; sample++)
        {
            for (int stage = 0; stage < weights.Count; stage++)
            {
                sum(sample, stage);
            }
        }
        predicted = true;
    }

    public List<float> value_of_predict()
    {
        return hidden_layers[0][hidden_layers[0].Count-1];
    }
    public void get_weights()
    {
        for(int i = 0; i < weights.Count; i++)
        {
            for(int ii = 0; ii < weights[i].Count; ii++)
            {
                for(int iii=0;iii<weights[i][ii].Count; iii++)
                {
                    print("Name of the agent:"+gameObject.name+" Stage of weight:" + i + " input number:" + ii + " output number:" + iii + " =" + weights[i][ii][iii]);
                }
            }
        }
    }
    public void attach_values2(List<List<float>> inputaa, List<List<float>> target_outputs)
    {
        inputs = inputaa;
        target_output = target_outputs;
    }

    public void attach_values(string function)
    {
        for (int iii = 0; iii < target_output.Count; iii++)
        {
            temporary_predicts.Add(new List<float>());
            temporary_predicts_sigmoid.Add(new List<float>());
            for (int i = 0; i < target_output[iii].Count; i++)
            {
                temporary_predicts[iii].Add(0);
                temporary_predicts_sigmoid[iii].Add(0);
            }
        }
        for (int i = 0; i < inputs.Count; i++)
        {
            hidden_layers[i].Insert(0, inputs[i]);
            hidden_layers[i].Insert(hidden_layers[i].Count, temporary_predicts_sigmoid[i]);
        }
        for (int ah = 0; ah < hidden_layers[0].Count-1; ah++)
        {
            weights.Add(new List<List<float>>());
        }
        for (int iii = 0; iii < weights.Count; iii++)
        {
            if (hidden_layers[0].Count != 0)
            {
                for (int i = 0; i < hidden_layers[0][iii].Count; i++)
                {
                    List<float> a = new List<float>();
                    for (int ii = 0; ii < hidden_layers[0][iii+1].Count; ii++)
                    {
                        a.Add(Random.Range(0.1f, 1f));
                    }
                    weights[iii].Add(a);
                }
            }
            else
            {
                for (int i = 0; i < inputs[0].Count; i++)
                {
                    List<float> a = new List<float>();
                    for (int ii = 0; ii < target_output[0].Count; ii++)
                    {
                        a.Add(Random.Range(0.1f, 1f));
                    }
                    weights[iii].Add(a);
                }
            }
        }
        for (int iii = 0; iii < weights.Count; iii++)
        {
            if (hidden_layers[0].Count != 0)
            {
                bias.Add(Random.Range(0.05f, 0.5f));
            }
            else
            {
                bias.Add(Random.Range(0.05f, 0.5f));
            }
        }
        activation_function.Add(function);
    }

    public void Dense(int node_count, List<List<float>> input,string function)
    {
        for (int i = 0; i < input.Count; i++)
        {
            if (hidden_layers.Count != input.Count)
            {
                hidden_layers.Add(new List<List<float>>());
            }
            if (node_count != 0)
                hidden_layers[i].Add(new List<float>());
            for (int ii = 0; ii < node_count; ii++)
            {
                hidden_layers[i][hidden_layers[i].Count-1].Add(0);
            }
        }
        activation_function.Add(function);
    }
}
