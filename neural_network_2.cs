using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;

public class neural_network_2 : MonoBehaviour
{
    List<List<float>> input = new List<List<float>>();
    List<List<float>> predicts = new List<List<float>>();
    List<List<List<float>>> hidden_layers = new List<List<List<float>>>();
    List<List<List<float>>> structure = new List<List<List<float>>>();
    List<List<float>> target_output = new List<List<float>>();
    List<float> bias = new List<float>();
    List<List<List<float>>> weights = new List<List<List<float>>>();
    List<List<List<float>>> weights_tempo = new List<List<List<float>>>();
    bool setted;
    bool added;
    List<string> function_name = new List<string>();
    List<List<List<float>>> delta_values = new List<List<List<float>>>();
    public float lr = 0.5f;
    float sigmoid(float z)
    {
        return 1 / (1 + Mathf.Exp(-z));
    }
    float sigmoid_deriv(float Z)
    {
        return Z*(1-Z);
    }
    float relu(float z)
    {
        if (z > 0)
            return z;
        else
            return 0;
    }
    float relu_deriv(float z)
    {
        if (z > 0)
            return 1;
        else
            return 0;
    }
    public void start(List<List<float>> input, List<List<float>> output)
    {
        for (int sample = 0; sample < input.Count; sample++)
        {
            this.input.Add(new List<float>());
            for (int i = 0; i < input[sample].Count; i++)
            {
                this.input[sample].Add(input[sample][i]);
            }
            target_output.Add(new List<float>());
            for (int i = 0; i < output[sample].Count; i++)
            {
                target_output[sample].Add(output[sample][i]);
            }
        }
    }
    public void attach_values(List<List<float>> input, List<List<float>> output)
    {
        for(int i = 0; i < input[0].Count; i++)
        {
            structure[0][0][i] = input[0][i];
        }
        for (int i = 0; i < output[0].Count; i++)
        {
            target_output[0][i] = output[0][i];
        }
    }
    public void add_hidden_layer(float hidden_layers_count,string activion_name)
    {
        function_name.Add(activion_name);
        bias.Add(0);
        if (!added)
        {
            for (int sample = 0; sample < input.Count; sample++)
            {
                hidden_layers.Add(new List<List<float>>());
                delta_values.Add(new List<List<float>>());
            }
        }
        for (int sample=0;sample<input.Count;sample++)
        {
            hidden_layers[sample].Add(new List<float>());
            for (int i=0; i < hidden_layers_count; i++)
            {
                hidden_layers[sample][hidden_layers[sample].Count-1].Add(0);
            }
        }
        added = true;
    }
    public List<float> scale(List<float> input)
    {
        float scaled_value;
        float highest = input[0];
        float lowest = input[0];
        List<float> new_input = new List<float>();
        for (int i = 0; i < input.Count; i++)
        {
            new_input.Add(input[i]);
            if (highest < input[i])
                highest = input[i];
            if (lowest > input[i])
                lowest = input[i];
        }
        for (int i = 0; i < input.Count; i++)
        {
            scaled_value = (input[i] - lowest) / (highest - lowest);
            new_input[i] = (scaled_value);
        }
        return new_input;
    }
    public List<float> get_predicted_values()
    {
        return structure[0][structure[0].Count - 1];
    }

    public void set_structure(List<List<float>> output, string activion_name)
    {
        if (setted)
            return;

        function_name.Add(activion_name);
        bias.Add(0);
        for (int sample = 0; sample < input.Count; sample++)
        {
            structure.Add(new List<List<float>>());
            structure[sample].Add(new List<float>());
            for (int i = 0; i < input[sample].Count; i++)
            {
                structure[sample][0].Add(input[sample][i]);
            }
            for(int i = 0; i < hidden_layers[sample].Count; i++)
            {
                structure[sample].Add(new List<float>());
                delta_values[sample].Add(new List<float>());
                for (int ii = 0; ii < hidden_layers[sample][i].Count; ii++)
                {
                    structure[sample][structure[sample].Count - 1].Add(0);
                    delta_values[sample][delta_values[sample].Count - 1].Add(0);
                }
            }
            predicts.Add(new List<float>());
            structure[sample].Add(new List<float>());
            delta_values[sample].Add(new List<float>());
            for (int i = 0; i < target_output[sample].Count; i++)
            {
                structure[sample][structure[sample].Count-1].Add(target_output[sample][i]);
                delta_values[sample][delta_values[sample].Count-1].Add(target_output[sample][i]);
            }
            bias.Add(0);
        }
        for (int layer = 0; layer < structure[0].Count-1; layer++)
        {
            weights.Add(new List<List<float>>());
            weights_tempo.Add(new List<List<float>>());
            for (int left = 0; left < structure[0][layer].Count; left++)
            {
                weights[layer].Add(new List<float>());
                for (int right = 0; right < structure[0][layer + 1].Count; right++)
                {
                    weights[layer][left].Add(Random.Range(0f,.99f));
                }
            }
        }
        setted = true;
    }

    void feed_forward(int sample)
    {
        for (int layer=0; layer< structure[sample].Count-1; layer++)
        {
            for (int right = 0; right < structure[sample][layer + 1].Count; right++)
            {
                float sum = 0;
                for (int left = 0; left < structure[sample][layer].Count; left++)
                {
                    sum += structure[sample][layer][left] * weights[layer][left][right]+bias[layer];
                }
                if (function_name[layer] == "relu")
                    structure[sample][layer + 1][right] = relu(sum);
                if (function_name[layer] == "sigmoid")
                    structure[sample][layer + 1][right] = sigmoid(sum);
            }
        }
    }
    public void predict()
    {
        feed_forward(0);
    }
    public void procces(int epochs)
    {
        for(int sample = 0; sample < structure.Count; sample++)
        {
            for(int repeats = 0; repeats < epochs; repeats++)
            {
                feed_forward(sample);
                for (int layer = structure[sample].Count - 1; layer > 0; layer--)
                {
                    for(int i = 0; i < structure[sample][layer].Count; i++)
                    {
                        if (layer == structure[sample].Count - 1)
                        {
                            if (function_name[layer - 1] == "sigmoid")
                            {
                                delta_values[sample][layer - 1][i] = -(target_output[sample][i] - structure[sample][layer][i]) * sigmoid_deriv(structure[sample][layer][i]);
                            }

                            if (function_name[layer - 1] == "relu")
                                delta_values[sample][layer - 1][i] = -(target_output[sample][i] - structure[sample][layer][i]) * relu_deriv(structure[sample][layer][i]);
                        }
                        else
                        {
                            float sum = 0;
                            for (int right = 0; right < delta_values[sample][layer].Count; right++)
                            {
                                sum += delta_values[sample][layer][right] * weights[layer][i][right];
                            }
                            if (function_name[layer - 1] == "sigmoid")
                                delta_values[sample][layer - 1][i] = sum * sigmoid_deriv(structure[sample][layer][i]);

                            if (function_name[layer - 1] == "relu")
                                delta_values[sample][layer - 1][i] = sum * relu_deriv(structure[sample][layer][i]);
                        }
                    }
                }
                for (int layer = structure[sample].Count - 1; layer > 0; layer--)
                {
                    for (int left = 0; left < structure[sample][layer - 1].Count; left++)
                    {
                        for (int right = 0; right < structure[sample][layer].Count; right++)
                        {
                            weights[layer - 1][left][right] -= lr * delta_values[sample][layer - 1][right] * structure[sample][layer - 1][left];
                            bias[layer] -= lr * delta_values[sample][layer - 1][right];
                        }
                    }
                }
            }
        }
    }

    void CreateText(string name, string variliable, bool reset)
    {
        string path = Application.dataPath + "/" + name + ".txt";
        if (reset)
            File.WriteAllText(path, "");
        File.AppendAllText(path, variliable);
    }
    string ReadText(string name, int sum)
    {
        string readPath = Application.dataPath + "/" + name + ".txt";
        List<string> values = File.ReadAllLines(readPath).ToList();
        string return_list;
        return_list = (values[sum]);
        return return_list;
    }
    public void save_weights(string name)
    {
        CreateText("weights " + name, "", true);
        int ali = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            for (int ii = 0; ii < weights[i].Count; ii++)
            {
                for (int iii = 0; iii < weights[i][ii].Count; iii++)
                {
                    string value = weights[i][ii][iii] + "\n";
                    CreateText("weights " + name, value, false);
                }
            }
        }
    }
    public void save_bias(string name)
    {
        CreateText("bias " + name, "", true);
        for (int iii = 0; iii < bias.Count; iii++)
        {
            string value = bias[iii] + "\n";
            CreateText("bias " + name, value, false);
        }
    }
    public void ChangeALittleWeights()
    {
        for (int i = 0; i < weights.Count; i++)
        {
            for (int ii = 0; ii < weights[i].Count; ii++)
            {
                for (int iii = 0; iii < weights[i][ii].Count; iii++)
                {
                    weights[i][ii][iii] = weights[i][ii][iii] * Random.Range(0.8f, 1);
                }
            }
        }
    }
    public void ChangeALittleBias()
    {
        for (int iii = 0; iii < bias.Count; iii++)
        {
            bias[iii] = bias[iii] * Random.Range(0.1f, 0.9f);
        }
    }
    public void set_weights(string name)
    {
        int sum = 0;
        for (int i = 0; i < weights.Count; i++)
        {
            for (int ii = 0; ii < weights[i].Count; ii++)
            {
                for (int iii = 0; iii < weights[i][ii].Count; iii++)
                {
                    weights[i][ii][iii] = float.Parse(ReadText("weights " + name, sum));
                    sum += 1;
                }
            }
        }
    }
    public void set_bias(string name)
    {
        for (int iii = 0; iii < bias.Count; iii++)
        {
            bias[iii] = float.Parse(ReadText("bias " + name, iii));
        }
    }
}
