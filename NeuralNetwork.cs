using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
public class NeuralNetwork : MonoBehaviour
{
    public Structure structure = new Structure();
    List<List<float>> inputs = new List<List<float>>();
    List<List<float>> outputs = new List<List<float>>();
    public Functions functions = new Functions();

    public void SetStructure(params LA[] la)
    {
        int[] layer_count = new int[la.Length];
        Functions.ActivationDelegate[] activation = new Functions.ActivationDelegate[la.Length];
        for (int i = 0; i < la.Length; i++)
        {
            layer_count[i] = la[i].LayerCount;
            activation[i] = la[i].Activation;
        }
        structure = new Structure(layer_count);
        structure.setActiavations(activation);
        structure.setLoss(la[la.Length - 1].Loss);
    }
    public void SetVariables(List<List<float>> inputs, List<List<float>> outputs)
    {
        this.inputs = inputs;
        this.outputs = outputs;
    }
    public void SetVariables(List<List<float>> inputs)
    {
        this.inputs = inputs;
        SetVariables(0);
    }
    public void SetVariables(int b)
    {
        if (inputs[b].Count == structure.Layer(0).NodeCount)
            structure.SetInputs(inputs[b]);
        else
            Debug.LogError("First Layer's Node Count(" + structure.Layer(0).NodeCount + ") and inputs' " + b + "th list's variables count(" + inputs[b].Count + ") does not match!");

        if (outputs[b].Count == structure.Layer(structure.LayerCount - 1).NodeCount)
            structure.SetTargets(outputs[b]);
        else
            Debug.LogError("Last Layer's Node Count(" + structure.Layer(structure.LayerCount - 1).NodeCount + ") and targets' " + b + "th list's variables count(" + outputs[b].Count + ") does not match!");
    }
    public void Predict()
    {
        structure.Forward();
    }
    public void Train(int epoch, int batch_size,float lr, float alpha)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size,alpha);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr,batch_size);
                structure.UpdateBias(-lr,batch_size);
            }
        }
    }
    public List<float> Scale(List<float> input)
    {
        float biggest = 0;
        float smallest = Mathf.Infinity;
        List<float> result = new List<float>();
        for (int n = 0; n < input.Count; n++)
        {
            float x = input[n];
            result.Add(x);
            if (x > biggest)
                biggest = x;
            if (x < smallest)
                smallest = x;
        }
        float dif = biggest - smallest;
        for (int n = 0; n < input.Count; n++)
        {
            result[n] = (result[n] - smallest) / dif;
        }
        return result;
    }
    public float output(int o)
    {
        return structure.Layer(structure.LayerCount - 1).Node(o).Value;
    }
    void CreateText(string name, string variliable, bool reset)
    {
        string path = Application.dataPath + "/Saves/" + name + ".txt";
        if (reset)
            File.WriteAllText(path, "");
        File.AppendAllText(path, variliable);
    }
    string ReadText(string name, int sum)
    {
        string readPath = Application.dataPath + "/Saves/" + name + ".txt";
        List<string> values = File.ReadAllLines(readPath).ToList();
        string return_list;
        return_list = (values[sum]);
        return return_list;
    }
    public void SaveAll(string name)
    {
        int weight_count = structure.WeightCount;
        CreateText("Save " + name, "" + weight_count + "\n", true);
        SaveWeights(name);
        SaveBias(name);
    }
    public void SetSave(string name)
    {
        SetWeights(name);
        SetBiases(name);
    }
    public void SaveWeights(string name)
    {
        for (int l = 0; l < structure.LayerCount - 1; l++)
        {
            for (int n = 0; n < structure.Layer(l).NodeCount; n++)
            {
                for (int w = 0; w < structure.Layer(l).NextNodeCount; w++)
                {
                    string value = structure.Layer(l).Node(n).Weight(w) + "\n";
                    CreateText("Save " + name, value, false);
                }
            }
        }
    }
    public void SaveBias(string name)
    {
        for (int l = 0; l < structure.LayerCount - 1; l++)
        {
            string value = structure.Layer(l).Bias + "\n";
            CreateText("Save " + name, value, false);
        }
    }
    public void SetWeights(string name)
    {
        int sum = 1;
        for (int l = 0; l < structure.LayerCount - 1; l++)
        {
            for (int n = 0; n < structure.Layer(l).NodeCount; n++)
            {
                for (int w = 0; w < structure.Layer(l).NextNodeCount; w++)
                {
                    structure.Layer(l).Node(n).SetWeight(w, float.Parse(ReadText("Save " + name, sum)));
                    sum += 1;
                }
            }
        }
    }
    public void SetBiases(string name)
    {
        int line = int.Parse(ReadText("Save " + name, 0)) + 1;
        for (int l = 0; l < structure.LayerCount - 1; l++)
        {
            structure.Layer(l).setBias(float.Parse(ReadText("Save " + name, line)));
            line += 1;
        }
    }
    public void ChangeLittle(bool weights, bool biases)
    {
        if (weights)
            structure.ChangeLittleWeight();
        if (biases)
            structure.ChangeLittleBiases();
    }
}
