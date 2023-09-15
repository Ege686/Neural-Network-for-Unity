using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
public class NeuralNetwork : MonoBehaviour
{
    public Structure structure = new Structure();
    public List<List<float>> inputs = new List<List<float>>();
    public List<List<float>> outputs = new List<List<float>>();
    public Functions functions = new Functions();

    public void Structure(Structure structure)
    {
        this.structure = structure;
    }
    public void SetStructure(params LA[] la)
    {
        int[] layer_count = new int[la.Length];
        Functions.ActivationDelegate[] activation = new Functions.ActivationDelegate[la.Length];
        float[] p = new float[la.Length];
        for (int i = 0; i < la.Length; i++)
        {
            layer_count[i] = la[i].LayerCount;
            activation[i] = la[i].Activation;
            if (i != la.Length - 1)
                p[i] = la[i].DropoutP;
        }
        structure = new Structure(layer_count);
        structure.setActiavations(activation);
        structure.setLoss(la[la.Length - 1].Loss);
        structure.DropoutP(p);
    }
    public Structure SetStructure(Structure structure,params LA[] la)
    {
        int[] layer_count = new int[la.Length];
        Functions.ActivationDelegate[] activation = new Functions.ActivationDelegate[la.Length];
        float[] p = new float[la.Length];
        for (int i = 0; i < la.Length; i++)
        {
            layer_count[i] = la[i].LayerCount;
            activation[i] = la[i].Activation;
            if (i != la.Length - 1)
                p[i] = la[i].DropoutP;
        }
        structure = new Structure(layer_count);
        structure.setActiavations(activation);
        structure.setLoss(la[la.Length - 1].Loss);
        structure.DropoutP(p);
        return structure;
    }
    public void SetVariables(List<List<float>> inputs, List<List<float>> outputs)
    {
        this.inputs.Clear();
        for (int i = 0; i < inputs.Count; i++)
        {
            this.inputs.Add(new List<float>());
            for (int ii = 0; ii < inputs[i].Count; ii++)
            {
                this.inputs[i].Add(inputs[i][ii]);
            }
        }
        this.outputs.Clear();
        for (int i = 0; i < outputs.Count; i++)
        {
            this.outputs.Add(new List<float>());
            for (int ii = 0; ii < outputs[i].Count; ii++)
            {
                this.outputs[i].Add(outputs[i][ii]);
            }
        }
    }
    public void SetVariables(List<List<float>> inputs)
    {
        this.inputs.Clear();
        for (int i = 0; i < inputs.Count; i++)
        {
            this.inputs.Add(new List<float>());
            for (int ii = 0; ii < inputs[i].Count; ii++)
            {
                this.inputs[i].Add(inputs[i][ii]);
            }
        }
        SetVariables(0,false);
    }
    public void SetVariables(List<float> inputs)
    {
        this.inputs.Clear();
        this.inputs.Add(new List<float>());
        for (int ii = 0; ii < inputs.Count; ii++)
        {
            this.inputs[0].Add(inputs[ii]);
        }
        SetVariables(0,false);
    }
    public void SetVariables(int b)
    {
        if (inputs[b].Count == structure.Layer(0).NodeCount)
        {
            //print(b + " " + inputs.Count);
            structure.SetInputs(inputs[b]);

            //print(b + " " + inputs.Count);
        }
        else
            Debug.LogError("First Layer's Node Count(" + structure.Layer(0).NodeCount + ") and inputs' " + b + "th list's variables count(" + inputs[b].Count + ") does not match!");

        if (outputs[b].Count == structure.Layer(structure.LayerCount - 1).NodeCount)
        {
            //print(b + " " + outputs[b].Count);
            structure.SetTargets(outputs[b]);
            //print(b + " " + outputs[b].Count);
        }
        else
            Debug.LogError("Last Layer's Node Count(" + structure.Layer(structure.LayerCount - 1).NodeCount + ") and targets' " + b + "th list's variables count(" + outputs[b].Count + ") does not match!");

    }
    public void SetVariables(int b,bool a)
    {
        //Debug.Log(inputs[b].Count+" "+ structure.Layer(0).NodeCount);
        if (inputs[b].Count == structure.Layer(0).NodeCount)
        {
            structure.SetInputs(inputs[b]);

        }
        else
            Debug.LogError("First Layer's Node Count(" + structure.Layer(0).NodeCount + ") and inputs' " + b + "th list's variables count(" + inputs[b].Count + ") does not match!");
    }
    public void Predict()
    {
        structure.Forward();
    }
    public void Train(int epoch, int batch_size,float lr)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                if (i + batch_size >= inputs.Count)
                    batch_size = inputs.Count - i - 1;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size,0);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr,batch_size);
                structure.UpdateBias(-lr,batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
    }
    public void Train(int epoch, int batch_size, float lr, float alpha)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                if (i + batch_size >= inputs.Count)
                    batch_size = inputs.Count - i - 1;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size, alpha);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
    }
    public void Train(int epoch, int batch_size, float lr, float alpha, bool dropout)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                if (dropout)
                    structure.Dropout();
                if (i + batch_size >= inputs.Count)
                    batch_size = inputs.Count - i - 1;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size, alpha);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
    }
    public Structure RLTrain(int epoch, int batch_size, float lr,int n)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                if (i + batch_size >= inputs.Count && i != inputs.Count - 1)
                    batch_size = inputs.Count - i - 1;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    //SetZeroDelta(n);
                    //structure.SetLoss();
                    structure.SetDeltaValues();
                    //for (int j = 0; j < structure.LayerCount; j++)
                        //print(structure.LastLayer.Node(j).DeltaValue);
                    structure.SetWeightsD(batch_size, 0);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
        return structure;
    }
    public void RLTrain(int epoch, int batch_size, float lr, int n,float alpha)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {

                if (i + batch_size >= inputs.Count && i != inputs.Count - 1)
                    batch_size = inputs.Count - i - 1;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    SetZeroDelta(n);
                    structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size, alpha);
                    structure.SetBiasD();
                }
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
    }
    public void RLTrain(int epoch, int batch_size, float lr, int n, float alpha,bool dropout)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0; i < inputs.Count; i += batch_size)
            {
                if (dropout)
                    structure.Dropout();

                if (i + batch_size >= inputs.Count)
                    batch_size = inputs.Count - i;
                if (batch_size == 0)
                    break;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    //structure.SetLoss();
                    //SetZeroDelta(n);
                    structure.SetDeltaValues(n);
                    structure.SetWeightsD(batch_size, alpha);
                    structure.SetBiasD();
                }
                //SaveAll("kem");
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
            }
        }
    }
    public void RLTrain(int epoch, int batch_size, float lr, int n, bool dropout)
    {
        for (int e = 0; e < epoch; e++)
        {
            for (int i = 0;i < inputs.Count; i += batch_size)
            {
                if (dropout)
                    structure.Dropout();

                if (i + batch_size >= inputs.Count)
                    batch_size = inputs.Count - i;
                for (int batch = 0; batch < batch_size; batch++)
                {
                    SetVariables(batch + i);
                    Predict();
                    SetZeroDelta(n);
                    //structure.SetLoss();
                    structure.SetDeltaValues();
                    structure.SetWeightsD(batch_size, 0);
                    structure.SetBiasD();
                }
                //print("b " + batch_size);
                structure.UpdateWeights(-lr, batch_size);
                structure.UpdateBias(-lr, batch_size);
                if (batch_size == 0)
                    batch_size = 1;
            }
        }
    }
    public List<float> Scale(List<float> input)
    {
        float biggest = 0;
        float smallest = Mathf.Infinity;
        float mean = 0;
        float sum = 0;
        List<float> result = new List<float>();
        for (int n = 0; n < input.Count; n++)
        {
            float x = input[n];
            sum += input[n];
            result.Add(x);
            if (x > biggest)
                biggest = x;
            if (x < smallest)
                smallest = x;
        }
        mean = sum/input.Count;
        float dif = biggest - smallest;
        for (int n = 0; n < input.Count; n++)
        {
            result[n] = (result[n] - smallest) / dif;
        }
        return result;
    }
    public List<float> Normalize(List<float> input)
    {
        float biggest = 0;
        float smallest = Mathf.Infinity;
        float mean = 0;
        float sum = 0;
        List<float> result = new List<float>();
        for (int n = 0; n < input.Count; n++)
        {
            float x = input[n];
            sum += input[n];
            result.Add(x);
            if (x > biggest)
                biggest = x;
            if (x < smallest)
                smallest = x;
        }
        mean = sum / input.Count;
        float dif = biggest - smallest;
        for (int n = 0; n < input.Count; n++)
        {
            result[n] = (result[n] - mean) / dif;
        }
        return result;
    }
    public float output(int o)
    {
        return structure.LastLayer.Node(o).Value;
    }
    public List<float> Output
    {
        get
        {
            List<float> result = new List<float>();
            for (int o = 0; o < structure.LastLayer.NodeCount; o++)
            {
                result.Add(structure.LastLayer.Node(o).Value);
            }
            return result;
        }
    }
    public void SetZeroDelta(int n)
    {
        List<float> ou= new List<float>();
        for(int o = 0; o < structure.LastLayer.NodeCount; o++)
        {
            if (o != n)
                ou.Add(output(o));
            else
                ou.Add(structure.LastLayer.GetTarget(o));
        }
        structure.LastLayer.SetTargets(ou);
        string oc = "";
        for(int i = 0; i < structure.LastLayer.NodeCount; i++)
        {
            oc += structure.LastLayer.GetTarget(i) + " ";
        }
        //print(oc);
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
    public Structure SetSave(string name,Structure structure)
    {
        SetWeights(name,structure);
        SetBiases(name,structure);
        return structure;
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
    public Structure SetWeights(string name,Structure structure)
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
        return structure;
    }
    public Structure SetBiases(string name, Structure structure)
    {
        int line = int.Parse(ReadText("Save " + name, 0)) + 1;
        for (int l = 0; l < structure.LayerCount - 1; l++)
        {
            structure.Layer(l).setBias(float.Parse(ReadText("Save " + name, line)));
            line += 1;
        }
        return structure;
    }
    public void ChangeLittle(bool weights, bool biases)
    {
        if (weights)
            structure.ChangeLittleWeight();
        if (biases)
            structure.ChangeLittleBiases();
    }
}
