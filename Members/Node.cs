using System.Collections.Generic;
using UnityEngine;

public class Node
{
    float value = 0;
    float delta_value = 0;
    List<float> weights = new List<float>();
    public List<float> weights_d = new List<float>();
    bool dropout = false;
    public Node(float value, float delta_value)
    {
        this.value = value;
        this.delta_value = delta_value;
    }
    public float Value { get { return value; } set { this.value = value; } }
    public float DeltaValue { get { return delta_value; } set { delta_value = value; } }
    public int WeightCount { get { return weights.Count; } }
    public float Weight(int w) { return weights[w]; }

    public float Forward(int n,float p) { if (dropout) value=0; return Value * Weight(n)*p; }


    public void SetWeight(int weight,int n) { for (int w = 0; w < weight; w++) { weights.Add(Random.Range(-Mathf.Sqrt(2/n), Mathf.Sqrt(2 / n))); weights_d.Add(0); } }
    public void SetWeight(int weight, float value) { weights[weight] = value; }
    public void SetWeightD(int w, float d,float alpha) { weights_d[w] += d+(2 * alpha * Weight(w)); }
    public void UpdateWeights(float lr, int batch_size) { for (int w = 0; w < weights.Count; w++) { weights[w] += weights_d[w] * lr / batch_size; weights_d[w] = 0; } }
    public void ChangeLittleWeight() { for (int n = 0; n < weights.Count; n++) { weights[n] *= Random.Range(.8f, 1f); } }

    public bool Droupout { get { return dropout; } set { dropout = value; } }
}
