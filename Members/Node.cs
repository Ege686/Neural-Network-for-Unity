using System.Collections.Generic;
using UnityEngine;

public class Node
{
    float value = 0;
    float delta_value = 0;
    List<float> weights=new List<float>();
    List<float> weights_d=new List<float>();
    public Node(float value, float delta_value)
    {
        this.value = value;
        this.delta_value = delta_value;
    }
    public float Value { get { return value; } }
    public float DeltaValue { get { return delta_value; } }
    public int WeightCount { get { return weights.Count; } }
    public float Weight(int w) { return weights[w]; }

    public float Forward(int n) { return Value*Weight(n); }

    public void SetValue(float value) { this.value = value; }
    public void SetDeltaValue(float delta_value) { this.delta_value = delta_value; }

    public void SetWeight(int weight) { for (int w = 0; w < weight; w++) { weights.Add(Random.Range(0f, .99f)); weights_d.Add(0); } }
    public void SetWeight(int weight,float value) { weights[weight] = value; }
    public void SetWeightD(int w,float d) { weights_d[w] += d; }
    public void UpdateWeights(float lr) { for(int w = 0; w < weights.Count; w++) { weights[w] += weights_d[w]*lr; weights_d[w] = 0; } }
}
