using System.Collections.Generic;
using UnityEngine;

public class Structure
{
    List<Layer> layers = new List<Layer>();
    List<float> target_outputs = new List<float>();
    float loss_value = 0;
    public Structure(params int[] layers_count)
    {
        for (int i = 0; i < layers_count.Length; i++)
        {
            Layer l = new Layer(layers_count[i], 0, i + 1 != layers_count.Length ? layers_count[i + 1] : 0, (i != 0)? layers[i - 1].NodeCount:3) ;
            this.layers.Add(l);
            if (i != 0)
                this.layers[i - 1].SetConnection(layers[i]);
        }
    }
    public void setActiavations(params Functions.ActivationDelegate[] functions)
    {
        for (int i = 0; i < functions.Length; i++)
        {
            Layer(i).setActivation(functions[i]);
            if (i != 0) Layer(i).setBeforeActivation(Layer(i - 1).activation);
        }
    }
    public void setLoss(Functions.LossDelegate loss) { Layer(LayerCount - 1).setLoss(loss); }
    public int LayerCount { get { return layers.Count; } }
    public float Loss { get { return loss_value; } }
    public int WeightCount { get { int c = 0; for (int l = 0; l < LayerCount - 1; l++) { c += Layer(l).WeightCount; } return c; } }
    public int NodeCount(int layer) { return layers[layer].NodeCount; }
    public Layer Layer(int l) { return layers[l]; }
    public Layer LastLayer { get { return layers[LayerCount-1]; } }
    public float Target(int node) { return target_outputs[node]; }

    public void SetTargets(List<float> target_outputs) { this.target_outputs = target_outputs; LastLayer.SetTargets(target_outputs); }
    public void UpdateWeights(float lr, int batch_size) { for (int l = 0; l < LayerCount; l++) { Layer(l).UpdateWeights(lr,batch_size); } }
    public void Dropout() { for(int l = 0; l < LayerCount-1; l++) { Layer(l).Dropout(); } }
    public void DropoutP(float[] p) { for(int l = 0; l < LayerCount-1; l++) { Layer(l).DropoutP=p[l]; } }

    public void Forward() { for (int l = 0; l < LayerCount - 1; l++) { Layer(l).Forward(); } }

    public void SetDeltaValues() { for (int l = LayerCount - 1; l > 0; l--) { Layer(l).DeltaValues(); } }
    public void SetDeltaValues(int max) { LastLayer.DeltaValue(max); for (int l = LayerCount - 2; l > 0; l--) { Layer(l).DeltaValues(); } }
    public void SetWeightsD(int batch_size,float alpha) { for (int l = LayerCount - 2; l >= 0; l--) { Layer(l).WeightsD(batch_size,alpha); } }
    public void SetInputs(List<float> inputs) { for (int i = 0; i < Layer(0).NodeCount; i++) { Layer(0).Node(i).Value=inputs[i]; } }

    public void SetBiasD() { for (int l = LayerCount - 2; l >= 0; l--) { Layer(l).BiasD(); } }
    public void UpdateBias(float lr, int batch_size) { for (int l = LayerCount - 2; l >= 0; l--) { Layer(l).UpdateBias(lr,batch_size); } }

    public void ChangeLittleWeight() { for (int l = 0; l < LayerCount - 1; l++) Layer(l).ChangeLittleWeights(); }
    public void ChangeLittleBiases() { for (int l = 0; l < LayerCount - 1; l++) Layer(l).ChangeLittleBias(); }

    public void SetLoss() { loss_value = Layer(LayerCount - 1).LossValue; }
}
