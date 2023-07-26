using System.Collections.Generic;

public class Structure 
{
    List<Layer> layers = new List<Layer>();
    List<float> target_outputs= new List<float>();
    public Structure(params int[] layers_count)
    {
        for(int i = 0; i < layers_count.Length; i++)
        {
            Layer l = new Layer(layers_count[i], 0, i + 1 != layers_count.Length ? layers_count[i + 1] : 0);
            this.layers.Add(l);
            if (i != 0)
                this.layers[i - 1].SetConnection(layers[i]);
        }
    }
    public void setActiavations(params Functions.ActivationDelegate[] functions)
    {
        for(int i = 0; i < functions.Length; i++)
        {
            Layer(i).setActivation(functions[i]);
            if (i != 0) Layer(i).setBeforeActivation(Layer(i - 1).activation);
        }
    }
    public int LayerCount { get { return layers.Count; } }
    public int WeightCount { get { int c = 0; for(int l = 0; l < LayerCount - 1; l++) { c += Layer(l).WeightCount; } return c; } }
    public int NodeCount(int layer) { return layers[layer].NodeCount; }
    public Layer Layer(int l) { return layers[l]; }
    public float Target(int node) { return target_outputs[node]; }

    public void SetTargets(List<float> target_outputs) { this.target_outputs = target_outputs; Layer(LayerCount - 1).SetTargets(target_outputs); }
    public void UpdateWeights(float lr) { for(int l = 0; l < LayerCount; l++) { Layer(l).UpdateWeights(lr); } }

    public void Forward() { for(int l=0;l< LayerCount-1;l++) { Layer(l).Forward(); } }

    public void SetDeltaValues() { for(int l = LayerCount - 1; l > 0; l--) { Layer(l).DeltaValues(); } }
    public void SetWeightsD(int batch_size) { for (int l = LayerCount - 2; l >= 0; l--) { Layer(l).WeightsD(batch_size); } }
    public void SetInputs(List<float> inputs) { for (int i = 0; i < Layer(0).NodeCount; i++) { Layer(0).Node(i).SetValue(inputs[i]); } }

    public void SetBiasD(int batch_size) { for(int l=LayerCount-2; l>=0;l--) { Layer(l).BiasD(batch_size); } }
    public void UpdateBias(float lr) { for (int l = LayerCount - 2; l >= 0; l--) { Layer(l).UpdateBias(lr); } }

    public void ChangeLittleWeight() { for (int l = 0; l < LayerCount - 1; l++) Layer(l).ChangeLittleWeights(); }
    public void ChangeLittleBiases() { for (int l = 0; l < LayerCount - 1; l++) Layer(l).ChangeLittleBias(); }
}
