using System.Collections.Generic;

public class LA
{
    int layer_count = 0;
    Functions.ActivationDelegate activation;
    public LA(int layer_count, params Functions.ActivationDelegate[] activation)
    {
        this.layer_count = layer_count;
        if (activation.Length != 0)
            this.activation = activation[0];
    }
    public int LayerCount { get { return layer_count; } }
    public Functions.ActivationDelegate Activation { get { return activation; } }
}
