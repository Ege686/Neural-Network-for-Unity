using System.Collections.Generic;

public class LA
{
    int layer_count = 0;
    Functions.ActivationDelegate activation;
    Functions.LossDelegate loss;
    public LA(int layer_count, params Functions.ActivationDelegate[] activation)
    {
        this.layer_count = layer_count;
        if (activation.Length != 0)
            this.activation = activation[0];
    }
    public LA(int layer_count, Functions.LossDelegate loss)
    {
        this.layer_count = layer_count;
        this.loss = loss;
    }
    public int LayerCount { get { return layer_count; } }
    public Functions.ActivationDelegate Activation { get { return activation; } }
    public Functions.LossDelegate Loss { get { return loss; } }
}
