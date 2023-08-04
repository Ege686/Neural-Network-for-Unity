using System.Collections.Generic;

public class LA
{
    int layer_count = 0;
    Functions.ActivationDelegate activation;
    Functions.LossDelegate loss;
    float dropout_p = 0;
    public LA(int layer_count, Functions.ActivationDelegate activation)
    {
        this.layer_count = layer_count;
        this.activation = activation;
    }
    public LA(int layer_count, Functions.ActivationDelegate activation, float dropout_p)
    {
        this.layer_count = layer_count;
        this.activation = activation;
        this.dropout_p = dropout_p;
    }
    public LA(int layer_count, Functions.LossDelegate loss)
    {
        this.layer_count = layer_count;
        this.loss = loss;
    }
    public int LayerCount { get { return layer_count; } }
    public float DropoutP { get { return dropout_p; } }
    public Functions.ActivationDelegate Activation { get { return activation; } }
    public Functions.LossDelegate Loss { get { return loss; } }
}
