using System.Collections.Generic;
using UnityEngine;

public class Layer 
{
    List<Node> nodes=new List<Node>();
    int node_count = 0;
    float bias = 0;
    public Functions.ActivationDelegate activation;
    public Functions.ActivationDelegate before_activation;
    Layer connection_layer;
    List<float> target_outputs = null;
    float bias_d = 0;
    public Layer(int node_count, float bias ,int weight_count)
    {
        this.node_count = node_count;
        this.bias = bias;
        for(int i = 0; i < node_count; i++)
        {
            Node n = new Node(0, 0);
            n.SetWeight(weight_count);
            nodes.Add(n);
        }
    }
    public void setActivation(Functions.ActivationDelegate activation)
    {
        this.activation = activation;
    }
    public void setBeforeActivation(Functions.ActivationDelegate before_activation)
    {
        this.before_activation = before_activation;
    }
    public void setBias(float bias)
    {
        this.bias = bias;
    }
    public float Activation(float x, bool derivative) { return activation(x,derivative); }
    public float BeforeActivation(float x, bool derivative) { return before_activation(x,derivative); }
    public int NextNodeCount { get { return connection_layer.NodeCount; } }
    public int NodeCount { get { return nodes.Count; } }
    public float Bias { get { return bias; } }
    public int WeightCount { get { int c = 0; for(int n = 0; n < NodeCount; n++) { c+= Node(n).WeightCount; }return c; } }
    public Node NextNode(int n) { return connection_layer.Node(n); }
    public Node Node(int n) { return nodes[n]; }
    public float GetTarget(int t) { return target_outputs[t]; }

    public void UpdateWeights(float lr) { for(int i = 0; i < nodes.Count; i++) { Node(i).UpdateWeights(lr);} }
    public void SetConnection(Layer connection_layer) { this.connection_layer = connection_layer; }
    public void SetTargets(List<float> target_outputs) { this.target_outputs = target_outputs; }

    public void Forward() { 
        float x = 0; 
        for(int n = 0; n < NextNodeCount;n++) { 
            for(int nn=0; nn < NodeCount; nn++) { x += Node(nn).Forward(n); }
            x += Bias;
            x = Activation(x, false);
            connection_layer.Node(n).SetValue(x);
        } 
    }

    public void DeltaValue(int node) {
        float x = 0;
        if(connection_layer==null)
            x = (2 / NodeCount) * (GetTarget(node) - Node(node).Value) * BeforeActivation(Node(node).Value, true);
        for (int n = 0; connection_layer!=null && n < connection_layer.NodeCount; n++) { x += connection_layer.Node(n).DeltaValue * Node(node).Weight(n); }
        x *= BeforeActivation(nodes[node].Value,true);
        Node(node).SetDeltaValue(x);
    }
    public void DeltaValues() { for(int n = 0; n < nodes.Count; n++) { DeltaValue(n); } }

    public void WeightD(int node,int batch_size) { for(int n = 0; n < connection_layer.NodeCount; n++) { Node(node).SetWeightD(n, connection_layer.Node(n).DeltaValue * Node(node).Value/batch_size); } }
    public void WeightsD(int batch_size) { for(int n = 0; n < NodeCount; n++) { WeightD(n,batch_size); } }

    public void UpdateBias(float lr) { bias += bias_d*lr; bias_d = 0; }
    public void BiasD(int batch_size) { for (int n = 0; n < connection_layer.NodeCount; n++) bias_d += connection_layer.Node(n).DeltaValue/batch_size; }

    public void ChangeLittleWeights() { for (int n = 0; n < NodeCount; n++) Node(n).ChangeLittleWeight(); }
    public void ChangeLittleBias() { bias*= Random.Range(.8f, .1f); }
}
