using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    List<Node> nodes = new List<Node>();
    int node_count = 0;
    float bias = 0;
    public Functions.ActivationDelegate activation;
    public Functions.ActivationDelegate before_activation;
    public Functions.LossDelegate loss;
    public Layer connection_layer;
    List<float> target_outputs = null;
    float bias_d = 0;
    float loss_value = 0;
    float dropout_p = 0;
    public Layer(int node_count, float bias, int weight_count,int p_n)
    {
        this.node_count = node_count;
        this.bias = bias;
        for (int i = 0; i < node_count; i++)
        {
            Node n = new Node(0, 0);
            n.SetWeight(weight_count,p_n);
            nodes.Add(n);
            this.bias = Random.Range(0f, 4f);
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
    public void setLoss(Functions.LossDelegate loss)
    {
        this.loss = loss;
    }
    public void setBias(float bias)
    {
        this.bias = bias;
    }
    public float Activation(float x, bool derivative, Functions.LossDelegate loss, params float[] a) { return activation(x, derivative,loss,a); }
    public float BeforeActivation(float x, bool derivative, Functions.LossDelegate loss, params float[] a) { return before_activation(x, derivative,loss,a); }
    public float Loss(float t, float p, int node_count, bool derivative) { return loss(t, p, node_count, derivative); }
    public int NextNodeCount { get { return connection_layer.NodeCount; } }
    public int NodeCount { get { return nodes.Count; } }
    public float Bias { get { return bias; } }
    public int WeightCount { get { int c = 0; for (int n = 0; n < NodeCount; n++) { c += Node(n).WeightCount; } return c; } }
    public Node NextNode(int n) { return connection_layer.Node(n); }
    public Node Node(int n) { return nodes[n]; }
    public float GetTarget(int t) { return target_outputs[t]; }
    public float LossValue { get { for (int o = 0; o < NodeCount; o++) loss_value += Loss(target_outputs[o], Node(o).Value, NodeCount, false); return loss_value; } }

    public void UpdateWeights(float lr, int batch_size) { for (int i = 0; i < nodes.Count; i++) { Node(i).UpdateWeights(lr,batch_size); Node(i).DeltaValue = 0; } }
    public void SetConnection(Layer connection_layer) { this.connection_layer = connection_layer; }
    public void SetTargets(List<float> target_outputs) { this.target_outputs = target_outputs; }
    public void Dropout() { for(int n = 0; n < (int)(dropout_p * NodeCount); n++) { int r_n = Random.Range(0, NodeCount); while (Node(r_n).Droupout) { r_n = Random.Range(0, NodeCount); } Node(r_n).Droupout=true; } }
    public float DropoutP { get { return dropout_p; } set { dropout_p = value; } }

    public void Forward()
    {
        float[] a=new float[NextNodeCount*2+1];
        for (int n = 0; n < NextNodeCount; n++)
        {
            float x = 0;
            for (int nn = 0; nn < NodeCount; nn++) { x += Node(nn).Forward(n,1-dropout_p); }
            x += Bias;
            a[n] = x;
            connection_layer.Node(n).Value=x;
        }
        for (int n = 0; n < NextNodeCount; n++)
        {
            float x = connection_layer.Node(n).Value;
            x = Activation(x, false, null ,a);
            connection_layer.Node(n).Value=x;
        }
        
    }

    public void DeltaValue(int node)
    {
        float x = 0;
        if (connection_layer == null) {
            float[] output = new float[NodeCount*2+1];
            for (int o = 0; o < NodeCount; o++) output[o] = Node(o).Value;
            for (int o = NodeCount; o < NodeCount*2; o++) output[o] = target_outputs[o-NodeCount];
            output[output.Length - 1] = target_outputs[node];
            x +=  BeforeActivation(Node(node).Value, true,this.loss,output);
            //Debug.Log(x+" "+ Node(node).Value+" "+ output[output.Length-1]);
            //Debug.Log(BeforeActivation(Node(node).Value, true, this.loss, output)+" "+ output[output.Length-1]+" "+Node(node).Value);
        }
        else
        {
            for (int n = 0; n < connection_layer.NodeCount; n++) { x += connection_layer.Node(n).DeltaValue * Node(node).Weight(n); }
            x *= BeforeActivation(Node(node).Value, true, null);
        }
        //Debug.Log("oçluk "+BeforeActivation(Node(node).Value, true, null));
        Node(node).DeltaValue=x;
    }
    public void DeltaValues() { for (int n = 0; n < nodes.Count; n++) { DeltaValue(n);} }

    public void WeightD(int node, int batch_size,float alpha) { for (int n = 0; n < connection_layer.NodeCount; n++) { Node(node).SetWeightD(n, (connection_layer.Node(n).DeltaValue * Node(node).Value),alpha);  } Node(node).Droupout=false; }
    public void WeightsD(int batch_size,float alpha) { for (int n = 0; n < NodeCount; n++) { WeightD(n, batch_size,alpha); } }

    public void UpdateBias(float lr, int batch_size) { bias += bias_d * lr/batch_size; bias_d = 0; }
    public void BiasD() { for (int n = 0; n < connection_layer.NodeCount; n++) bias_d += connection_layer.Node(n).DeltaValue; }

    public void ChangeLittleWeights() { for (int n = 0; n < NodeCount; n++) Node(n).ChangeLittleWeight(); }
    public void ChangeLittleBias() { bias *= Random.Range(.8f, 1f); }
}
