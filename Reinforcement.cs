using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Reinforcement : MonoBehaviour
{
    public NeuralNetwork network;
    public NeuralNetwork target;
    List<TimeStep> time_steps = new List<TimeStep>();
    Agent[] agents;
    public Settings settings;

    public string name;
    public bool save = false;
    public bool set = false;

    public int gen;
    public int len;
    float t;
    public int game_speed;
    public int fps;
    int speed;
    void Start()
    {
        Application.targetFrameRate = fps;
        t = Time.fixedDeltaTime;
        agents = FindObjectsOfType<Agent>();
        settings = GetComponent<Settings>();
        LA[] layers = new LA[settings.length];
        for (int i = 0; i < settings.length; i++)
        {
            if (i != settings.length - 1)
                layers[i] = new LA(settings.Node_Count[i], settings.getActivation(i), settings.DropoutP.Count != 0 ? settings.DropoutP[i] : 0);
            else
                layers[i] = new LA(settings.Node_Count[i], settings.getLoss(i));
        }
        SetStructure(layers);
        if (set)
        {
            network.SetSave(name);
            target.SetSave(name);
        }
        Time.timeScale = orman_kebabı;
        speed = game_speed;
    }
    void FixedUpdate()
    {
        if (speed!=game_speed)
        {
            Time.timeScale = game_speed;
            speed = game_speed;
        }
        int count = 0;
        foreach(Agent agent in agents)
        {
            if (agent.ded)
                count += 1;
        }
        if (count == agents.Length)
        {
            time_steps = Shuffle();
            Train(settings.epoch, settings.batch_size, settings.learning_rate, settings.N,settings.alpha,settings.Dropout);
            foreach (Agent agent in agents)
            {
                agent.ded = false;
            }
            gen += 1;
        }
        if (save)
        {
            network.SaveN(name);
            network.SaveAll(name);
            save = false;
        }
        len = time_steps.Count;
    }
    void Simulate()
    {
        Physics.autoSimulation = false;
        for(int i = 0; i < 1000; i++)
        {
            Physics.Simulate(Time.fixedDeltaTime);
        }
        Physics.autoSimulation = true;
    }
    public void SetStructure(params LA[] la)
    {
        network.SetStructure(la);
        target.SetStructure(la);
        Eshitle();
    }
    public void AddStep(TimeStep step)
    {
        time_steps.Add(step);
    }
    List<TimeStep> Shuffle()
    {
        List<float> rewards = new List<float>();
        float min = Mathf.Infinity;
        float max = 0;
        for(int r = 0; r < time_steps.Count; r++)
        {
            rewards.Add(time_steps[r].Reward);
        }
        rewards = Normalize(rewards);

        for (int r = 0; r < time_steps.Count; r++)
        {
            time_steps[r].reward = rewards[r];
        }
        int count=time_steps.Count;
        List<TimeStep> steps=new List<TimeStep>();
        for(int i = 0; i < count; i++)
        {
            int r = Random.Range(0, time_steps.Count);
            steps.Add(time_steps[r]);
            time_steps.RemoveAt(r);
        }
        return steps;
    }
    public int SetVariables(int s,int e,int N)
    {
        List<List<float>> inputs = new List<List<float>>();
        List<List<float>> outputs = new List<List<float>>();
        int max = 0;
        for (int t = s; t < e; t++)
        {
            max = time_steps[t].Action;
            inputs.Add(new List<float>());

            for (int i = 0; i < time_steps[t].state.Count; i++)
                inputs[t < N ? t : t - s].Add(time_steps[t].state[i]);
            target.SetVariables(time_steps[t].next_state);
            target.Predict();
            outputs.Add(new List<float>());
            float max_q = 0;
            for (int o = 0; o < target.structure.LastLayer.NodeCount; o++)
            {
                if (target.output(o) > max_q)
                    max_q = target.output(o);
            }
            for (int o = 0; o < target.structure.LastLayer.NodeCount; o++)
            {
                outputs[t < N ? t : t - s].Add((o==max)?time_steps[t].Reward + settings.discount_rate * max_q : target.output(o));
            }
            string a = "o ";
            for (int o = 0; o < target.structure.LastLayer.NodeCount; o++)
                a += target.structure.LastLayer.Node(o).Value + " ";
        }
        network.SetVariables(inputs, outputs);
        return max;
    }
    public void Train(int epoch, int batch_size, float lr,int N,float alpha,bool dropout)
    {
        for (int t = 0; t < time_steps.Count / N + 1; t++)
        {
            int e = t * N + N;
            int s = t * N;
            if (e > time_steps.Count)
                e = time_steps.Count;
            if (e - s < batch_size)
                batch_size = e - s;
            int max = SetVariables(s,e,N);
            network.RLTrain(epoch, batch_size, lr,max,alpha,dropout);
            Eshitle();
        }
        time_steps.Clear();
    }
    public void Predict(List<float> input)
    {
        network.SetVariables(input);
        network.Predict();
    }
    public List<float> Output
    {
        get
        {
            return network.Output;
        }
    }
    public List<float> Scale(List<float> input)
    {
        return network.Scale(input);
    }
    public List<float> Normalize(List<float> input)
    {
        return network.Normalize(input);
    }
    void Eshitle()
    {
        for(int l=0;l< network.structure.LayerCount-1; l++)
        {
            for(int n = 0; n < network.structure.Layer(l).NodeCount; n++)
            {
                for(int w=0;w< network.structure.Layer(l).Node(n).WeightCount; w++)
                {
                    float a = network.structure.Layer(l).Node(n).Weight(w);
                    target.structure.Layer(l).Node(n).SetWeight(w, a);
                }
            }
            float b = network.structure.Layer(l).Bias;
            target.structure.Layer(l).setBias(b);
        }
    }
    void Eshitle(Structure network)
    {
        for (int l = 0; l < network.LayerCount - 1; l++)
        {
            for (int n = 0; n < network.Layer(l).NodeCount; n++)
            {
                for (int w = 0; w < network.Layer(l).Node(n).WeightCount; w++)
                {
                    network.Layer(l).Node(n).SetWeight(w, network.Layer(l).Node(n).Weight(w));
                }
            }
            network.Layer(l).setBias(network.Layer(l).Bias);
        }
    }
}
