using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeStep
{
    public List<float> state=new List<float>();
    public int action;
    public float reward=0;
    public List<float> next_state=new List<float>();
    public TimeStep(List<float> state,int action, float reward)
    {
        this.state = eshitle(this.state,state);
        this.action = action;
        this.reward = reward;
    }
    List<float> eshitle(List<float> a, List<float> b)
    {
        for (int i = 0; i < b.Count; i++)
        {
            a.Add(b[i]);
        }
        return a;
    }
    public void SetNext(List<float> next_state) { this.next_state = eshitle(this.next_state,next_state); }
    public int Action { get { return action; } }
    public float Reward { get { return reward; } set { reward = value; } }
}
