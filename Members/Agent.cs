using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Agent : MonoBehaviour
{
    public Reinforcement big_bro;
    List<TimeStep> steps=new List<TimeStep>();
    public bool ded;
    void Start()
    {
        big_bro = FindObjectOfType<Reinforcement>(); 
    }
    public void AddStep(List<float> state, int action,float reward,List<float> next_state)
    {
        TimeStep step = new TimeStep(state, action,reward);
        step.SetNext(next_state);
        steps.Add(step);
        //big_bro.AddStep(step);
    }
    void CalculateCumulativeReward(float discount_rate)
    {
        for(int r = 0; r < steps.Count; r++)
        {
            for(int n_r=r+1; n_r < steps.Count; n_r++)
            {
                steps[r].Reward = 0;
                steps[r].Reward += Mathf.Pow(discount_rate, n_r-r-1) * steps[n_r].Reward;
            }
        }
    }
    public void AddSteps(float discount_rate)
    {
        CalculateCumulativeReward(discount_rate);
        steps.RemoveAt(steps.Count - 1);
        foreach(TimeStep step in steps) { big_bro.AddStep(step); }
        steps.Clear();
    }
}
