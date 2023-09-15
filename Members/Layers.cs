using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Settings))]
public class Layers:Editor
{
    int length = 0;
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        Settings set=(Settings)target;
        if(set.length<0) set.length=0;
        int node = set.Node_Count.Count;
        int activ=set.activations.Count;
        int drop=set.DropoutP.Count;
        if (activ != set.length)
        {
            for(int i = 0;set.length>activ && i < set.length -activ; i++)
            {
                set.activations.Add(new Settings.Activation());
            }
            for (int i = 0; set.length < activ && i < activ-set.length; i++)
            {
                set.activations.RemoveAt(set.activations.Count-1);
            }
        }
        if (node != set.length)
        {
            for (int i = 0; set.length > node && i < set.length - node; i++)
            {
                set.Node_Count.Add(new int());
            }
            for (int i = 0; set.length < node && i < node - set.length; i++)
            {
                set.Node_Count.RemoveAt(set.Node_Count.Count - 1);
            }
        }
        if (set.Dropout&&drop != set.length)
        {
            for (int i = 0; set.length > drop && i < set.length - drop-1; i++)
            {
                set.DropoutP.Add(new float());
            }
            for (int i = 0; set.length < drop && i < drop - Mathf.Max(set.length-1,0); i++)
            {
                set.DropoutP.RemoveAt(set.DropoutP.Count - 1);
            }
        }
        if (!set.Dropout)
        {
            set.DropoutP.Clear();
        }
    }
}
