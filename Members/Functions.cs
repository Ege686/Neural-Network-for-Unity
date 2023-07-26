using UnityEngine;

public class Functions
{
    public delegate float ActivationDelegate(float x, bool derivative,params float[] a);

    public float Identity(float x, bool derivative, params float[] a) 
    {
        if (!derivative) return x;
        else return 1;
    }
    public float BinaryStep(float x, bool derivative, params float[] a)
    {
        if (derivative) { if (x >= 0) return 1; else return 0; }
        else return 0;
    }
    public float ReLU(float x, bool derivative,params float[] a)
    {
        if (!derivative) { if (x < 0) return 0; else return x; }

        else { if (x > 0) return 1; else return 0; }
    }
    public float LeakyReLU(float x, bool derivative, params float[] a)
    {
        if (!derivative) { if (x >= 0) return x; else return x * 0.01f; }
        else { if (x >= 0) return 1; else return 0.01f; };
    }
    public float PReLU(float x, bool derivative, params float[] a)
    {
        if (a.Length == 0) { Debug.LogError("PReLU function needs a extra parameter. Please enter a value!"); return 0; }
        float A = a[0];
        if (!derivative) { if (x >= 0) return x; else return x * A; }
        else { if (x >= 0) return 1; else return A; };
    }
    public float ELU(float x, bool derivative, params float[] a)
    {
        if (a.Length == 0) { Debug.LogError("ELU function needs a extra parameter. Please enter a value!"); return 0; }
        float A = a[0];
        if (!derivative) { if (x >= 0) return x; else return A * (Mathf.Exp(x) - 1); }
        else { if (x >= 0) return 1; else return A*Mathf.Exp(x); }
    }
    public float Sigmoid(float x, bool derivative, params float[] a)
    {
        float X = 1 / (1 + Mathf.Exp(-x));
        if (!derivative) return X;
        else return X * (1 - X);
    }
    public float Softplus(float x, bool derivative, params float[] a)
    {
        if (!derivative) return Mathf.Log(1 + Mathf.Exp(x));
        else return 1/(1+Mathf.Exp(-x));
    }
    public float TanH(float x, bool derivative, params float[] a)
    {
        float X = (2 / (1 + Mathf.Exp(-2 * x))) - 1;
        if (!derivative) return X;
        else return 1 - (X * X);
    }
    public float SiLU(float x, bool derivative, params float[] a)
    {
        if (!derivative) return x / (1 + Mathf.Exp(-x));
        else return (1 + Mathf.Exp(-x) + x * Mathf.Exp(-x)) / Mathf.Pow(1 + Mathf.Exp(-x), 2);
    }
    public float Gaussian(float x, bool derivative, params float[] a)
    {
        if(!derivative) return Mathf.Exp(-x * x);
        else return -2*x*Mathf.Exp(-x*x);
    }
}
