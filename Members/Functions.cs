using UnityEngine;
using System.Collections;
public class Functions
{
    public delegate float ActivationDelegate(float x, bool derivative, LossDelegate loss, params float[] a);
    public delegate float LossDelegate(float t, float p, int node_count, bool derivative);

    public float Identity(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        float b = 1f;
        if (!derivative) return x*b;
        else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true)*b : b;
    }
    public float BinaryStep(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (derivative) { if (x >= 0) return 1; else return 0; }
        else return 0;
    }
    public float ReLU(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (!derivative) { if (x < 0) return 0; else return x; }

        else { if (x > 0) return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1; else return 0; }
    }
    public float LeakyReLU(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (!derivative) { if (x >= 0) return x; else return x * 0.01f; }
        else { if (x > 0) return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1 * 1; else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1 * 0.01f; };
    }
    public float PReLU(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (a.Length == 0) { Debug.LogError("PReLU function needs a extra parameter. Please enter a value!"); return 0; }
        float A = a[0];
        if (!derivative) { if (x >= 0) return x; else return x * A; }
        else { if (x > 0) return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1 * 1; else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1 * A; };
    }
    public float ELU(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (a.Length == 0) { Debug.LogError("ELU function needs a extra parameter. Please enter a value!"); return 0; }
        float A = a[0];
        if (!derivative) { if (x >= 0) return x; else return A * (Mathf.Exp(x) - 1); }
        else { float X =Mathf.Log(x/A); if (x >= 0) return 1; else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) * 1 * A * Mathf.Exp(X) : 1 * A * Mathf.Exp(X); }
    }
    public float Sigmoid(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        float X = 1 / (1 + Mathf.Exp(-x));
        if (!derivative) return X;
        else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true)* x * (1 - x) : x * (1 - x);
    }
    public float Softplus(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (!derivative) return Mathf.Log(1 + Mathf.Exp(x));
        else { float X =Mathf.Log(Mathf.Exp(x)-1); return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) * (1 / (1 + Mathf.Exp(-X))) :  1 / (1 + Mathf.Exp(-X)); }
    }
    public float TanH(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        float X = (2 / (1 + Mathf.Exp(-2 * x))) - 1;
        if (!derivative) return X;
        else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) * (1 - (x * x)) : (1 - (x * x));
    }
    public float SiLU(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (!derivative) return x / (1 + Mathf.Exp(-x));
        else return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true) : 1 * (1 + Mathf.Exp(-x) + x * Mathf.Exp(-x)) / Mathf.Pow(1 + Mathf.Exp(-x), 2);
    }
    public float Gaussian(float x, bool derivative, LossDelegate loss, params float[] a)
    {
        if (!derivative) return Mathf.Exp(-x * x);
        else { float X =Mathf.Sqrt(-Mathf.Log(x)); return (loss != null) ? loss(a[a.Length - 1], x, (a.Length - 1) / 2, true)* -2 * X * Mathf.Exp(-X * X) : -2 * X * Mathf.Exp(-X * X); }
    }
    public float Softmax(float x, bool derivative, LossDelegate loss,params float[] a)
    {
        int l = (a.Length-1) / 2;
        if (!derivative)
        {
            float sum = 0;
            for(int j = 0; j < l; j++)
            {
                sum += Mathf.Exp(a[j]);
            }
            return (Mathf.Exp(x) / sum);
        }
        else
        {
            float sum = 0;
            for (int j = 0; j < l; j++)
            {
                int i = 0;
                if (a[j] == x)
                    i = 1;
                sum += x * (i - a[j]) * loss(a[l + j], a[j], l, true);
            }
            return sum;
        }
    }
    //Mean Squared Error
    public float MSE(float t, float p, int node_count, bool derivative)
    {
        return 2f /node_count* (p - t);
    }
    //Categorical Cross Entropy
    public float CCE(float t, float p, int node_count, bool derivative)
    {
        float sum = 0;
        if (!derivative) {  sum += -t * Mathf.Log(p);return sum; }
        else
        {
            float epsilon = 1e-3f;
            if (p == 0) p += epsilon;

            return -t / p;
        }
    }
    //Binary Cross Entropy
    public float BCE(float t, float p, int node_count, bool derivative)
    {
        float epsilon = 1e-3f;
        if (p == 0||p==1) p += epsilon;
        float x = -((t / p) - ((1 - t) / (1 - p)));
        //Debug.Log(t + " " + p + " " + x);
        return x;
    }
    //Mean Absolute Percentage Error
    public float MAPE(float t, float p, int node_count, bool derivative)
    {
        return 100 / node_count * (1 / t);
    }
}
