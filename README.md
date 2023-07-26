# Neural-Network-for-Unity
It is a neural network brain(AI) for anything in Unity


If you want to make something think in Unity you can attach the NeuralNetwork.cs script to it. But you need to write some basic commands in another script. Don't worry it is easy to use.

Before first step:
 The input and target output values need to be float list. And they have to be 2 dimension lists.
 
    List<List<float>> input = new List<List<float>>();
    List<List<float>> output = new List<List<float>>();
  Like this. You can think first list -which caries another float list- caries samples. And second list-which is in he first list- caries values of your sample.
 You can add samples as much as you want and you can add values as much as you want too.
  
  Output is same. First list caries input sample's target output values list. And second list caries target output values. First List of output needs to match with
  input's first list. But second list of output can be any number.(Second list of outputs represents the number of predictions)
  
 Example:
 
  Inputs:  
  (1,1)        
  (1,0)      
  (0,1)        
  (0,0)    

  Target Outputs:
  (1)
  (1)
  (1)
  (0)

Also there is a Scale() method to scale the inputs between 0 and 1. It takes 1 dimensional float list and returns also 1 dimensional float list.

     List<float> scaled_input=nn.Scale(input);
  
In this example input and output lists' first lists' counts have to be 4. And each input's second layers' count has to be 2. Finally each target ouput's second layers' count has to be 1.

1st Step:
 You need to find the AI script in the control script.
 All you need is create a variliable and then define the variliable to it:

    public NeuralNetwork nn;
    void Start()
    {
        brain = GetComponent<NeuralNetwork>();  
    }
 2nd Step:
  You need to set up the structure. And you can do it with SetStructure() method. It allows you to create layers. First one is input, till the last hidden and the last one is of course output layer.
  SetStructure() method takes array argument of LA. Don't worry LA is not a complicated thing it just takes "node count" for the layer and the activation function for that layer.
  There are bunch of activation functions: ReLU,ELU,LeakyReLU,Sigmoid,Tanh,Gaussian... You can find them in the Functions.cs script in the Members file.

     nn.SetStructure(new LA(5, nn.functions.ReLU), new LA(3, nn.functions.Sigmoid), new LA(2));
     //here it creates 3 layer(1 input with 5 nodes,1 hidden with 3 nodes, 1 output with 2 nodes). Output layer doesn't need to take a activation function
     
 3rd Step:
  You need to set the inputs and outputs before training and predicting(Before predicting you don't need to set the output). This can be done with SetVariables() method.
  It takes 2 arguments: input list and output list(again output does not necessary before predicting)
 
       nn.SetVariables(inputs);
       nn.Predict();
       //or
       nn.SetVariables(inputs,target_outputs);
       nn.Train(100,10);
       
   
   Predicting Values:
     This can be done with Predict() method. 
     It takes 0 argument
        
        nn.SetVariables(input); //don't forget this command
        nn.Predict();
     
    
 Also you will need to take output values. You can take it with ouput() method. It returns a float value of the outputs' node that you clarified when you calling the method.
 
         float output = nn.output(1);
         
         
 Training Part:
   You need to train the neural network. So it can be done with Train() method.
   It takes 2 argument: Epochs(how many times you want to train it), Batch Size(How many examples you want to feed forward through the structure at once)
   And you also need to call SetVariables() method to update input and output values. Here you need to attach the exact same values as target_output
   (Don't lie, now it works with 1 sized input and output, I am working on how to that, it won't take much. If you are reading here probably I updated the script newly. So in 1 or 2 days it will be done don't worry)
   
          nn.SetVariables(input, target_output); //don't forget this command
          nn.Train(100,10);


  Saving Part
    Yes you also need to save your weights and bias values. So this can be done with SaveAll() method. These method saves values as txt file.
    Method takes 1 argument: name: name of the txt file you want to save(You need to make a Saves folder in the Assets folder by manually to save them in there. I am working on that).
         
          brain.SaveAll("Meahmut");
    
  Setting Part
    So you saved your values but then what? Of course reading them(if you want). This can be done with SetSave() method. It will read the txt file you had saved for values.
    Method takes 1 argument: name: name of the saved txt file.
    
          nn.SetSave("Meahmut");
          
    
  Change a little Part(It will come in soon, I am working on that too. Oh boy you came early, I was not expecting you to come this early.)
     If you are making something learning by trying and with lots of agents you can use these ChangeALittleWeights() and ChangeALittleBias() methods. It will change agents saved weights and bias values. It will not rewrite the saved values txt file. It is only valid for that agent.

Little Favor:
 If you are going to make a video about AI learns something and if you will use my code please don't be selfish and give references. Like giving the link of the script. Or saying it in the video. Thanks 
 
Example of AI learning something:
https://youtu.be/pDmKVT0qgjM?t=445

I made that video with my code(It is a really old version of my AI. I changed lots of things since then. I made a art with code writing. Also I could take off its shit. I so dived into making it clean I could add extra variables to the classes. It would be done with an external script easly but I choose the cleaner version. It may effect on the performance but shouldn't effect too much. Just few extra variables. It is all that.). Video is Turkish but you can still see how it learns.
