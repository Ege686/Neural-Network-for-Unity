# Neural-Network-for-Unity
It is a neural network brain(AI) for anything in Unity


If you want to something think in Unity you can attach this script to it. But you need to write some basic commands in another script. Don't worry it is easy to use.

Before first step:
 The input and target output values need to be float list. And they have to be 2 dimension lists.
 
    List<List<float>> input = new List<List<float>>();
    List<List<float>> output = new List<List<float>>();
  Like this. You can think first list -which caries another float list- caries samples. And second list-which is in he first list- caries values of your sample.
 You can add samples as much as you want and you can add values as much as you want too.
  
  Output is same. First list caries input sample's target output values list. And second list caries target output values. First List of output needs to match with
  input's first list. But second list of output can be any number.(Second list of outputs represents the number of predictions)
  
 Example:
  Inputs:     Target Outputs:
  1,1              1
  1,0              1
  0,1              1
  0,0              0

In this example input and output lists' first lists' counts have to be 4. And each input's second layers' count has to be 2. Finally each target ouput's second layers' count has to be 1.

First Step:
 You need to find the AI script in the control script.
 All you need is create a variliable and then define the variliable to it:

    public neural_network_AI brain;
    void Start()
    {
        brain = GetComponent<neural_network_AI>();  
    }
 
 Second Step:
  You need to set up the structure. And you can do it with Dense() method.
  Dense() method takes 3 argument: node_count(neuron_count), input_values, activation_function
  Basically this method creates a hidden layer. Input and output layer will be created aotomatically.
  And there is two activation functions: relu and sigmoid
  
     brain.Dense(5, input,"relu");
     brain.Dense(2, input,"relu");
     //here it creates 2 hidden layers. First has 5 nodes and second has 2 nodes 
     
 Third Step:
  You need to attach the input and target output values. And this can be done with attach_values2() method.
  If you calling this method for the first time you dont need to attach the exact target output values which you want. But it needs to be the same achitecture as your target output values list. If you want to predict two values it has to be 2 values in the target_output list.
  And you also need to call this method before every learning and predicting method. Else you won't get the result you want.
  It takes 2 argument: Input values list and target output values list
  
       brain.attach_values2(input, output);
  Fourth Step:
   It is like third step. Now network has to combine the input,outpu and hidden layer to create the full structure. This can be done with attach_values() method
   It takes 1 argument and it is last layer's activation function.(There is 2:relu and sigmoid)
   
       brain.attach_values("sigmoid");
       
   So structure is complited.
   
   Predicting Values:
     This can be done with predict() method. 
     It takes 1 argument: Input values list
     But first as I mention you need to call attach_values2 method to update the inputs and outputs. You also dont need to attach the exact same values as target output but again it has to be same architecture as target outputs.
     
        brain.attach_values2(input, output);
        brain.predict(input);
     
    
 Also you will need to take output values. You can take it with value_of_predict() method. It returns a float list that caries the predicted values.
 
         float output = brain.value_of_predict()[0];
         
         
 Training Part:
   You need to train the neural network. So it can be done with procces() method.
   It takes 2 argument: Epochs(how many times you want to train it), input values list
   And you also need to call attach_values2() method to update input and output values. Here you need to attach the exact same values as target_output
   
          brain.attach_values2(input, output);
          brain.procces(6,input);



Little Favor:
 If you goint to make a video about AI learns something and if you will use my code please don't be selfish and give references. Like giving the link of the script. Or saying it in the video. Thanks 
 
Example of AI learning something:
https://youtu.be/pDmKVT0qgjM?t=445

I made that video with my code(my code=this attached script). Video is Turkish but you can still see how it learns.
