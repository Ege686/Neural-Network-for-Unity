# Neural-Network-for-Unity
It is a neural network brain(AI) for anything in Unity


If you want to something think in Unity you can attach this script to it. But you need to write some basic commands in another script. Don't worry it is easy to use.


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
