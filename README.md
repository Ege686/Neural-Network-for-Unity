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
  You need to sset some things. Or we can say you need to start the engine. You can do it with start() method.
 It takes 2 argument: Input list and output list as you setted them before the first step.
   
    brain.start(input,output);
 Third Step:
  You need to set up the structure. And you can do it with add_hidden_layer() method.
  add_hidden_layer() method takes 2 argument: node_count(neuron_count), activation_function
  Basically this method creates a hidden layer. Input and output layer will be created aotomatically.
  And there is two activation functions: relu and sigmoid
  
     brain.add_hidden_layer(5,"relu");
     brain.add_hidden_layer(2,"relu");
     //here it creates 2 hidden layers. First has 5 nodes and second has 2 nodes 
     
 Fourth Step:
  You need to compile all the input, hidden and output layers. This can be done with set_structure() method.
  It takes 2 arguments: output list and last layer's activicion function.
 
       brain.set_structure(output, "sigmoid);
   
   So structure is complited.
  Attaching input and output list:
   Before every time you train or make a predict you need to assign the input and target outputs. When you making predict you don't need to assign the exact output values. It just can be all zeros. But output list's structure has to be same as you started.
   So it takes 2 argument: input list and target_output list
   
       brain.attach_values(input, target_output);
       
   
   Predicting Values:
     This can be done with predict() method. 
     It takes 0 argument
        
        brain.attach_values(input, target_output); //don't forget this command
        brain.predict();
     
    
 Also you will need to take output values. You can take it with get_predicted_values() method. It returns a float list that caries the predicted values.
 
         float output = brain.get_predicted_values()[0];
         
         
 Training Part:
   You need to train the neural network. So it can be done with procces() method.
   It takes 1 argument: Epochs(how many times you want to train it)
   And you also need to call attach_values() method to update input and output values. Here you need to attach the exact same values as target_output
   
          brain.attach_values(input, output); //don't forget this command
          brain.procces(6);


  Saving Part
    Yes you also need to save your weights and bias values. So this can be done with save_weights() and save_bias() methods. These methods save values as txt file.
    Each method takes 1 argument: name: name of the txt file.
         
          brain.save_weights("weights1");
          brain.save_biass("bias1");
    
  Setting Part
    So you saved your values but then what? Of course reading them(if you want). This can be done with set_weights() and set_bias() methods. It will read the txt file you had saved for values.
    Each method takes 1 argument: name: name of the saved txt file.
    
          brain.set_weights("weights1");
          brain.set_biass("bias1");
          
    
  Change a little Part
     If you are making something learning by trying and with lots of agents you can use these ChangeALittleWeights() and ChangeALittleBias() methods. It will change agents saved weights and bias values. It will not rewrite the saved values txt file. It is only valid for that agent.
     I will multiply yout weights and bias values something between 0.1 and 0.9 by random. You can change the range by editing the script. Methods are at the bottom of the script.
 
            ChangeALittleWeights();
            ChangeALittleBias();

Little Favor:
 If you are going to make a video about AI learns something and if you will use my code please don't be selfish and give references. Like giving the link of the script. Or saying it in the video. Thanks 
 
Example of AI learning something:
https://youtu.be/pDmKVT0qgjM?t=445

I made that video with my code(my code=this attached script). Video is Turkish but you can still see how it learns.
