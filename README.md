# imageclassify4
This is using Vgg16 to classify 102 different types of flower with Python.

# predict.py 
# This file is using Vgg16 to predict type of flower. You can using this predict.py with the different option to cater your need.
--image_path: Which image you would like to predict?
--checkpoint: File which stores weights and biases of the trained model. 
--top_k: How many prediction you want this *.py to show.
--gpu: Use the option to turn on the GPU for faster result.

Example:
>>>python predict.py --image_path flowers --checkpoint checkpoint.pth --top_k 2 --gpu=True

# train.py
# This file is using Vgg16 train the model of the neural network. The neural network I used was a pre-trained model and 
# I customised it to suit the requirement of this project.
#--data_directory: Data used to train the model
#--gpu: Use the option to turn on the GPU for faster result.
#--epochs: Number of epochs. Default is 5. 
#--save_directory: Where to save the checkpoint file. Checkpoint file is the file which stores the weight and biases of the trained model.
#--lr: Learning rate. Default is 0.001. A lower number give a better prediction.
