# Udacity-Secure-and-Private-AI-Challenge-Scholar
Udacity:Secure and Private AI Scholarship Challenge Nanodegree Program

Welcome to the Udacity-Secure-and-Private-AI-Challenge-Scholar wiki!

### Projects :
* [The OpenMined Project for Secure and Private Ai](https://www.openmined.org)

 

#### [Deep Learning Frameworks](https://youtu.be/SJldOOs4vB8)
- Tensorflow powered by Google ,Tensorboard , Tensorflow lite
- PyTorch powered by Facebook, easy to debug, Modular, TensorBoardX
- Caffe2+Pytorch is similar to TensorflowLite

### What are Keras : [Explanation])(https://youtu.be/j_pJmXJwMLA) 
- Keras is an Interface that wraps up high level frameworks like TensorFlow, Theano , CNTK 
- How it Works , You Define it -> Compile it -> Fit Network -> Evaluate Network -> Make Prediction
-Easiest way to get started followed by PyTorch


### Other Frameworks
- CNTK : Microsoft (Cognitive tool kit),No Convential Open Source License  
- MXNet: Mixed Programming paradigm (Imperative +Declarative)
- Chainer : Supported by IBM , Intel 
- DL4J : Machine Learning for JAVA World
- ONIX : Open Neural Network Exchange Format (Facebook + Microsoft) for Interoperability.
- For Beginners use Keras
- For Production on GCP- TensorFlow
- For Research -Pytorch , SONNET
- For Azure -MXNet
- FOr Java -DL4J
- FOr Ios -CoreML


##### About [Facebook PyTorch Framework](https://pytorch.org/)
- Pytorch was made from Deep Learning Framework Torch which used the LUA Programming Language
- LUA has a high entry barrier, dosent have Modularity
- Its Impletmentation in Pythone made it call as PyTorch

##### Feature One (Imperative Programming)
- Difference between Imperative and Declarative Programming
- Imperative Programming : Performs computation as you type it (much of Python),more effecient due to safe value reusability 
- Symbolic Programming : Generate Symbolic Graph and Computation perform at the end of function (C++)
- Tensorflow uses Symbolic Programming

##### Feature Two (Dynamic Computation Graphing)
- PyTOrch is defined by Run, System Generate the Graph at the Run Time,easy to debug , input is dynamic , Best Suited for Researchers
- Dynamic Graphs are build at runtime which lets use us standard Python Statement
- TensorFlow, System Generate the whole graph before running it ,used for distributed compution, and Production and is best suited for Beginners

* [Tensors : the Data Structure of PyTorch](https://youtu.be/tIqoI4jGE-w) 
- Rank 0  : Type  Scaler  : ex [1]
- Rank 1  : Type  Vectors : ex [1,1]
- Rank 2  : Type  Matrix  : ex [[1,1],[1,1]]
- Rank 3  : Type 3 tensor : ex [[[1,1],[1,1]],[[1,1],[1,1]]]
- Rank n  : N Tensors 


* NumPy : [NumPy Explained](https://youtu.be/Tkv45wgxlEU) Its a Wrapper around C , Fortran Class , array and Matrices implementation similar to MATLAB

* AutoGrad : --- 
* Gradients for Training Neural Network :  [Video](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3&t=0s)
* Validation : 
* Transfer Learning :
* Activation Function : ReLU or Sigmoid activation function or TanH , Why do we need them [Video](https://youtu.be/-7scQpJT7uo)
* BackPropogation : [Back Propogation in 5 Minutes](https://www.youtube.com/watch?v=q555kfIFUCM)











#### Sequence 

- First Watch the Video 1 of the 3Blue1Brown on Machine Learning 
- Then open the Notebook 1 and read until you encounter the word gradient and weights
- Now watch the remaining video 2 and 3 from 3Blue1Brown S2 video on Youtube  
- Complete Part 1 notebook 1 With Sample Code as Code 1 and Code 2 in Code Folder
- Start the Notebook 2 and running the Code Snippets Side by Side on the Google Colab
- As soon as you Encounter the SoftMax Function Look up the Term and Try to wrap you Head around the Softmax funtion of what its trying to do by reading the Excersise of the Softmax function in Part 2
- 







### Resources :  

Online Free Courses
* [Deep Learning and Other Courses by Stanford](https://lagunita.stanford.edu/)  
* [Introduction to Deep Learning at MIT](http://introtodeeplearning.com/)



### Setting up a Notebook (Cloud)
Don't want to Heat up your Machine here are the Solution: 
* [Source Code Github Repo](https://github.com/udacity/deep-learning-v2-pytorch)
* [Using Azure Notebook](https://blogs.msdn.microsoft.com/uk_faculty_connection/2019/02/12/using-pytorch-with-azure/)
* [Using Google Colab](https://colab.research.google.com)