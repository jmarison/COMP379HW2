<h5>Homework 2</h5>

For this homework you will build your own ML models using the perceptron
and adaline algorithms. Please implement your own version of the perceptron
algorithm and compare it to the textbook implementation (available via the
author’s github site). You can also implement your own version of adaline, but
it is perfectly acceptable to use the author’s (or any other) implementation.
Once you have the implementations ready:

It is a common practice in machine learning to create synthetic data with
well-understood properties to investigate the behavior of an algorithm.
Please create your own dataset (at least 10 examples) that is linearly
separable. Now train a perceptron model. Provide evidence that your
perceptron found a decision boundary. Finally, measure the accuracy of
your model on the training set and comment on the result. [undergrads:
25 points; grads: 20 points]  

Create your own small dataset (at least 10 examples) that is not linearlyseperable. Now train a perceptron model. Did the algorithm converge?
Provide evidence. Now measure the accuracy of your model on the training
set and comment on the result. [undergrads: 25 points; grads: 20 points]


Download the Titanic dataset and randomly split it into training (70%) and 
test (30%) sets. Train an adaline model using the training data. Evaluate
it on (a) training data; (b) test data. Is there a difference in performance?
Please report your performance and explain the difference. You are free to
use either the SGD or the batch version of adaline. [30 points] 

What were the most predictive features of your Titanic model? Provide
evidence. [undergrads: 10 points; grads: 20 points]

A common approach for evaluating a machine learning model is to compare it to a baseline model, which usually uses simple heuristics and/or
randomness. The purpose of this comparison is to ensure that the machine
learning model is behaving in an ‘intelligent’ way (e.g. rather than merely
guessing). Please create a baseline model and compare its performance to
the performance of your perceptron and adaline models. One possibility
here is to simply set the weights to random numbers and have the model
generate predictions using these weights. [10 points] 


Good luck!
