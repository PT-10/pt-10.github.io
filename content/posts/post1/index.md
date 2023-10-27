+++
title = 'Intuition behind Optimization'
date = 2023-10-27T10:59:30+05:30

+++

An attempt to explore the intuition of optimization -- how minimizing an objective results in optimal parameters.

### The Idea
Remember the last time you picked up only a few important things to read, a day before an exam. What you did there was optimize your available time (not really, but for the day) to maximize your output (marks).

In machine learning, the idea of optimization revolves around improving metrics like F1 score, accuracy of models etc. This further translates to minimizing the loss.

Problems in machine learning usually fit some sort of mathematical hypothesis - A relation that maps inputs X to outputs Y. This relation could be anything -- linear, polynomial and so on. Once you've picked a hypothesis, you have to determine the best it can do. 

How do you determine the best? Rather, what does the best exactly mean?

Let's take the given sample of points
![alt text](images/graph.png#center)
$$ f(x) = mx+c $$
$$ \theta \text{ denotes the set of parameters } \{m, c\} $$
Let's say your hypothesis is linear. To figure out the best we can do, we need to find the line that fits more datapoints than any other line. 
The line depends on the parameters m, c. Thus, our goal is to find those values.

### Intuition behind errors, loss, cost, objective functions
**Error** is a statistical object given by
$$\text{Error}(\hat{y_i}, y_i) = \hat{y_i} - y_i \text{ where } \hat{y_i} = f(x_i)$$

**Loss function** tells how bad are the consequences of a certain error i.e, quantifies the error. The type of loss function used differs based on the task at hand. Given below is the example of one such loss function called **L2 loss** or **Squared Error**.
$$ L(\theta;x_i,y_i) = (\theta^T x_i - y_i)^2 $$

A **cost function** is simply the average of loss function over the entire dataset.
$$ \text{MSE}(\theta; X, y) = \frac{1}{m} \sum_{i=1}^{m} (\theta^T x_i - y_i)^2 $$

An **objective function** is any function that we seek to either maximize or minimize during training. A loss/cost function is essentially an objective function that is being minimized.

It is clear that the cost depends on the value of the parameters. As the parameters change, $\hat{y_i}$ changes, changing the cost function as well.

Thus, as we minimize our objective function, we obtain the optimal values of our parameters, which give us the best possible line, given our linear hypothesis.