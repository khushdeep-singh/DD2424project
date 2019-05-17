# Project Log Team Jönsson

### 2 May 2019

Had a group-call at 15:00 decided the following:

### 12 May 2019 

it is unlikely that you will find where the model starts learning when doing the LR Range-Test. When increasing the lr linearly from a very low number. less than 1/10 of the plot will be in the lower range where the min_lr for model learning will be happening. 

I suggest we use a logscale as a last experiment to try to capture where the model starts learning and use that as a minium learning rate when doing Cyclic Learning and see what that gives us. 

At the moment we are supposed to use the maximum learning-rate divided by 3 or 4 as the minimum value. 

Doing experiments on batchsize of 256 without momentum (=0) or weight_decay (=0): 
Doing this on resnet56 doesn't seem to make the network diverge even if the upper limit for learning rate is set to 20. 

### 13 May 2019

Still no clear insight in what the largest maximum learning rate should be choosen from the experiments. In the papers no maximum learningrate is over 1. In the super convergence paper no maximum learningrate is over 4. 

The takeaway I guess is that there are two outcomes for learningrate range test. 
(i)  the accuracy plot takes the shape of a hill – not an indicator of SuperConvergence
(ii) the accuracy plot rises and slowly tampers of to a horizontal accuracy – indicator of SuperConvergence. 

(iii) we found a third alternative. For really high learningrates the type (ii) plot of the accuracy would collapse to around 10% accuracy and then form a new plot of the type (ii) from around learningrate 12 to 30... See figure rangetest/ola/11acc_bs1024_minlr1e-09_maxlr15.0_iterations4000_momentum0_weightdecy0_acc_val_netresnet56.png

I believe a maximum learningrate of 1 is needed when doing cyclical learning and when trying to do superconvergence the maximum learningrate is even higher. 

What bothers me is that we cannot find a rule for choosing a proper value for the maximum learning-rate immediately. This means we might have to go back and redo the process of setting hyperparameters starting again from scratch with another maximum learning rate for the cyclical momentum. 

Khushdeep got the responsibility to build a function for testing best value of momentum. 

Andreas would continue to do his experiments. 

I created a overleaf document for the report and added a bibfile with references to the two papers we have used so far. I also cleaned up the code even more and 

### 14 May 2019 

I ran a couple of tests for SuperConvergence. It's hard to know if the convergence is fast or not because I need to compare to regular training somehow, but first I would like to get the result they have in the paper. 

Now, the problem I see is that the training accuracy hits 100% before I even do annealing around 1200 iterations which is weird becaus in the paper they run more than 10k iterations with the same settings I use (excluding some hyperparameters which values were not given in the paper weight-decay and momentum) 

I also changed the ResNets BatchNorm2ds momentum from 0.1 to 0.95 which is used in the SuperConvergence paper. The plot becomes much less jittery than before. 

Now what is the reason that I don't get the same result? Not enough regularization I would guess. We don't want to overfit that early and the only way to not overfit is to lower the learning-rate right? That's what they do with "constant" learningrate. With constant learningrate they lower the learningrate when the loss evens out for the validation set. 

I did one experiment this evening with only 1150 iterations and it yielded an accuracy of 84% which is the best so far with Super Convergence. 

So what I have left to do is change the parameters the paper didn't give us i.e the weightdecay and the momentum. 

### 17 May 2019 

So I started training a piecewise constant learning-rate. So far I did 1000, 100, 100, 300 and dividing the learning-rate by 10 after each set or itterations. So far, I don't have a divergence but I don't see any improvement after 1100 iterations. Tomorrow I will try to run it for a longer time (maybe 1000) iterations at learningrate 0.001 and see if I can catch any divergence or improvement in the validation accuracy. 

I also found out how to create two networks with the same initial weights. torch.manual_seed(seed). 

For the Constant Learning Rate I used the same momentum and weight-decay I found best results for when doing the streamline hyperparameter setup. 0.95 momentum and 1e-9 weight-decay. 

### things to try for the SuperConvergence 

higher maximum learning rate 

add the whole validation set

change the weightdecay (higher value for more regulariztion) 

do fewer iterations 


