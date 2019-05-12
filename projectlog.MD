# Project Log Team Jönsson

### 2 May 2019

Had a group-call at 15:00 decided the following:

### 12 May 2019 

it is unlikely that you will find where the model starts learning when doing the LR Range-Test. When increasing the lr linearly from a very low number. less than 1/10 of the plot will be in the lower range where the min_lr for model learning will be happening. 

I suggest we use a logscale as a last experiment to try to capture where the model starts learning and use that as a minium learning rate when doing Cyclic Learning and see what that gives us. 

At the moment we are supposed to use the maximum learning-rate divided by 3 or 4 as the minimum value. 

Doing experiments on batchsize of 256 without momentum (=0) or weight_decay (=0): 
Doing this on resnet56 doesn't seem to make the network diverge even if the upper limit for learning rate is set to 20. 
