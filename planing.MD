### after Saturday (4 May 2019)

Tasks:
* add a transform when creating the dataloader that normalizes the data. Do we want to do any augmentation for the experiments? 
* decide on which computer to use depending on time it takes to train the networks. 
* build a cyclical learning function (There is a cyclical function built into PyTorch) 
* find out how to change learning-rate/weight-decay/momentum on optimizer after creation
* draw a sketch of file structure for saving experiments 

Hyperparameters to perform experiments on. Order listed is order of application. This is also the order they were listed in the paper:
* Batchsize       [page 7-8]   128, 256, 512, 1024 depending on the amount of ram in GPU 
* Cyclic-momentum [page 8-11]  compared to Constant-momentum 
* Weight Decay    [page 11-13]

