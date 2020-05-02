# camera-ready

from datasets import ToyDataset # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributions

import math
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# NOTE! change this to not overwrite all log data when you train the model:
model_id = "2-dsm"

num_epochs = 75
batch_size = 32
learning_rate = 0.001

num_samples = 1024
print (num_samples)

###########################
sigma = 0.075
print (sigma)
###########################

train_dataset = ToyDataset()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

num_models = 20
for i in range(num_models):
    network = ToyNet(model_id + "_%d" % i, project_dir="/root/ebms_regression/1dregression").cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("model: %d/%d  |  epoch: %d/%d" % (i+1, num_models, epoch+1, num_epochs))

        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (xs, ys) in enumerate(train_loader):
            xs = xs.cuda().unsqueeze(1) # (shape: (batch_size, 1))
            ys = ys.cuda().unsqueeze(1) # (shape: (batch_size, 1))

            q_distr = torch.distributions.normal.Normal(loc=ys.squeeze(1), scale=sigma)
            y_samples = q_distr.sample(sample_shape=torch.Size([num_samples])) # (shape: (num_samples, batch_size))
            y_samples = torch.transpose(y_samples, 1, 0) # (shape: (batch_size, num_samples))
            y_samples = y_samples.reshape(-1).unsqueeze(1) # (shape: (batch_size*num_samples, 1))

            y_samples.requires_grad = True

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            x_features = x_features.view(xs.size(0), 1, -1).expand(-1, num_samples, -1) # (shape: (batch_size, num_samples, hidden_dim))
            x_features = x_features.reshape(xs.size(0)*num_samples, -1) # (shape: (batch_size*num_samples, hidden_dim))

            fs = network.predictor_net(x_features, y_samples) # (shape: (batch_size*num_samples, 1))
            fs = fs.squeeze(1) # (shape: (batch_size*num_samples))

            ########################################################################
            # compute loss:
            ########################################################################
            grad_y_fs = torch.autograd.grad(fs.sum(), y_samples, create_graph=True)[0] # (shape: (batch_size*num_samples, 1)) (like y_samples)

            ys = ys.view(xs.size(0), 1, -1).expand(-1, num_samples, -1) # (shape: (batch_size, num_samples, 1))
            ys = ys.reshape(xs.size(0)*num_samples, -1) # (shape: (batch_size*num_samples, 1))

            loss = torch.norm(grad_y_fs + (y_samples-ys)/(sigma**2), dim=1)**2 # (shape: (batch_size*num_samples))
            loss = torch.mean(loss)

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

        print("max_fs = {}".format(fs.max().item()))

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)

        # save the model weights to disk:
        checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(network.state_dict(), checkpoint_path)
