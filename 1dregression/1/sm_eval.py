# camera-ready

from datasets import ToyDatasetEvalKL # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 32

num_samples = 2048

model_id = "1-sm"
epoch = 75
num_models = 20

epsilon = 1.0e-30

with open("/root/ebms_regression/1dregression/1/gt_x_values_2_scores.pkl", "rb") as file: # (needed for python3)
    gt_x_values_2_scores = pickle.load(file)

val_dataset = ToyDatasetEvalKL()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

torch.autograd.set_grad_enabled(False)

KL_values = []
for model_i in range(num_models):
    network = ToyNet(model_id, project_dir="/root/ebms_regression/1dregression").cuda()

    network.load_state_dict(torch.load("/root/ebms_regression/1dregression/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))

    x_values = []
    x_values_2_scores = {}
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (x) in enumerate(val_loader):
        x = x.cuda().unsqueeze(1) # (shape: (batch_size, 1))

        y_samples = np.linspace(-3.0, 3.0, num_samples) # (shape: (num_samples, ))
        y_samples = y_samples.astype(np.float32)
        y_samples = torch.from_numpy(y_samples).cuda() # (shape: (batch_size, num_samples))

        x_features = network.feature_net(x)
        scores = network.predictor_net(x_features, y_samples.expand(x.shape[0], -1)) # (shape: (batch_size, num_samples))

        x_values.extend(x.squeeze(1).cpu().tolist())

        for i, x_val in enumerate(x):
            x_values_2_scores[x_val.item()] = scores[i,:].cpu().numpy()

    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    num_x_values = float(len(x_values))
    print (num_x_values)

    KL = 0.0
    for step, x_value in enumerate(x_values):
        scores = np.exp(x_values_2_scores[x_value].flatten()) # (shape: (num_samples, ))
        if np.sum(scores) > 1e-40:
            prob = scores/np.sum(scores) # (shape: (num_samples, ))
        else:
            scores = np.ones((num_samples, ))
            prob = scores/np.sum(scores)

        scores_gt = gt_x_values_2_scores[x_value].flatten() + epsilon # (shape: (num_samples, ))
        prob_gt = scores_gt/np.sum(scores_gt) # (shape: (num_samples, ))

        KL_i = np.sum(prob_gt*np.log(prob_gt/prob))
        KL = KL + KL_i/num_x_values

    print ("KL: %g" % KL)
    KL_values.append(KL)

    print (KL_values)
    print ("KL: %g +/- %g" % (np.mean(np.array(KL_values)), np.std(np.array(KL_values))))
    KL_values.sort()
    print (KL_values[0:5])
    print ("KL top 5: %g +/- %g" % (np.mean(np.array(KL_values[0:5])), np.std(np.array(KL_values[0:5]))))

    print ("##################################################################")
