# camera-ready

from datasets import ToyDatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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
num_plot_samples = 1

model_id = "1-sm"

network = ToyNet(model_id, project_dir="/root/ebms_regression/1dregression").cuda()

epoch = 75

network.load_state_dict(torch.load("/root/ebms_regression/1dregression/training_logs/model_%s_0/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_id, epoch)))

val_dataset = ToyDatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

torch.autograd.set_grad_enabled(False)

x_values = []
x_values_2_scores = {}
network.eval() # (set in eval mode, this affects BatchNorm and dropout)
for step, (x) in enumerate(val_loader):
    if (step % 1000) == 0:
        print (step)
    x = x.cuda().unsqueeze(1) # (shape: (batch_size, 1))

    y_samples = np.linspace(-3.0, 3.0, num_samples) # (shape: (num_samples, ))
    y_samples = y_samples.astype(np.float32)
    y_samples = torch.from_numpy(y_samples).cuda() # (shape: (batch_size, num_samples))

    x_features = network.feature_net(x)
    scores = network.predictor_net(x_features, y_samples.expand(x.shape[0], -1))

    x_values.extend(x.squeeze(1).cpu().tolist())

    for i, x_val in enumerate(x):
        x_values_2_scores[x_val.item()] = scores[i,:].cpu().numpy()

print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

y_values = []
most_y = []
for step, x_value in enumerate(x_values):
    if (step % 1000) == 0:
        print (step)
    scores = np.exp(x_values_2_scores[x_value].flatten()) # (shape: (num_samples, ))
    if np.sum(scores) > 1e-40:
        prob = scores/np.sum(scores) # (shape: (num_samples, ))
    else:
        scores = np.ones((num_samples, ))
        prob = scores/np.sum(scores)

    max_score_ind = np.argmax(scores)
    max_score_y = np.linspace(-3.0, 3.0, num_samples)[max_score_ind]
    most_y.append(max_score_y)

    y_plot_samples = np.random.choice(np.linspace(-3.0, 3.0, num_samples), size=(num_plot_samples, ), p=prob) # (shape: (num_plot_samples, ))
    y_values.append(y_plot_samples[0])

plt.figure(1)
plt.plot(x_values, y_values, "k.", alpha=0.01, markeredgewidth=0.00001)
plt.xlabel("x")
plt.ylim([-1.5, 1.5])
plt.savefig("%s/pred_dens_epoch_%d.png" % (network.model_dir, epoch+1))
plt.close(1)
