# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class ToyPredictorNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_y = nn.Linear(input_dim, hidden_dim)

        self.fc1_xy = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_xy = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature, y):
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        if y.dim() == 1:
            y = y.view(-1,1)

        batch_size, num_samples = y.shape

        # Replicate
        x_feature = x_feature.view(batch_size, 1, -1).expand(-1, num_samples, -1) # (shape: (batch_size, num_samples, hidden_dim))

        # resize to batch dimension
        x_feature = x_feature.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, hidden_dim))
        y = y.reshape(batch_size*num_samples, -1) # (shape: (batch_size*num_samples, 1))

        y_feature = torch.tanh(self.fc1_y(y)) # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat([x_feature, y_feature], 1) # (shape: (batch_size*num_samples, 2*hidden_dim))

        xy_feature = torch.tanh(self.fc1_xy(xy_feature)) # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc2_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = torch.tanh(self.fc3_xy(xy_feature)) + xy_feature # (shape: (batch_size*num_samples, hidden_dim))
        score = self.fc4_xy(xy_feature) # (shape: (batch_size*num_samples, 1))

        score = score.view(batch_size, num_samples) # (shape: (batch_size, num_samples))

        return score


class ToyFeatureNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_x = nn.Linear(input_dim, hidden_dim)
        self.fc2_x = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # (x has shape (batch_size, 1))

        x_feature = F.relu(self.fc1_x(x)) # (shape: (batch_size, hidden_dim))
        x_feature = F.relu(self.fc2_x(x_feature)) # (shape: (batch_size, hidden_dim))

        return x_feature


class ToyNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(ToyNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        input_dim = 1
        hidden_dim = 10

        self.feature_net = ToyFeatureNet(input_dim, hidden_dim)
        self.predictor_net = ToyPredictorNet(input_dim, hidden_dim)

    def forward(self, x, y):
        # (x has shape (batch_size, 1))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.predictor_net(x_feature, y)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
