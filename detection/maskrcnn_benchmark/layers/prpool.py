from torch import nn
from external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d

class PrPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(PrPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return prroi_pool2d(input, rois, self.output_size[0], self.output_size[1], self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr
