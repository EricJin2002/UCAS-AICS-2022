# coding:utf-8
from pickletools import optimize
import numpy as np
import struct
import os
import time

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = \
                            np.sum(self.input_pad[idxn, :, idxh*self.stride : idxh*self.stride+self.kernel_size, idxw*self.stride : idxw*self.stride+self.kernel_size] \
                                * self.weight[:, :, :, idxc]) \
                                    + self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        
        self.input = input # [N, C, H, W]
        self.height = height = self.input.shape[2] + self.padding * 2
        self.width = width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) / self.stride + 1
        width_out = (width - self.kernel_size) / self.stride + 1
        
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
#        for idxn in range(self.input.shape[0]):
#            for idxc in range(self.channel_out):
#                for idxh in range(height_out):
#                    for idxw in range(width_out):
#                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
#                        self.output[idxn, idxc, idxh, idxw] = \
#                            np.sum(self.input_pad[idxn, :, idxh*self.stride : idxh*self.stride+self.kernel_size, idxw*self.stride : idxw*self.stride+self.kernel_size] \
#                                * self.weight[:, :, :, idxc]) \
#                                    + self.bias[idxc]
        
#        for idxh in range(height_out):
#            for idxw in range(width_out):
#                self.output[:, :, idxh, idxw] = \
#                    np.matmul(self.input_pad[:, :, idxh*self.stride : idxh*self.stride+self.kernel_size, idxw*self.stride : idxw*self.stride+self.kernel_size] \
#                        .reshape(self.input.shape[0], -1), self.weight.reshape(-1, self.channel_out)) + self.bias.reshape(1, -1)

        self.input_pad_ext = np.lib.stride_tricks.as_strided(
            self.input_pad,
            shape=(self.input_pad.shape[0], self.input_pad.shape[1], height_out, width_out, self.kernel_size, self.kernel_size), 
            strides=(self.input_pad.strides[0], self.input_pad.strides[1], self.input_pad.strides[2]*self.stride, self.input_pad.strides[3]*self.stride, self.input_pad.strides[2], self.input_pad.strides[3]),
        )
        self.output = np.einsum('nihwxy,ixyo->nohw', self.input_pad_ext, self.weight, optimize='optimal') + self.bias.reshape(1, -1, 1, 1)

        self.forward_time = time.time() - start_time
        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()
        
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
#        for idxn in range(top_diff.shape[0]):
#            for idxc in range(top_diff.shape[1]):
#                for idxh in range(top_diff.shape[2]):
#                    for idxw in range(top_diff.shape[3]):
#                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
#                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]
#                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
#                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        
#        for idxh in range(top_diff.shape[2]):
#            for idxw in range(top_diff.shape[3]):
#                self.d_weight += np.matmul( \
#                    self.input_pad[:, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] \
#                        .transpose(1, 2, 3, 0), \
#                    top_diff[:, :, idxh, idxw])
#                bottom_diff[:, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += \
#                    np.matmul(self.weight, top_diff[:, :, idxh, idxw].T).transpose(3, 0, 1, 2)
#        self.d_bias = np.sum(top_diff, (0, 2, 3))

        self.d_bias = np.sum(top_diff, (0, 2, 3))
        self.d_weight = np.einsum('nihwxy,nohw->ixyo', self.input_pad_ext, top_diff, optimize='optimal')

#        bottom_diff_ext = np.lib.stride_tricks.as_strided(
#            bottom_diff,
#            shape=self.input_pad_ext.shape, 
#            strides=self.input_pad_ext.strides,
#        )

#        bottom_diff_ext = np.einsum('nohw,ixyo->nihwxy', top_diff, self.weight, optimize='optimal')
#        bottom_diff_ext_enlarged = np.zeros((self.input_pad.shape[0], self.input_pad.shape[1], self.height, self.width, self.kernel_size, self.kernel_size))
#        bottom_diff_ext_enlarged[:, :, :1-self.kernel_size:self.stride, :1-self.kernel_size:self.stride, :, :] = bottom_diff_ext
        
        top_diff_enlarged = np.zeros((self.input_pad.shape[0], self.channel_out, self.height, self.width))
        top_diff_enlarged[:, :, :1-self.kernel_size:self.stride, :1-self.kernel_size:self.stride] = top_diff
        bottom_diff_ext_enlarged = np.einsum('nohw,ixyo->nihwxy', top_diff_enlarged, self.weight, optimize='optimal')

#         for idxh in range(top_diff.shape[2]):
#             for idxw in range(top_diff.shape[3]):
#                 bottom_diff[:, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += \
#                     bottom_diff_ext_enlarged[:, :, idxh*self.stride, idxw*self.stride, :, :] # bottom_diff_ext[:, :, idxh, idxw, :, :] # np.matmul(self.weight, top_diff[:, :, idxh, idxw].T).transpose(3, 0, 1, 2)
        
        for m in range(bottom_diff.shape[2]):
            for n in range(bottom_diff.shape[3]):
                for x in range(self.kernel_size):
                    for y in range(self.kernel_size):
                        bottom_diff[:, :, m, n] += \
                            bottom_diff_ext_enlarged[:, :, m-x, n-y, x, y] # bottom_diff_ext[:, :, idxh, idxw, :, :] # np.matmul(self.weight, top_diff[:, :, idxh, idxw].T).transpose(3, 0, 1, 2)
        
        bottom_diff = bottom_diff[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        self.backward_time = time.time() - start_time
        return bottom_diff
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding:-self.padding, self.padding:-self.padding]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = \
                            np.max(self.input[idxn, idxc, idxh*self.stride : idxh*self.stride+self.kernel_size, idxw*self.stride : idxw*self.stride+self.kernel_size])
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output
    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) / self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) / self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])

        self.input_ext = np.lib.stride_tricks.as_strided(
            self.input,
            shape=(self.input.shape[0], self.input.shape[1], height_out, width_out, self.kernel_size, self.kernel_size), 
            strides=(self.input.strides[0], self.input.strides[1], self.input.strides[2]*self.stride, self.input.strides[3]*self.stride, self.input.strides[2], self.input.strides[3]),
        )

        self.output = np.max(self.input_ext, axis=(4, 5))

        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        bottom_diff = np.zeros(self.input.shape)
        bottom_diff_ext = np.lib.stride_tricks.as_strided(
            bottom_diff,
            shape=(bottom_diff.shape[0], bottom_diff.shape[1], top_diff.shape[2], top_diff.shape[3], self.kernel_size, self.kernel_size), 
            strides=(bottom_diff.strides[0], bottom_diff.strides[1], bottom_diff.strides[2]*self.stride, bottom_diff.strides[3]*self.stride, bottom_diff.strides[2], bottom_diff.strides[3]),
        )
        
        #for idxn in range(top_diff.shape[0]):
        #    for idxc in range(top_diff.shape[1]):
        #        for idxh in range(top_diff.shape[2]):
        #            for idxw in range(top_diff.shape[3]):
        #                curren_max_index = np.argmax(self.input_ext[idxn, idxc, idxh, idxw, :, :])
        #                curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
        #                bottom_diff_ext[idxn, idxc, idxh, idxw, curren_max_index[0], curren_max_index[1]] = top_diff[idxn, idxc, idxh, idxw]

        bottom_diff_ext = np.einsum('nchw,nchwxy->nchxwy', top_diff, self.input_ext==np.max(self.input_ext, axis=(4,5))[...,np.newaxis,np.newaxis])
        bottom_diff = bottom_diff_ext.reshape(self.input.shape) # only work when kernel_size == stride

        return bottom_diff
    def backward_raw_book(self, top_diff):
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        max_index = np.array(np.where(
                            self.max_index[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]
                            ==1))[:,0] # 当kernel_size > stride时会出错，因为无法从窗口内的多个1中挑选正确index
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff
