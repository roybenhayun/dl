import torch
import torch.nn as nn


class MyConvolution(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(1, 1), stride=1, padding=0):
        super(MyConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.biases = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, in_channels, input_height, input_width = x.shape
        output_height = (input_height - self.kernel_size[0] + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size[1] + 2 * self.padding) // self.stride + 1

        padded_input = self.pad_input(x, self.padding)

        output = torch.zeros((batch_size, self.out_channels, output_height, output_width), device=x.device)

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(output_height):
                    for w_out in range(output_width):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]

                        receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h_out, w_out] = torch.sum(receptive_field * self.weights[c_out]) + self.biases[
                            c_out]

        return output

    def pad_input(self, x, padding):
        batch_size, in_channels, input_height, input_width = x.shape
        padded_height = input_height + 2 * padding
        padded_width = input_width + 2 * padding

        padded_input = torch.zeros((batch_size, in_channels, padded_height, padded_width), device=x.device)
        padded_input[:, :, padding:padding + input_height, padding:padding + input_width] = x

        return padded_input



if __name__ == '__main__':
    print("ex 1")
    # Create an instance of MyConvolution
    conv_layer = MyConvolution(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

    # Pass an input tensor through the convolutional layer
    input_tensor = torch.randn(1, 3, 32, 32)
    output1 = conv_layer.forward(input_tensor)
    print(output1.shape)

    import torch
    from torch import nn
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    output2 = conv(torch.FloatTensor(input_tensor))
    print(output2.shape)

    print(output1[0][0][0][0] == output2[0][0][0][0])
    print("done")


