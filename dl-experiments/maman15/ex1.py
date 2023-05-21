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
        self.biases = nn.Parameter(torch.zeros(out_channels))  # TODO: see 34min in lecture

    def forward(self, x):
        batch_size, in_channels, input_height, input_width = x.shape
        output_height = (input_height - self.kernel_size[0] + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size[1] + 2 * self.padding) // self.stride + 1

        padded_input = self.pad_input(x, self.padding)  # pad the whole input

        output = torch.zeros((batch_size, self.out_channels, output_height, output_width))

        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        # place kernel on receptive field, according to stride
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]

                        # calc output pixel[h, w]
                        receptive_field = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = torch.sum(receptive_field * self.weights[c]) + self.biases[c]

        return output

    def pad_input(self, x, padding):
        batch_size, in_channels, input_height, input_width = x.shape
        # add paddin in the image dimensions
        padded_height = input_height + 2 * padding
        padded_width = input_width + 2 * padding

        padded_input = torch.zeros((batch_size, in_channels, padded_height, padded_width))
        padded_input[:, :, padding:padding + input_height, padding:padding + input_width] = x

        return padded_input



if __name__ == '__main__':
    print("ex 1")


    # if getting "Process finished with exit code -1073741571 (0xC00000FD)", see:
    # NOTE: https://youtrack.jetbrains.com/issue/PY-17069#comment=27-1804561

    # pass an input tensor through MyConvolution
    conv_layer = MyConvolution(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
    input_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float)
    output1 = conv_layer.forward(input_tensor)
    print(output1.shape)

    # pass the input tensor through nn.Conv2d
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, dtype=torch.float)
    output2 = conv(input_tensor)
    print(output2.shape)

    # TODO: compare time it took for each
    print(output1[0][0][0][0] == output2[0][0][0][0])
    print("done")

    # reduce image size
    # TODO: explain stride
    conv_layer = MyConvolution(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=4, padding=1)
    input_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float)
    output1 = conv_layer.forward(input_tensor)
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=4, padding=1, dtype=torch.float)
    output2 = conv(input_tensor)
    print(output1.shape)
    print(output2.shape)


    # try Kernel 5*5 and keep output image size
    # TODO: Need padding 2 because...
    conv_layer = MyConvolution(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
    input_tensor = torch.randn(1, 3, 32, 32, dtype=torch.float)
    output1 = conv_layer.forward(input_tensor)
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=1, padding=2, dtype=torch.float)
    output2 = conv(input_tensor)
    print(output1.shape)
    print(output2.shape)

