import torch
import torch.nn as nn

class MnistCNNModel(nn.Module):
    def __init__(self, in_channels, out, apply_softmax=False):
        super(MnistCNNModel, self).__init__()
        self.apply_softmax = apply_softmax

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, bias=False),  # input: (b, 1, 28, 28) -> (b, 32, 26, 26)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, bias=False),  # input: (b, 32, 26, 26) -> (b, 64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()     # input: (b, 64, 24, 24) -> (b, 64*24*24)

        self.fn = nn.Sequential(
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fn(x)
        if self.apply_softmax:
            x = nn.Softmax(dim=1)(x)
        return x

class ConvModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModel, self).__init__()
        self.act = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=3)    # input: (b, 3, 28, 28) -> (b, 32, 26, 26)
        self.conv_2 = nn.Conv2d(32, out_channels, kernel_size=3)    # input: (b, 32, 26, 26) -> (b, 64, 24, 24)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.conv_2(x)
        return x

class InnerModel(nn.Module):
    def __init__(self, in_params, out_params):
        super(InnerModel, self).__init__()
        self.layer_1 = nn.Linear(in_params, 128, bias=False)
        self.layer_2 = nn.Linear(128, out_params)
        self.dropout = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(128)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x

class MnistModel(nn.Module):
    def __init__(self, in_params = 784, out_params=10):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(in_params, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, out_params)
        # )

        # self.model = nn.Sequential()
        # self.model = nn.ModuleList()
        # self.model = nn.ModuleDict()
        # self.model.add_module("layer_1", nn.Linear(in_params, 128))
        # self.model.add_module("relu", nn.ReLU())
        # self.model.add_module("layer_2", nn.Linear(128, out_params))
        #
        self.layer_1 = nn.Linear(in_params, 128)
        self.act = nn.ReLU()
        self.layer_2 = nn.Linear(128, out_params)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.layer_2(x)

        # f = self.model(x)

        # for layer in self.model:      # ModuleList
        #     x = layer(x)
        # for key in self.model:
        #     leyer = self.model[key]     # ModuleDict
        #     x = leyer(x)

        return x

if __name__ == '__main__':
    BATCH_SIZE = 128
    # model = MnistModel()
    # print(model)

    # print(model.state_dict())
    # print(f"model.0.weight rows - {len(model.state_dict()['model.0.weight'])}")
    # print(f"model.0.weight columns - {len(model.state_dict()['model.0.weight'][0])}")
    # print(f"model.0.weight", model.state_dict()["model.0.weight"])
    # print("model.0.bias rows - ", len(model.state_dict()["model.0.bias"]))
    # print("model.0.bias - ", model.state_dict()["model.0.bias"])

    # for param in model.parameters():
    #     print(param)
    #     print(param.shape, end="\n\n")

    # input = torch.randn(BATCH_SIZE, 784, dtype=torch.float32)
    # out = model(input)
    # print("Shape - ", out.shape)
    # print("Layer_1", model.model.layer_1)
    # print("Layer_2", model.model.layer_2)

    # model.train()
    # model.eval()

    # output = torch.tensor([2.0])  # логит
    # sigmoid = nn.Sigmoid()
    # prob = sigmoid(output)
    # print(f"output - {output} = ", prob)
    #
    # logits = torch.tensor([2.0, 0.5, 1.2])  # 3 класса
    # softmax = nn.Softmax(dim=0)
    # probs = softmax(logits)
    # print("probs", probs)
    # print(torch.sum(probs))

    # model = nn.Sequential(
    #     nn.Conv2d(3, 32, kernel_size=3),    # input: (batch_size, 3, 28, 28) -> (batch_size, 32, 26, 26)
    #     nn.ReLU(),
    #     nn.Conv2d(32, 64, kernel_size=3),   # input: (batch_size, 32, 26, 26) -> (batch_size, 64, 24, 24)
    # )
    # model = ConvModel(in_channels=3, out_channels=64)
    #
    # input = torch.randn(BATCH_SIZE, 3, 28, 28, dtype=torch.float32)
    #
    # output = model(input)
    # l_output = output.reshape(BATCH_SIZE, -1)
    # l_output2 = output.flatten(start_dim=1, end_dim=-1)
    # l_output3 = nn.Flatten()(output)
    #
    # print("output - ",output.shape)
    # print("l_output - ", l_output.shape)
    # print("l_output2 - ", l_output2.shape)
    # print("l_output3 - ", l_output3.shape)

    input = torch.rand(BATCH_SIZE, 3, 28, 28, dtype=torch.float32)
    model = MnistCNNModel(in_channels=3, out=10)
    output = model(input)
    print("output - ", output)
    print("shape - ", output.shape)