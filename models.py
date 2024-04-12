import torch.nn as nn

#Класс блока Conv + Relu + MaxPool
class CnnBlock1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.act = nn.ReLU()

        self.layers = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (3, 3), stride=1, padding=1),
                self.drop,
                self.act,
                nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, x):
        return self.layers(x)



#Класс блока Conv + Relu + Conv + Relu + MaxPool (Не используется)
class CnnBlock2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.act = nn.ReLU()

        self.layers = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, (3, 3), stride=1, padding=1),
                self.drop,
                self.act,

                nn.Conv2d(out_ch, out_ch, (3, 3), stride=1, padding=1),
                self.drop,
                self.act,

                nn.MaxPool2d(kernel_size=2)
        )


    def forward(self, x):
        return self.layers(x)

#Финальная модель
class DrinksClassificationModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.count_class = conf.count_class
        self.ch = conf.ch
        self.count_blocks = conf.count_blocks

        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        self.blocks = []

        self.cnn_layer = nn.Conv2d(3, self.ch, (1, 1), stride=1, padding=1)
        for i in range(self.count_blocks):
            self.blocks.append(CnnBlock1(self.ch * 2 ** i, self.ch * 2 ** (i + 1)))     # Добавляем блоки CnnBlock1 в финальную модель

        self.blocks = nn.Sequential(*self.blocks)

        self.flat1 = nn.Flatten()
        inp_line1 = (conf.img_size // 2 ** self.count_blocks) ** 2 * self.ch * 2 ** self.count_blocks    # Подсчет входов для линейного слоя
        self.lin1 = nn.Linear(inp_line1, self.count_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        out = self.cnn_layer(x)
        out = self.blocks(out)
        out = self.flat1(out)
        out = self.lin1(out)
        out = self.softmax(out)
        return out