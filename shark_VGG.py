
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv_layer(chanIn, chanOut, kernel_size = 3, padding=0):
    return nn.Sequential(
        nn.Conv1d(chanIn, chanOut, kernel_size, padding=padding),        
        nn.ReLU(),
        nn.BatchNorm1d(chanOut)
        )

def dense(chanIn):
    return nn.Sequential(
        Flatten(),
        nn.Dropout(.25),
        nn.Linear(chanIn, 1000),
        nn.ReLU(),
        nn.BatchNorm1d(1000),
        nn.Linear(1000,100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 4)
    )

class Shark_VGG(nn.Module):
    def __init__(self, chn):
        super(Shark_VGG, self).__init__()
        chan = 8*2
        ks = 5
        pad = 2       
               
        self.conv11 = conv_layer(chn, chan, kernel_size=ks, padding=pad)
        self.conv12 = conv_layer(chan, chan, kernel_size=ks, padding=pad)
        self.pool13 = nn.MaxPool1d(2,2)
        self.conv14 = conv_layer(chan, chan*2, kernel_size=ks, padding=pad)
        self.conv15 = conv_layer(chan*2, chan*2, kernel_size=ks, padding=pad)
        self.pool16 = nn.MaxPool1d(2,2)
        self.conv17 = conv_layer(chan*2, chan*4, kernel_size=ks, padding=pad)
        self.conv18 = conv_layer(chan*4, chan*4, kernel_size=ks, padding=pad)
        self.fc = dense(768) #12*chan*4
        
    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool13(x)
        
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.pool16(x)
        
        x = self.conv17(x)
        x = self.conv18(x)

        x = self.fc(x)
        return x
    