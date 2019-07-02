

class Inception_Block(nn.Module):
    def __init__(self, chanIn, chanOut):
        super(Inception_Block, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv1d(chanIn, chanOut, kernel_size=1, padding=0),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(chanOut))
        self.conv3 = nn.Sequential(
                        nn.Conv1d(chanIn, chanOut, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(chanOut))
        self.conv5 = nn.Sequential(
                        nn.Conv1d(chanIn, chanOut, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(chanOut))
        self.conv7 = nn.Sequential(
                        nn.Conv1d(chanIn, chanOut, kernel_size=7, padding=3),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(chanOut))        
        self.reduce = nn.Sequential(
                        nn.Conv1d(chanOut*4, chanOut, kernel_size=1, padding=0),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(chanOut))
            
    def forward(self, x):                
        cn1 = self.conv1(x)
        cn3 = self.conv3(x)
        cn5 = self.conv5(x)
        cn7 = self.conv7(x)        
        cat = torch.cat([cn1, cn3, cn5, cn7], dim=1)        
        x = self.reduce(cat)
        return x

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

class Shark_Inception(nn.Module):
    def __init__(self, chn):
        super(Shark_Inception, self).__init__()
        chan= 32
        
        self.block1 = Block(chn, chan)
        self.block2 = Block(chan, chan)
        self.block3 = Block(chan, chan)
        self.pool = nn.MaxPool1d(2)
        
        self.block4 = Block(chan, chan*2)
        self.block5 = Block(chan*2, chan*2)
        self.block6 = Block(chan*2, chan*2)
        self.pool2 = nn.MaxPool1d(2)

        self.block7 = Block(chan*2, chan*4)
        self.block8 = Block(chan*4, chan*4)
        self.block9 = Block(chan*4, chan*4)        
        self.pool3 = nn.MaxPool1d(2)

        self.block10 = Block(chan*4, chan*8)
        self.block11 = Block(chan*8, chan*8)
        self.block12 = Block(chan*8, chan*8)    
        
        self.fc = dense(1536)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)    
        x = self.pool (x)
        x = self.block4(x)    
        x = self.block5(x)    
        x = self.block6(x)  
        x = self.pool2 (x)
        x = self.block7(x)    
        x = self.block8(x)    
        x = self.block9(x) 
        x = self.pool3 (x)
        x = self.block10(x)    
        x = self.block11(x)    
        x = self.block12(x) 
        x = self.fc(x)
        return x