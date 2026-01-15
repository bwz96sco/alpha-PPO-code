import numpy as np


distribution = {'h':[0.3, 20, 200], 'm':[0.5, 12, 125], 'l':[0.65, 6, 50]}

class RandomConfig:
    def __init__(self, load, partNum, machNum):
        
        # self.updateParam(partNum, machNum, load)
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,10000)
        

    def updateParam(self, partNum, machNum = -1, distType = None):
        self.partNum = partNum
        if machNum > 0:
            self.machNum = machNum
        else:
            group = (self.partNum - 5) // 10
            self.machNum = group * 5
        
        if distType is not None:
            self.distType = distType
            disParam = distribution[self.distType]
            self.tight = disParam[0]
            self.priority = disParam[1]
            self.maxTime = disParam[2]
            
        self.printParam()
            
    def printParam(self):
        print('Resource: ',' part, mach = ', \
              self.partNum, self.machNum)
        disParam = distribution[self.distType]
        print('Distribution Type :', self.distType, \
              ' | ', disParam)
        
load = 'h'        
scheConfig = RandomConfig(load, 15, 5)