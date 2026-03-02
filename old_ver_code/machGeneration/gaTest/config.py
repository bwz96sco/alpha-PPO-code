import numpy as np

from excel import ExcelDeal

distribution = {
        'h' : [0.3, 20, 200],
        'm' : [0.5, 12, 125],
        'l' : [0.65, 6, 50]
        }

class ScheConfig:
    def __init__(self):
        ex = ExcelDeal()
        
        self.partMat,self.machMat = ex.getPaLi('ft10')
        self.partNum,self.orderNum = self.partMat.shape

class RandomConfig:
    def __init__(self):
        self.partNum = 35
        self.distType = 'h'
#------------Adjusting parameter up-------
        
        group = (self.partNum - 5) // 10
        disParam = distribution[self.distType]
        self.machNum = group * 5
        print('Resource: ',' part, mach = ', \
              self.partNum, self.machNum)
        print('Distribution Type :', self.distType, \
              ' | ', disParam)
              
        self.tight = disParam[0]
        self.priority = disParam[1]
        
        self.maxTime = disParam[2]
        self.minTime = 5
        
        self.testSeed = 0
        self.valSeed = 1000
        self.trainSeed = np.random.randint(2000,10000)
        
        self.period = 50

scheConfig = RandomConfig()