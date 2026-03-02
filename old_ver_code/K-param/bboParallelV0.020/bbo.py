import numpy as np

from venvs.scheduler import Scheduler
from venvs.EnvirConf import config

scheConfig = config.envConfig


class Biogeography:
    def __init__(self, scheduler):
        self.mutateRate = 0.05
        self.eliteNum = 2

        self.schConfig = scheConfig
        self.partNum = self.schConfig.partNum
        self.chromLen = self.partNum
        
        self.BBOOp = BiogeographyOP(scheduler)

    def iteration(self, popuSize,iterNum):
        self.initBBO(popuSize)
        for i in range(iterNum):
            self.evalAndSort()
            self.migrate()
            self.mutate()
            self.elitePolicy()
        self.evalAndSort()
        
    def initBBO(self, popuSize):
        self.popuSize = popuSize
        self.population = np.zeros((self.popuSize,self.chromLen),dtype = np.int)
        for m in range(self.popuSize):
            seq = list(range(self.partNum))
            np.random.shuffle(seq)
            self.population[m] = seq
            
        self.mu = np.arange(popuSize,0,-1)/(popuSize+1)
        self.lamb = 1 - self.mu
        
        self.temp_popu = self.population.copy()
        
        self.BBOOp.reset(self.mu)
#        self.elite = []

    def evalAndSort(self):
        grade_arr = self.BBOOp._calFitness(self.temp_popu)
        sortIndex = np.argsort(grade_arr)
        grade_arr_sort =  grade_arr[sortIndex]
        
        self.temp_popu = self.temp_popu[sortIndex]
        self.temp_grade = grade_arr_sort
        
        self.population = self.temp_popu.copy()
        self.elite = self.temp_popu[:self.eliteNum].copy()
        
        self.BBOOp.log(grade_arr_sort[0],self.temp_popu[0])
        
        
    def migrate(self):
        lamRanMat = np.random.rand(self.popuSize, self.chromLen)
        for i in range(self.popuSize):
            self.temp_popu[i] = self.migOp(lamRanMat[i],self.temp_popu[i],self.lamb[i])
            
    def mutate(self):
        mutateRan = np.random.rand(self.popuSize, self.chromLen)
        for i in range(self.popuSize):
            self.temp_popu[i] = self.mutOp(mutateRan[i],self.temp_popu[i])
            
    def elitePolicy(self):
        self.temp_popu[-self.eliteNum:,:] = self.elite


    def migOp(self,lamRand,ind,lamb):
        move_in_pos = np.where(lamRand<lamb)[0]
        num = move_in_pos.shape[0]
        if num > 0:
            ind_indexes = self.BBOOp.roulette(num)
            for i in range(num):
                pos = move_in_pos[i]
                trans_in_index = ind_indexes[i]
                value = self.population[trans_in_index,pos]
                ind = self.BBOOp.transferIn(ind,pos,value)
        new_ind = ind
        return new_ind

    def mutOp(self,randNum, ind):
        pos_list = np.where(randNum < self.mutateRate)[0]
        pos_num = pos_list.shape[0]
        if pos_num == 1:
            new_pos = np.random.randint(self.chromLen)
            pos = pos_list[-1]
            
            temp = ind[new_pos]
            ind[new_pos] = ind[pos]
            ind[pos] = temp
        elif pos_num > 1:
            value_list = ind[pos_list]
            np.random.shuffle(value_list)
            ind[pos_list] = value_list
        return ind
        
        

class BiogeographyOP:
    def __init__(self, sche):
        if sche is None:
            self.sche = Scheduler()
        else:
            self.sche = sche
#        self.loggerReset()
        
    def reset(self, mu):
        self.prob = mu/mu.sum()
        self.loggerReset()
        self.popuSize = mu.shape[0]
    
    def loggerReset(self):
        self.logger = []
        
        self.bestGrade = np.float('inf')
        self.bestIndi = None
    
    def log(self, grade, ind):
        if grade <= self.bestGrade:
            self.bestIndi = ind.copy()
            self.bestGrade = grade

    def _calFitness(self, popu):
        popuSize = popu.shape[0]
        gradeArr = np.zeros(popuSize)
        for i in range(popuSize):
            gradeArr[i] = self._calIndivi(popu[i])      
        return gradeArr
    
    def _calIndivi(self, individual,isPlot = False):
        indi = individual.tolist()
        grade = self.sche.scheStatic(indi)
        return grade
    
    def roulette(self, num):
#        if self.prob is None:
#            self.prob = mu/mu.sum()
        chooseIndex = np.random.choice(self.popuSize,
                                       size=num,
                                       p=self.prob,
                                       replace=True)
        return chooseIndex
    
    def transferIn(self, ind, pos, value):
        pos_value = np.where(ind ==value)[0]
        ind[pos_value] = ind[pos]
        ind[pos] = value
        return ind

