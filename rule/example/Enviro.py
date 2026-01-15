# -*- coding: utf-8 -*-
import numpy as np
import cv2
import gym
from .EnvirConf import *
from .GenPart import GenPartSeed,GenConsSeed

class ZhangEnv(gym.Env):
    def __init__(self,seed = 0,mode=None):
        self.machNum = MACHNUM
        self.partNum = PARTNUM
        self.stock = STOCKNUM
        self.commonSaveNum = SAVENUM
        self.maxStep = 10000
        
        self.toolNum = TOOLNUM
        self.workerNum = WORKERNUM 
        
        self.genTool = GenConsSeed('tool',self.machNum,self.toolNum,TOOLINITSEED)
        self.genWorker = GenConsSeed('worker',self.machNum,self.workerNum,WORKERINITSEED)
        self.genPart = GenPartSeed(num = self.partNum,seed = seed)
                
        machStateCount = 8
        stockStateCount = 18 #5 + 5
        commonStateCount = 2 
        self.machStateSize = (self.machNum,machStateCount)
        self.stockStateSize =(self.stock,stockStateCount)
        self.commonSaveStateSize = (self.commonSaveNum,commonStateCount)
        self.otherStateSize = 2
        self.stateSize = (self.machStateSize[0]*self.machStateSize[1]+self.stockStateSize[0]*self.stockStateSize[1] \
                          + self.commonSaveStateSize[0] * self.commonSaveStateSize[1] + self.otherStateSize,
                          self.machStateSize[0]*self.machStateSize[1],
                          self.stockStateSize[0]*self.stockStateSize[1],
                          self.commonSaveStateSize[0] * self.commonSaveStateSize[1],
                          self.otherStateSize)
        
        self.act_space = self.stock + 1
        self.mach_act_space = self.machNum + 1
        self.tool_act_space = self.toolNum + 1
        self.worker_act_space = self.workerNum + 1
        
        self.normReward = (PRIORITYMAX-1)*(self.machNum+self.stock)
        
        self.modeTypes = ['CANVAS','VALUE']
        mode = self.modeTypes[0] #if mode is None else mode 
        assert mode in self.modeTypes
        self.mode = mode
        self.viewer = None
        
        self.canvasInit()
        self.drawConsInits()
        self.actObsInit()
        self.stateSizeInit()

        
    def stateSizeInit(self):
        stateSize, machStateSize, stockStateSize, saveStateSize,otherStateSize = self.stateSize
        self.stateSizeAdd = [machStateSize,
                             stockStateSize+machStateSize,
                             stockStateSize+machStateSize+saveStateSize]
       
    def actObsInit(self):
#        self.action_space = gym.spaces.Discrete(self.act_space)
        self.action_space = [
                gym.spaces.Discrete(self.act_space),
                gym.spaces.Discrete(self.mach_act_space),
                gym.spaces.Discrete(self.tool_act_space),
                gym.spaces.Discrete(self.worker_act_space)
                ]
        
        self.noopAct = []
        for act_sp in self.action_space:
            self.noopAct.append(act_sp.n-1)
        
        self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.rowConLength, self.colConLength, 3),
                dtype=self.canvasType,
                )
        
        self.valueType = np.float32        
        self.observation_space_value  = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(self.stateSize[0],),
                dtype=self.valueType,
                )
          
    def setSeed(self,seed):
        self.genPart.setSeed(seed)
        
    def drawConsInits(self): #undo:tool drawing init
        self.dyColorCoef = 0.5
        
        self.drawConsNum = [self.stock,self.toolNum,self.workerNum]
        
        self.drawRowIndexes = [int(0)]
        for i in range(len(self.drawConsNum)-1):
            self.drawRowIndexes.append(int(self.drawConsNum[i+1] + self.drawRowIndexes[-1]))
        
        self.rowCon = max(sum(self.drawConsNum[1:])+1,self.drawConsNum[0])
        self.colCon = self.machNum*2 + 1
        assert (self.rowCon <= 84 and self.colCon <= 84)
        
        self.rowConLength = self.rowCon #* self.rowConFactor
        self.colConLength = self.colCon #* self.colConFactor
    
    def canvasInit(self): 
        self.row = self.machNum * (3+1) + self.stock * (3+1) #1 is interval between stock and mach 
        self.col = WORKHOURMAX*(1+TIGHT)
        self.rowFactor = 2
        self.colFactor = 2
        self.rowLength = self.row * self.rowFactor
        self.colLength = self.col * self.colFactor
        
        self.halfRowNum = int(self.col * self.colFactor/2)
        self.machInterVal = int(self.halfRowNum/self.machNum)
        
        self.halfRowFac = int(self.rowFactor/2)
                
        self.mach2 = self.machNum * 2
        self.mach3 = self.machNum * 3
        self.machEnd = self.mach3
        self.stock1 = self.machEnd + self.stock
        self.stock2 = self.machEnd + self.stock * 2   
        
        self.canvasType = np.uint8
        self._colorInit()
        
        priorityNum = PRIORITYMAX
        self.priFactor = int((self.col * self.colFactor)/priorityNum)

    def reset(self):
        self.clock = 1
        self.countNum = 1
        self.genPart.reset()
        
        self.workSta = np.zeros(self.machNum)
        self.machPrio = np.zeros(self.machNum)
        self.deadline = np.zeros(self.machNum)
        self.finals = np.zeros(self.machNum)
        self.baseWorkHour = np.zeros(self.machNum)
        self.saveUse = np.zeros(self.machNum)
        self.machTool = np.zeros((self.machNum,self.toolNum))
        self.machWorker = np.zeros((self.machNum,self.workerNum))
#        self.useTool = -1 * np.ones(self.machNum,dtype =np.int)
        
        self.partInfo = np.zeros((6,self.stock)) #index 4 is final
        self.partInfo[3] = np.inf
        self.partEquCons = np.zeros((self.stock,self.machNum))
        
        self.toolConsMat = self.genTool.consMat  
        self.workerConsMat = self.genWorker.consMat
        self.commonSave = np.zeros((2,self.commonSaveNum))

        self.lastClock = self.clock
        _,self.newMachFlag = self.getNewEvent()   

        greyCanvas = self.getEnvState()
        
        self.steps = 0
        self.grade = 0                    
        return greyCanvas
    
    def step(self,action):
        proceedingFlag = self.machSche(action)
        reward = 0
        if not proceedingFlag:
            reward = self.proceeding()
            self.grade += reward
            
            reward = reward/self.normReward

        greyCanvas = self.getEnvState()
        
        done,info = self.isDone()
        self.steps += 1
        return greyCanvas,reward,done,info
                    
    def render(self,mode = 'human'):
        if self.viewer is None:
            self._renderInit()
            
        self.canvas = self.plotState()
        zoomOut = cv2.resize(self.canvas,self.screenSize,interpolation=cv2.INTER_CUBIC)
        if mode == 'human':           
            self.viewer.imshow(zoomOut)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return zoomOut

    def _renderInit(self):      
        from gym.envs.classic_control import rendering
        self.viewer = rendering.SimpleImageViewer()
        self.screenSize = (400,400)
    
    def getEnvState(self):
        self.machMask,self.stockMask,self.toolMask,self.workerMask = self.getMask()
#        self.canvas = self.plotState()
        self.valueState = self.getState()
        greyCanvas = self.drawConState()

        return greyCanvas#state
                        
    def proceeding(self):
        self.clock += 1
        reward = self.getReward()
        newPartState,self.newMachFlag = self.getNewEvent() 

        if newPartState > 1:
            reward -= 1             
        return reward
    
    def machSche(self,action):
        if self.getConstrains(action,'PRE') and \
        self.getConstrains(action,'NOTNOOP') and\
        self.getConstrains(action,'CONS'):
            stockAct, machAct, toolAct, workerAct = action
            priority,deadline,workHour,release,final,saveUse = self.partInfo[:,stockAct]
            machIndex = machAct
                        
            self.workSta[machIndex] = workHour + self.clock
            self.machPrio[machIndex] = priority #priority
            self.deadline[machIndex] = deadline
            self.finals[machIndex] = final
            self.baseWorkHour[machIndex] = workHour
            self.saveUse[machIndex] = saveUse     
            self.machTool[machIndex,toolAct] = 1
            self.machWorker[machIndex,workerAct] = 1
            
            self.updateStocks(stockAct)
            
            scheFlag = True
            self.newMachFlag -= 1                        
        else:
            scheFlag = False

        return scheFlag  

    def getConstrains(self,action,ruleMode):
        stockAct, machAct, toolAct, workerAct = action
        constrains = True
        
        if ruleMode == 'PRE':
            constrains = constrains and self.newMachFlag > 0
            constrains = constrains and machAct < (self.mach_act_space - 1)
            constrains = constrains and stockAct < (self.act_space - 1)
            constrains = constrains and toolAct < (self.tool_act_space - 1)
            constrains = constrains and workerAct < (self.worker_act_space - 1)
            
        elif ruleMode == 'NOTNOOP':
            constrains = constrains and self.machPrio[machAct] == 0
            constrains = constrains and self.partInfo[0][stockAct] != 0 
            constrains = constrains and self.toolMach[toolAct].sum() == 0
            constrains = constrains and self.workerMach[workerAct].sum() == 0
            
        elif ruleMode == 'CONS':
            constrains = constrains and self.partInfo[5][stockAct] <= self.getRestSave()
            constrains = constrains and self.partEquCons[stockAct,machAct] > 0
            constrains = constrains and self.toolConsMat[toolAct,machAct] > 0
            constrains = constrains and self.workerConsMat[workerAct,machAct] > 0
  
        else:
            raise Exception('constrains mode error')
        return constrains
    
    @property
    def toolMach(self):
        return self.machTool.T
    @property
    def workerMach(self):
        return self.machWorker.T
      
    def getRestSave(self):
        save = self.saveUse.sum() + self.commonSave[0].sum()
        rest = self.commonSaveNum - save
        return rest
    
    def getState(self):
        machState,stockState,saveState = self.getMatState()
        machState = machState.flatten()#.astype(np.float32)
        stockState = stockState.flatten()#.astype(np.float32)
        saveState = saveState.flatten()
        
        stateSize, _, _, _,otherStateSize = self.stateSize
        state = np.zeros(stateSize,dtype = self.valueType)
        state[:self.stateSizeAdd[0]] = machState
        state[self.stateSizeAdd[0]:self.stateSizeAdd[1]] = stockState
        state[self.stateSizeAdd[1]:self.stateSizeAdd[2]] = saveState
        
        state[-2] = self.getRestSave()
        state[-1] = self.getStateTimeDelat(self.lastClock)
        return state

    def getMatState(self):
        machState = np.zeros(self.machStateSize)
        stockState = np.zeros(self.stockStateSize)
        saveState = np.zeros(self.commonSaveStateSize)
        
        restSaveNum = self.getRestSave()
      
        machMask = self.machPrio == 0       
        machState[:,0] = machMask[:self.machNum]
        machState[:,1] = self.machPrio#priority
        machState[:,2] = np.maximum(self.workSta - self.clock,0)#workHour
        machState[:,3] = np.maximum(self.deadline - self.clock,0)#deadline
        machState[:,4] = np.logical_and(self.deadline < self.clock,self.deadline > 0) #delay
        machState[:,5] = np.maximum(self.finals - self.clock,0)#final
        machState[:,6] = self.baseWorkHour #original workHour
        machState[:,7] = self.saveUse
        
        stockMask = self.partInfo[0] != 0 
        priority = self.partInfo[0]
        workHour = self.partInfo[2]
        deadline = np.maximum(self.partInfo[1] - self.clock,0) 
        delta = -np.maximum(deadline-workHour,0)/(workHour+10e-10) #-max(deadline-workHour,0)/workHour if workHour > 0 else 0
        expDelta = np.exp(delta/h) * priority/(workHour+10e-10)#ATC
        negaDelta = np.maximum(1+delta/Kt,0) * priority/(workHour+10e-10)#WCONVERT
        stockState[:,0] = stockMask
        stockState[:,1] = priority
        stockState[:,2] = workHour
        stockState[:,3] = deadline
        stockState[:,4] = np.logical_and(self.partInfo[1] < self.clock,self.partInfo[1] > 0)
        stockState[:,5] = delta
        stockState[:,6] = workHour/(priority + 10e-10) 
        stockState[:,7] = np.maximum(workHour,deadline)/(priority + 10e-10)  
        stockState[:,8] = expDelta 
        stockState[:,9] = negaDelta      
        final = np.maximum(self.partInfo[4] - self.clock,0)
        finalDelta = final - workHour
        stockState[:,10] = final
        stockState[:,11] = finalDelta
        stockState[:,12] = np.sign(finalDelta)
        stockState[:,13] = priority * workHour
        stockState[:,14] = stockMask & (finalDelta > 0)      
        stockSaveUse = self.partInfo[5]
        saveMask = restSaveNum >= stockSaveUse
        stockState[:,15] = stockSaveUse
        stockState[:,16] = saveMask
        stockState[:,17] = stockMask & (final - workHour > 0) & saveMask
            
        saveState[:,0] = self.commonSave[0]
        saveState[:,1] = np.maximum(self.commonSave[1] - self.clock,0)
            
        return machState,stockState,saveState
    

    def plotState(self):
        rowFactor = self.rowFactor
        colFactor = self.colFactor
        halfRowFac = self.halfRowFac
        halfRowNum = self.halfRowNum
        machInterVal = self.machInterVal
        
        canvas = np.zeros(shape = (self.rowLength,self.colLength,3),dtype = self.canvasType)
        index = 0
        for i in range(self.machNum + self.stock):
            if i < self.machNum:
                gridNum = int(self.workSta[i] - self.clock) #workHour
                deadline = int(max(self.deadline[i] - self.clock,0)) #restTime
                priority = int(self.machPrio[i])#state[i]
            else:
                partIndex = i - self.machNum
                gridNum = int(self.partInfo[2][partIndex])
                deadline = int(max(self.partInfo[1][partIndex] - self.clock,0))
                priority = int(self.partInfo[0][partIndex])
                               
            if gridNum > 0:
                canvas[index:index+rowFactor,:gridNum* colFactor] = self.colorBox[5]
                canvas[index+rowFactor:index+rowFactor*2,:max(0,deadline) * colFactor] = self.colorBox[1]
                canvas[index+rowFactor*2:index+rowFactor*3,:priority * self.priFactor] = self.colorBox[3]

            if i < self.machNum:
                canvas[index:index+3*rowFactor,-colFactor:] = self.colorBox[2]# egde color
                canvas[index+3*rowFactor:index+3*rowFactor+halfRowFac,:halfRowNum+machInterVal*(i+1)]\
                = self.colorBox[6] #interval
                index += rowFactor*4
            else:
                canvas[index+rowFactor:index+3*rowFactor,-colFactor:] = self.colorBox[4]# egde color
                canvas[index+3*rowFactor:index+3*rowFactor+halfRowFac,:halfRowNum] = self.colorBox[6] #interval
                index += rowFactor*4      
        return canvas#.reshape([1, 80, 80])
    
    def drawConState(self):#undo: dytoolCons  worktoolCons   
        drawIndex = self.drawRowIndexes
        drawing = np.zeros((self.rowConLength,self.colConLength,3),dtype = self.canvasType)
        
        dyToolCons = self.getDyToolCons()
        drawing[:drawIndex[1],:self.machNum] = np.repeat(dyToolCons[:,:,np.newaxis],3,axis = -1) * self.colorBox[-1]
        
        dyWorkerCons = self.getDyWorkerCons()
        drawing[drawIndex[1]:drawIndex[2],:self.machNum] = np.repeat(dyWorkerCons[:,:,np.newaxis],3,axis = -1) * self.colorBox[-1]
        
        dyEquCons = self.getDyEquCons()
        drawing[:self.stock,self.machNum+1:] = np.repeat(dyEquCons[:,:,np.newaxis],3,axis = -1) * self.colorBox[-1]
        return drawing
    
    def getDyEquCons(self):   
        dyEquCons = self.partEquCons.copy()
        dyEquCons[:,self.machPrio != 0] = self.dyColorCoef
        return dyEquCons
    
    def getDyToolCons(self):
        dyToolCons = self.toolConsMat - self.toolMach * (1 - self.dyColorCoef)#tool is busy
        return dyToolCons
    
    def getDyWorkerCons(self):
        dyWorkerCons = self.workerConsMat - self.workerMach * (1 - self.dyColorCoef)
        return dyWorkerCons

    def isDone(self):
        done = False
        info = dict()
        if self.countNum >= self.partNum or self.steps > self.maxStep:
            done = True
            info['episode'] = {'r':round(self.grade/self.partNum,4)}
        return done,info
    
    def getReward(self):
        reward = 0
        
        machIndexes = (self.finals - self.clock <= 0) & (self.finals > 0)
        machClearPunish = self.machPrio[machIndexes] * self.baseWorkHour[machIndexes]
        self.updateMachs(machIndexes)
        stockIndexes = (self.partInfo[4] - self.clock <= 0) & (self.partInfo[4] > 0)
        stockClearPunish = self.partInfo[0][stockIndexes] * self.partInfo[2][stockIndexes]
        self.updateStocks(stockIndexes)
        reward -= (machClearPunish.sum() + stockClearPunish.sum())
        
        dealIndex = np.where((self.deadline-self.clock) < 0)[0]
        deal = self.machPrio[dealIndex].sum()
        waitIndex = np.where((self.partInfo[1]-self.clock) < 0)[0]
        wait = self.partInfo[0,waitIndex].sum()
        reward += -1 * (deal + wait)
        
        return reward
      
    def getNewEvent(self):
        newPart,newPartFlag = self.genPart.getNewPart(self.clock)
        newPartState = 0
        
        if newPartFlag:
            newPartState += 1
            self.lastPartCome = self.clock
            idlePos = np.where(self.partInfo[0] == 0)[0]
            if len(idlePos):
#                ranIdlePos = np.random.choice(idlePos,1)[0]
                ranIdlePos = idlePos[0]
                self.updateStocks(ranIdlePos,newPart)
                self.countNum += 1
                
            else:
                newPartFlag = False 
                newPartState += 1
                
        machIndexes = np.where((self.workSta <= self.clock))[0]       
        newMachNum = len(machIndexes)
        if newMachNum > 0:
            self.useMachIndexSetSave(None,mode = 'SET')
            self.updateMachs(machIndexes)
        
        saveIndexes = np.where((self.commonSave[1] <= self.clock))[0]
        self.useMachIndexSetSave(saveIndexes,mode = 'CLEAR')
        
        return newPartState,newMachNum#newMachFlag
    
    def useMachIndexSetSave(self,indexes,mode):
        if mode == 'SET':
            indexes = np.where((self.workSta <= self.clock) & (self.workSta > 0))[0]
            num = len(indexes)
            if num > 0:
                saveIndexes = np.where(self.commonSave[0] == 0)[0][0:num]
                self.commonSave[0][saveIndexes] = self.saveUse[indexes]
                self.commonSave[1][saveIndexes] = self.deadline[indexes]
        elif mode == 'CLEAR':
            num = len(indexes)
            if num > 0:
                self.commonSave[:,indexes] = 0
        else:
            raise Exception('save mode error')
        
    
    def updateMachs(self,machIndexes):
        self.workSta[machIndexes] = 0
        self.machPrio[machIndexes] = 0
        self.deadline[machIndexes] = 0
        self.finals[machIndexes] = 0
        self.baseWorkHour[machIndexes] = 0
        self.saveUse[machIndexes] = 0
        self.machTool[machIndexes] = 0
        self.machWorker[machIndexes] = 0
#        toolIndex = 
        
    def updateStocks(self,stockIndex, newPart = None):
        if newPart is None:
            self.partInfo[:,stockIndex] = 0
            self.partInfo[3,stockIndex] = np.inf
            self.partEquCons[stockIndex,:] = 0
        else:
            self.partInfo[:,stockIndex] = np.array\
            ([newPart.priority,newPart.deadline,newPart.workHour,newPart.release,newPart.final,newPart.saveStock])
            self.partEquCons[stockIndex] = newPart.equCons

    def getStateTimeDelat(self,lastClock):
        delta = self.clock - lastClock
        self.lastClock = self.clock
        return delta
    
    def getMask(self):
        machBusyState = (self.machPrio == 0)
        
        toolMask = self.toolConsMat.T * (self.machTool.sum(0) == 0) #mach * tool
        
        workerMask = self.workerConsMat.T * (self.machWorker.sum(0) == 0)
        
        machMask = self.partEquCons * machBusyState * (toolMask.sum(1) != 0) * (workerMask.sum(1) != 0)
        
        stockMask = (self.partInfo[0] != 0) & \
        (self.partInfo[4] - self.partInfo[2] > self.clock) & \
        (self.partInfo[5] <= self.getRestSave()) &\
        (machMask.sum(1) > 0) 
    
        return machMask,stockMask,toolMask,workerMask
    
    def getActionMask(self):
        stockMaskOne = np.zeros(self.act_space)
        stockMaskOne[:-1] = self.stockMask
        if self.stockMask.sum() < 1:
            stockMaskOne[:] = 1
        stockMaskOut = stockMaskOne
        
        machMaskOut = np.zeros((self.act_space,self.mach_act_space))
        machMaskOut[:-1,:-1] = self.machMask        
        machMaskAddIndex = machMaskOut.sum(1) < 1
        machMaskOut[machMaskAddIndex,:] = 1
        
        toolMaskOut = np.zeros((self.mach_act_space,self.tool_act_space))
        toolMaskOut[:-1,:-1] = self.toolMask
        toolMaskAddIndex = toolMaskOut.sum(1) < 1
        toolMaskOut[toolMaskAddIndex,:] = 1
        
        workerMaskOut = np.zeros((self.mach_act_space,self.worker_act_space))
        workerMaskOut[:-1,:-1] = self.workerMask
        workerMaskAddIndex = workerMaskOut.sum(1) < 1
        workerMaskOut[workerMaskAddIndex,:] = 1
  
        return self.valueState,stockMaskOut,machMaskOut,toolMaskOut,workerMaskOut  #stockMaskOne
        
    
    def _colorInit(self):
        self.colorBox = [np.array([0,0,0]).astype(np.uint8).reshape((1,1,3))]#0
        self.colorBox.append(np.array([255,0,0]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([255,127,0]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([255,255,0]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([0,255,0]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([0,255,255]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([0,0,255]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([139,0,255]).astype(np.uint8).reshape((1,1,3)))
        self.colorBox.append(np.array([255,255,255]).astype(np.uint8).reshape((1,1,3)))#1 8
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
            
class ZhEnvSkipFrame(ZhangEnv):
    def __init__(self,seed = 0,mode=None):
        super().__init__(seed = seed,mode = mode)
        self.skipNum = SKIPNUM#float('inf')
    
    def reset(self):
        self.skipEps = 0
        state = super().reset()
        state, _, _, _ = self.step(self.noopAct)
        return state
    
    def step(self,action):      
        lastClock = self.lastClock #keep the lastClock which is before stepping
        state, reward, done, info = super().step(action)
        
        stepsReward = reward        
#        stockMask = self.stockMask
#        machMask = self.machMask
        
        skipCount = 0
        while not (done or skipCount >= self.skipNum or self.notSkipCon()):#decide condition
#            stockAct = self.act_space - 1
#            machAct = self.mach_act_space - 1
            action = self.noopAct#[stockAct,machAct]
            state,reward,done,info = super().step(action)
            
            stepsReward += reward            
            skipCount += 1
        self.valueState = self.reDealOtherState(self.valueState,lastClock)
        self.skipEps += 1
        skipState = state
        return skipState,stepsReward,done,info
    
    def reDealOtherState(self,state,lastClock):
        state[-1] = self.getStateTimeDelat(lastClock)
        return state
    
    def notSkipCon(self):
        stockMask = self.stockMask
        machMask = self.machMask
        
        action_mask = machMask * stockMask.reshape(-1,1)
        
        condition  = action_mask.sum() > 0
        return condition
        
ruleNameStr = ['FIFO','MAXPRI','WSPT','WMDD','ATC','WCONVERT']
def maxArray(p1,p2):
    p = []
    for i in range(len(p1)):
        p.append(max(p1[i],p2[i]))
    return np.array(p)

class ZhEnvSimpleWrap(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space_value = self.env.observation_space_value

    def reset(self):
        i_obs = self.env.reset()
        v_obs, self.stock_action_masks, self.mach_action_masks, self.tool_action_masks, self.worker_action_masks = \
            self.env.getActionMask()
        return i_obs, v_obs

    def step(self, action):
        mach_act, tool_act, worker_act = self.get_extra_act_by_rule(action)
        hybrid_act = (action, mach_act, tool_act, worker_act)
        i_obs, reward, done, info = self.env.step(hybrid_act)
        v_obs, self.stock_action_masks, self.mach_action_masks, self.tool_action_masks, self.worker_action_masks = \
            self.env.getActionMask()
        return (i_obs, v_obs), reward, done, info

    def random_step(self):
        action = np.random.choice(range(len(self.stock_action_masks)), 1, False,
                                  p=(self.stock_action_masks / self.stock_action_masks.sum()))[0]
        state, reward, done, info = self.env.step(action)
        return action, state, reward, done, info

    def get_extra_act_by_rule(self,action):
        machAct = self.env.mach_act_space - 1
        if action != self.env.act_space - 1:
            machs = np.where(self.env.machMask[action] > 0)[0]
            if len(machs) > 0:
                equConsReq = self.env.partEquCons[:, machs].sum(0)
                machIndex = np.argmin(equConsReq)
                machAct = machs[machIndex]
        #            machAct = machs[0]

        toolAct = self.env.tool_act_space - 1
        if action != self.env.act_space - 1 and machAct != self.env.mach_act_space - 1:
            tools = np.where(self.env.toolMask[machAct] > 0)[0]
            if len(tools) > 0:
                toolConsReq = self.env.toolConsMat[tools, :].sum(1)
                toolIndex = np.argmin(toolConsReq)
                toolAct = tools[toolIndex]
        #            toolAct = tools[0]

        workerAct = self.env.worker_act_space - 1
        if action != self.env.act_space - 1 and machAct != self.env.mach_act_space - 1:
            workers = np.where(self.env.workerMask[machAct] > 0)[0]
            if len(workers) > 0:
                workerConsReq = self.env.workerConsMat[workers, :].sum(1)
                workerIndex = np.argmin(workerConsReq)
                workerAct = workers[workerIndex]
        return machAct, toolAct, workerAct

    def get_extra_act_random(self,action):
        mach_act = np.random.choice(range(len(self.mach_action_masks[action])), 1, False,
                                    p=(self.mach_action_masks[action] / self.mach_action_masks[action].sum()))[0]
        tool_act = np.random.choice(range(len(self.tool_action_masks[mach_act])), 1, False,
                                    p=(self.tool_action_masks[mach_act] / self.tool_action_masks[mach_act].sum()))[0]
        worker_act = np.random.choice(range(len(self.worker_action_masks[mach_act])), 1, False,
                                      p=(self.worker_action_masks[mach_act] / self.worker_action_masks[
                                          mach_act].sum()))[0]
        return mach_act, tool_act, worker_act

    def get_act_by_rule(self, ruleNum):
        h = 1
        Kt = 1
        stock = np.where(self.env.stockMask > 0)[0]
        num = len(stock)
        if len(stock) > 0:
            priority, deadline, workHour, release, _, _ = self.env.partInfo[:, stock]
            p = np.sum(workHour) / len(stock)
            if ruleNum == 'FIFO':
                index = np.argmin(release)
            elif ruleNum == 'MAXPRI':
                index = np.argmax(priority)
            elif ruleNum == 'WSPT':
                index = np.argmin(workHour / priority)
            elif ruleNum == 'WMDD':
                index = np.argmin(maxArray(workHour, deadline - self.env.clock) / priority)  # WMDD
            elif ruleNum == 'ATC':
                index = np.argmax \
                    (priority * np.exp(
                        -maxArray(deadline - workHour - self.env.clock, np.zeros(len(priority))) / (p * h)) / workHour)
            elif ruleNum == 'WCONVERT':
                index = np.argmax \
                    ((priority / workHour) * maxArray(
                        1 - maxArray(deadline - workHour - self.env.clock, np.zeros(num)) / (Kt * workHour), np.zeros(num)))
            else:
                raise Exception('rule number error')
            action = stock[index]
        else:
            action = self.env.act_space - 1
        return action
