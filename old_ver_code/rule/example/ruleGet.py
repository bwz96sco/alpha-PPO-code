# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt # plt 用于显示图片

from game.ExcelLog import ExcelLog
from game.Enviro import ZhEnvSkipFrame
#from game.Enviro2 import ZhEnvSkipFrame2
from game.EnvirConf import TESTSEED,EVALSEED

EPISODE = 240 # Episode limitation
STEP = 6000 # Step limitation in an episode
#TEST = 5 # The number of experiment test every 100 episode
ruleNameStr = ['FIFO','MAXPRI','WSPT','WMDD','ATC','WCONVERT']
def maxArray(p1,p2):
    p = []
    for i in range(len(p1)):
        p.append(max(p1[i],p2[i]))
    return np.array(p)


def heurRule(ruleNum):
#    z = np.random.randint(6)    
    h = 1
    Kt = 1
    stock = np.where(env.stockMask > 0)[0]
    num = len(stock)
    if len(stock) > 0:
        priority,deadline,workHour,release,_,_ = env.partInfo[:,stock]
        p = np.sum(workHour)/len(stock)
        if ruleNum == 'FIFO':
            index = np.argmin(release)
        elif ruleNum == 'MAXPRI':
            index = np.argmax(priority)
        elif ruleNum == 'WSPT':
            index = np.argmin(workHour/priority)
        elif ruleNum == 'WMDD':
            index = np.argmin(maxArray(workHour,deadline-env.clock)/priority)# WMDD
        elif ruleNum == 'ATC':
            index = np.argmax\
    (priority*np.exp(-maxArray(deadline-workHour-env.clock,np.zeros(len(priority)))/(p*h))/workHour)
        elif ruleNum == 'WCONVERT':
            index = np.argmax\
        ((priority/workHour)*maxArray(1-maxArray(deadline-workHour-env.clock,np.zeros(num))/(Kt*workHour),np.zeros(num)))
        else: 
            raise Exception('rule number error')
        action = stock[index]
    else:
        action = env.act_space - 1
    
    machAct = env.mach_act_space - 1
    if action != env.act_space - 1:
        machs = np.where(env.machMask[action] > 0)[0]
        if len(machs) > 0:
            equConsReq = env.partEquCons[:,machs].sum(0)
            machIndex = np.argmin(equConsReq)
            machAct = machs[machIndex]
#            machAct = machs[0]
            
    toolAct = env.tool_act_space - 1
    if action != env.act_space - 1 and machAct != env.mach_act_space - 1:
        tools = np.where(env.toolMask[machAct] > 0)[0]
        if len(tools) > 0:
            toolConsReq = env.toolConsMat[tools,:].sum(1)
            toolIndex = np.argmin(toolConsReq)
            toolAct = tools[toolIndex]
#            toolAct = tools[0]
            
    workerAct = env.worker_act_space - 1
    if action != env.act_space - 1 and machAct != env.mach_act_space - 1:
        workers = np.where(env.workerMask[machAct] > 0)[0]
        if len(workers) > 0:
            workerConsReq = env.workerConsMat[workers,:].sum(1)
            workerIndex = np.argmin(workerConsReq)
            workerAct = workers[workerIndex]
#            workerAct = workers[0]
    
    return [action,machAct,toolAct,workerAct]
    
for i in range(0,6):
    ruleIndex = ruleNameStr[i]
    env = ZhEnvSkipFrame(seed=TESTSEED)#ZhangEnv(seed=-1)#EVALSEED
    rewardCount = 0
    rewardLi = []
    exLog = ExcelLog(name=ruleIndex,isLog = True)
    for episode in range(EPISODE):
    # initialize task
        state = env.reset()
        k = 0
        li = []
        while True:
            action = heurRule(ruleIndex)#heurRule(env.state)#zhangRule(state)
            next_state,reward,done,info = env.step(action)
            li.append(next_state)      
            k += 1
            if done:
                break
            
        rewardSum = info['episode']['r'] * env.partNum
        rewardCount += rewardSum
        rewardLi.append(rewardSum)
        print(ruleIndex,' ',episode+1,' ',round(rewardSum/env.partNum,3),' mean = ',round(rewardCount/(episode+1)/env.partNum,3))
        exLog.saveTest(episode,rewardSum/env.partNum)
    print(rewardCount/(EPISODE*env.partNum))

