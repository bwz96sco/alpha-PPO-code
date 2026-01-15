# -*- coding: utf-8 -*-
import time

from utils import get_args

from pso import ParticleSwarmOptimization
from excel import ExcelLog
from genpart import Scheduler
from config import scheConfig

SC = scheConfig
if __name__ == '__main__':
    args = get_args()
    SC.updateParam(partNum = args.part_num, machNum = args.mach_num,\
                       distType = args.dist_type)
    c1 = 2
    c2 =2.1
    Wstart = 0.9
    Wend = 0.4

    popuSize =  args.popu
    iterNum = args.iter

    testNum = args.test_num
    
    sche = Scheduler()
    pso=ParticleSwarmOptimization(c1,c2,Wstart,Wend,popuSize,sche)
    
    log_name = 'EA-' + str(SC.partNum) + '-' + str(SC.machNum) + '-' \
    + SC.distType + '(P' + str(popuSize) + '-I' + str(iterNum) + ')'
    start = time.time()    
    grade_list = []
    logger = ExcelLog(log_name,True)
    for i in range(testNum):
        pso.PsoOp.sche.recreate('test')
        pso.initPopulation()
        pso.iteration(iterNum)
        # grade = pso.PsoOp._calIndivi(pso.PsoOp.globalBestIndi,False)
        grade = pso.PsoOp.globalBestGrade
        grade_list.append(grade)
        print(i+1, ' ',grade)
        logger.saveTest(grade)
        # print(pso.PsoOp.sche.T)
        
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))
    end = time.time()
    print("Execution Time: ", end - start)