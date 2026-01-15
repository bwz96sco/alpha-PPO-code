# -*- coding: utf-8 -*-
import time
from ga import GeneAlgorithm
from venvs.excel import ExcelLog
from utils import get_args

from venvs.EnvirConf import config
from venvs.scheduler import Scheduler

if __name__ == '__main__':
    args = get_args()
    config.updateParam(partNum = args.part_num, machNum = args.mach_num,\
                       distType = args.dist_type)

    testNum = args.test_num
    popuSize = args.popu
    iteration = args.iter

    SC = config.envConfig
    log_name = 'GA-' + str(SC.partNum) + '-' + str(SC.machNum) + '-' \
    + SC.distType + '(P' + str(popuSize) + '-I' + str(iteration) + ')'
    
    logger = ExcelLog(log_name,True)
    start = time.time()    
    grade_list = []
    
    game = Scheduler('test',0)
    ga = GeneAlgorithm(game)
    for i in range(testNum):
        game.reset(True)
        ga.iteration(popuSize,iteration,False)
        
#        wrong_grade = ga.GaOp._calIndivi(ga.GaOp.bestIndi,False)
        grade = ga.GaOp.bestGrade
        grade_list.append(grade)
        print(i+1, ' bestGrade = ',grade)
#        print(i+1, ' bestGrade = ',grade, ' | wrong_grade = ', wrong_grade)
        logger.saveTest(grade)
        # print(ga.GaOp.sche.T)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)