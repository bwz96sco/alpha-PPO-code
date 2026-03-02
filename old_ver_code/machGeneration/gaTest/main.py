# -*- coding: utf-8 -*-
import time
from config import scheConfig as SC
from ga import GeneAlgorithm
from excel import ExcelLog

if __name__ == '__main__':
    testNum = 2
    popuSize = 200
    iteration = 400
    
    log_name = 'GA-' + str(SC.partNum) + '-' + str(SC.machNum) + '-' \
    + SC.distType + '(P' + str(popuSize) + '*I' + str(iteration) + ')'
    logger = ExcelLog(log_name,False)
    start = time.time()    
    grade_list = []
    ga = GeneAlgorithm()
    for i in range(testNum):
        ga.GaOp.sche.recreate('test')
        ga.iteration(popuSize,iteration,False)
        grade = ga.GaOp._calIndivi(ga.GaOp.bestIndi,False)
        grade_list.append(grade)
        print(i+1, ' ',grade)
        logger.saveTest(grade)
        # print(ga.GaOp.sche.T)
    average = sum(grade_list)/testNum
    print("num_playouts:, min: {}, average: {}, max:{}".format(
          min(grade_list), 
          average, max(grade_list)))

    end = time.time()
    print("Execution Time: ", end - start)