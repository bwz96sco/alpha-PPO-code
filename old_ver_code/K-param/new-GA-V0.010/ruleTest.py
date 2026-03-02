# -*- coding: utf-8 -*-
import time
from utils import get_args
from venvs.excel import ExcelLog
from venvs.scheduler import Scheduler


from venvs.EnvirConf import config

if __name__ == '__main__':
    args = get_args()
    config.updateParam(partNum = args.part_num, machNum = args.mach_num,\
                       distType = args.dist_type)
    
    ec = config.envConfig
    n_games = args.test_num
    
    start = time.time()
    policy_mode =  args.mode
    name = args.env_name + '-'\
    + str(ec.partNum)+ '-' + str(ec.machNum) + '-' + ec.distType
    log_name = policy_mode + '-' + name
    
    
    logger = ExcelLog(log_name,True)        
    game = Scheduler('test',0)
    grade_list = []
    for i in range(n_games):
        grade = game.scheRule(policy_mode)
        grade_list.append(grade)
        print('n_play=',i+1,' grade:',grade)
        logger.saveTest(grade)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
           n_games, min(grade_list), 
           average, max(grade_list)))    
    end = time.time()
    print("Execution Time: ", end - start)