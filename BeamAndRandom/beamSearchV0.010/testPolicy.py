# -*- coding: utf-8 -*-
import time
from utils import get_args
from venvs.excel import ExcelLog
from venvs.game import Game
from mcts_policy import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from venvs.EnvirConf import config

if __name__ == '__main__':
    args = get_args()
    config.updateParam(partNum = args.part_num, machNum = args.mach_num,\
                       distType = args.dist_type)
    
    ec = config.envConfig
    n_games = args.test_num
    beam_size = args.beam_size
    
    start = time.time()
    policy_mode = 'mcts_policy'
    base_model_name = str(ec.partNum)+ '-' + str(ec.machNum) +'-' + ec.distType
    name = args.env_name + '-' + base_model_name
    # env_dist_name = '(Env-' + ec.distType + ')' 
    log_name = name 
    
    path = './models/' + base_model_name + '-weight.model'
    #g2c useGpu
    update_net_model = PolicyValueNet(use_gpu= True,
                                      is_train = False,
                                      resblock_num=args.resblock_num)
    update_net_model.load_pretrained_weight(path)
    mcts_player = MCTSPlayer(update_net_model.policy_value_fn,
                             beam_size = beam_size,
                             mode = policy_mode)
    game = Game(update_net_model,beam_size,mode ='test',
                search_mode = policy_mode,seed=0) # need adjust to 0, bug in instance 16
#    sa search error is because last act is '-1' which making seq is error after '-1' exchanged to front
    
    logger = ExcelLog(log_name,True)
    grade_list = []
    for i in range(n_games):
        grade = game.start_play(mcts_player,
                                is_shown=0)
        grade_list.append(grade)
        print('n_play=',i+1,' grade:',grade)
        logger.saveTest(grade)
    average = sum(grade_list)/n_games
    print("num_playouts:{}, min: {}, average: {}, max:{}".format(
           100, min(grade_list), 
           average, max(grade_list)))    
    end = time.time()
    print("Execution Time: ", end - start)