# -*- coding: utf-8 -*-
import time
from utils import get_args
from venvs.excel import ExcelLog
from venvs.game import Game
from mcts_policy import MCTSPlayer

from policy_value_net_pytorch import PolicyValueNet

from venvs.EnvirConf import envConfig as ec

if __name__ == '__main__':
    args = get_args()
    start = time.time()
    policy_mode =  'pure_policy' if args.mode == 'pure' else 'mcts_policy'
    
    base_model_name = str(ec.partNum)+ '-' + str(ec.machNum) + '-' + 'h'
    name = args.env_name + '-K' + str(args.beam_size) + '-' + base_model_name
    log_name = policy_mode + '-' + name
#    log_name = args.env_name + '_' + str(args.resblock_num) + \
#    '_' + policy_mode + '_' + str(ec.partNum)+str(ec.machNum) + '_' + ec.distType 
    
    n_games = args.test_num
    beam_size = args.beam_size
    
    
    path = './models/' + base_model_name + '-weight.model'
    update_net_model = PolicyValueNet(use_gpu=True,
                                      is_train = False,
                                      resblock_num=args.resblock_num)
    
    update_net_model.load_pretrained_weight(path)
    mcts_player = MCTSPlayer(update_net_model.policy_value_fn,
                             beam_size = beam_size,
                             mode = policy_mode)
    
    game = Game(update_net_model,beam_size,mode ='test',
                search_mode = policy_mode,seed=0) # need adjust to 0, bug in instance 16
#    sa search error is because last act is '-1' which making seq is error after '-1' exchanged to front
    grade_list = []
    logger = ExcelLog(log_name,True)
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