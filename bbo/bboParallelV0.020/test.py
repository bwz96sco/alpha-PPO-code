#from venvs.EnvirConf import config
from venvs.scheduler import Scheduler
from bbo import Biogeography

game = Scheduler('test',0)
bbo = Biogeography(game)

game.reset(True)
##bbo.BBOOp.reset(bbo.mu)
#bbo.initBBO(2)
bbo.iteration(50,2)
grade = bbo.BBOOp._calIndivi(bbo.BBOOp.bestIndi,False)