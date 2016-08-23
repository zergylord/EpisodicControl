from episodic_agent import *
from ale_experiment import *
import atari_py
game_path = atari_py.get_game_path('pong')
ale = atari_py.ALEInterface()
#ale.setFloat('repeat_action_probability',0.0)
ale.loadROM(game_path)
num_actions = len(ale.getMinimalActionSet())
agent = EpisodicControlAgent(4)
experiment = ALEExperiment(ale,agent,84,84,'scale',5000,10000,0,4,True,30,rng = np.random.RandomState(123456))
experiment.run()
