import sys
sys.path.append("./model")

from trainer import Trainer

t = Trainer(experiment_name='BIMODAL_pretraining')
t.run('TB_CA_experiment')