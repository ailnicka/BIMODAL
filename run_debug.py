import sys
sys.path.append("./model")
from trainer import Trainer

t = Trainer(experiment_name='BIMODAL_finetuning_debug')
t.run('TB_results')