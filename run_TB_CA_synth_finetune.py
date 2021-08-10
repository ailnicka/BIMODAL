import sys
sys.path.append("./model")
from trainer import Trainer
for expname in ['BIMODAL_finetuning_TB_CA_small', 'BIMODAL_finetuning_TB_CA_synth']:
    t = Trainer(experiment_name=expname)
    t.run('TB_CA_experiment')