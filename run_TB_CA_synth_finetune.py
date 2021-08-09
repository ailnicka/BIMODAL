from model.trainer import Trainer

t = Trainer(experiment_name='BIMODAL_finetuning_TB_CA_synth')
t.run('TB_CA_experiment')