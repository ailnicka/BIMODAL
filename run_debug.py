from model.trainer import Trainer

t = Trainer(experiment_name='BIMODAL_debug')
t.beam_search('TB_results/beam_search_debug.csv')