import sys
sys.path.append("./model")
from trainer import Trainer

t = Trainer(experiment_name='BIMODAL_finetuning_debug')
t.beam_search('TB_results/beam_search_test/molecules_fixed_1024.txt')
