from preprocessing.main_preprocessor import preprocess_data

# preprocess training data
preprocess_data(filename_in='data/chembl24_cleaned_unique_canon',
                model_type='BIMODAL',
                starting_point='random',
                augmentation=10,
                min_len=-1)  # no limit, we have some short molecules

# preprocess fine-tuning data
preprocess_data(filename_in='data/TB_CA_synth',
                model_type='BIMODAL',
                starting_point='random',
                augmentation=10,
                min_len=-1)  # no limit, we have some short molecules

preprocess_data(filename_in='data/TB_CA_small',
                model_type='BIMODAL',
                starting_point='random',
                augmentation=10,
                min_len=-1)  # no limit, we have some short molecules

preprocess_data(filename_in='data/TB_CA',
                model_type='BIMODAL',
                starting_point='random',
                augmentation=10,
                min_len=-1)  # no limit, we have some short molecules