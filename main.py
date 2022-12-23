import os
import sys
import argparse
from tqdm import tqdm
from src.preprocessing import Preprocessor
from src.neural_network_keras.train import TFTrainer
from src.supervised_learning.features_3_sec_compare import fit_models as fit_classical_models
from src.supervised_learning.evaluate_models import evaluate_models as evaluate_classical_models

import matplotlib.pyplot as plt


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'process_data':
        processor = Preprocessor()
        processor.split_audio()
        processor.extract_mel_spectrogram()
        processor.convert_mel_folder_to_img('data/mel_specs', 'data/images_split')
        processor.scale_features()
        processor.train_test_validation_split()

    elif command == 'train_nn_keras':
        trainer = TFTrainer()
        trainer.run()

    elif command == 'dir_inference_nn_keras':
        trainer = TFTrainer()
        trainer.inference_dir('logs/keras/run-11/epoch-27.keras',
                              '/home/j/Desktop/Programming/Uni/Music-Genre-Classification/data/keras_dataset/val')

    elif command == 'train_nn_torch':
        raise DeprecationWarning('Pytorch code is deprecated in favour of keras, due to administrative requirements, please run "train_nn_keras" instead')
        print(f'Found: {len(os.listdir("configs"))} configs starting runs')
        i = 0
        for i, config in enumerate(tqdm(os.listdir('configs'), f'Experiment: {i+1}', leave=False)):
            if 'keras' in config: continue
            trainer = TorchTrainer(f'./configs/{config}')
            trainer()

    elif command == 'fit_classical_models':
        fit_classical_models()
    
    elif command == 'evaluate_classical_models':
        evaluate_classical_models('data/train_test_val_split')

    elif command == 'pretrained_logs':
        trainer = TorchTrainer('./config.yaml')
        _, test_acc, test_cf = trainer.eval_iter(trainer.test_dl, True)


