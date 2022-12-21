import os
import sys
import argparse
from tqdm import tqdm
from streamlit.web import cli as stcli
from src.preprocessing import Preprocessor
from src.neural_network.train import Trainer as ResnetTrainer
from src.supervised_learning.features_3_sec_compare import fit_models as fit_classical_models
from src.supervised_learning.evaluate_models import evaluate_models as evaluate_classical_models


if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'process_data':
        processor = Preprocessor()
        processor.scale_features()
        processor.train_test_validation_split()

    elif command == 'train_nn':
        print(f'Found: {len(os.listdir("configs"))} configs starting runs')
        i = 0
        for i, config in enumerate(tqdm(os.listdir('configs'), f'Experiment: {i+1}', leave=False)):
            trainer = ResnetTrainer(f'./configs/{config}')
            trainer()

    elif command == 'fit_classical_models':
        fit_classical_models()
    
    elif command == 'evaluate_classical_models':
        evaluate_classical_models('data/train_test_val_split')

    elif command == 'pretrained_logs':
        trainer = ResnetTrainer('./config.yaml')
        _, test_acc, test_cf = trainer.eval_iter(trainer.test_dl, True)

    elif command == 'ui':
        sys.argv = ["streamlit", "run", "src/frontend/frontend.py"]
        sys.exit(stcli.main())
