import sys
import argparse
from streamlit.web import cli as stcli
from src.preprocessing import Preprocessor
from src.neural_network.train import Trainer as ResnetTrainer

if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'process_data':
        processor = Preprocessor()
        processor.train_test_validation_split()
        # processor.extract_mel_spectrogram()

    elif command == 'train':
        # parser.add_argument('-m', '--model', required=True, choices=['resnet'])
        # args, lf_args = parser.parse_known_args()

        # if args.model == 'resnet':
        trainer = ResnetTrainer()
        trainer()

    elif command == 'ui':
        sys.argv = ["streamlit", "run", "src/frontend/frontend.py"]
        sys.exit(stcli.main())
