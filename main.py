import sys
import argparse
from src.preprocessing import Preprocessor
from src.neural_network.train import train as resnet_train

if __name__ == '__main__':
    command = sys.argv[1]
    parser = argparse.ArgumentParser()

    if command == 'process_data':
        processor = Preprocessor()
        processor.extract_mel_spectrogram()

    elif command == 'train':
        parser.add_argument('-m', '--model', required=True, choices=['resnet'])
        args, lf_args = parser.parse_known_args()

        if args.model == 'resnet':
            resnet_train()