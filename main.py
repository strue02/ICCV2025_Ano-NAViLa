import json
import argparse
import warnings

from engine.runner import run

def main():
    warnings.filterwarnings("ignore")

    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)    # Converting argparse Namespace to a dict.
    args.update(param)   # Add parameters from json

    run(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Pipeline of Ano-NAViLa.')
    parser.add_argument('--config', type=str, default='./configs/Ano-NAViLa_lymphnode.json', help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()