import argparse
import parse_args
import pickle
from models.baseline import BasicModel


def main(opt):
    base_model = BasicModel(opt)
    for e in range(opt['total_epochs']):
        base_model.train()

    


if __name__=="__main__":
    opt = parse_args.collect_args_main()
    main(opt)
