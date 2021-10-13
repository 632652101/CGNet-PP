from train import train
from val import val


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training',
                                     add_help=add_help)

    parser.add_argument("--val",
                        default=False,
                        help="Only val the model",
                        action="store_true", )

    parser.add_argument("--train",
                        default=False,
                        help="Only train the model",
                        action="store_true", )

    parser.add_argument("--trainval",
                        default=False,
                        help="Only train the model",
                        action="store_true", )

    parser.add_argument("--configfile",
                        default="config/cgnet/M3N21_512x1024.py",
                        help="training configuration file path.")

    parser.add_argument("--checkpoint",
                        default=None,
                        help="checkpoint file path.")

    parser.add_argument("--inform_data_file",
                        default="data/Cityscapes/cityscapes_inform.pkl",
                        help="inform data file path.")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # train(args)
    if args.train:
        train(args)
    elif args.val:
        val(args)