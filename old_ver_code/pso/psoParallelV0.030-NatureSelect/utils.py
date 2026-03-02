import argparse

def get_args():
    parser = argparse.ArgumentParser(description='bbo')
    parser.add_argument(
        '--test-num',
        type=int,
        default=100,
        help='number of test episodes (default: 100)')
    parser.add_argument(
        '--part-num',
        type=int,
        default=15,
        help='part number set')
    parser.add_argument(
        '--mach-num',
        type=int,
        default=-1,
        help='mach number set, -1 means auto (now not use)')
    parser.add_argument(
        '--dist-type',
        default='h',
        help='environment distribution parameter (h/l/m)')
    parser.add_argument(
        '--popu',
        type=int,
        default=100,
        help='number of test episodes (default: 100)')
    parser.add_argument(
        '--iter',
        type=int,
        default=200,
        help='number of test episodes (default: 100)')
    args = parser.parse_args()
    return args