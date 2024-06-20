import argparse
 

def f(x, y):
    return (1/(x * y) + 1/(4 + y**2)) ** (1/2) - 1/(2 * y)


parser = argparse.ArgumentParser(description='Calculator')
parser.add_argument('--lr', default=None, type=float, help='x')
parser.add_argument('--rho', default=None, type=float, help='y')
args = parser.parse_args()

print("result: ", f(args.lr, args.rho))