import random
import argparse

# Define the number of lines and the range of seeds
num_lines = 25000
seed_range = range(0, num_lines)

# Create a list to store the lines
lines = []

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba"],
        default="mnist",
    )

    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default=None,
    )

    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "gd", "ga", "esd"],
        required=True,
    )
    return parser.parse_args()

def main(args):

    # Generate the lines with different seeds
    for seed in seed_range:
        line = f'--dataset {args.dataset} --device cuda:0 --load /gscratch/scrubbed/mingyulu/diffusion-attr/{args.dataset}/{args.method}/models/{args.removal_dist}/{args.removal_dist}_alpha=0.5_seed={seed} --removal_dist {args.removal_dist} --method {args.method} --removal_seed {seed}\n'
        lines.append(line)

    # Create sublists of lines every 1500 lines
    batch_size = 1500
    line_batches = [lines[i:i+batch_size] for i in range(0, len(lines), batch_size)]

    # Write the lines to text files
    for i, batch in enumerate(line_batches, start=1):
        filename = f'params/{args.removal_dist}_{args.method}_{args.dataset}_{i}.txt'

        with open(filename, 'w+') as file:
            file.writelines(batch)

    print(f'{num_lines} lines with different seeds have been split into {len(line_batches)} files.')

if __name__ == "__main__":
    args = parse_args()
    main(args)