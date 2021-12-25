import pathlib, argparse 
from mascots.experiments.MTGenerator import MTGenerator
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MT cache configuration \
        based on device performance based on read/write and request size.")
    parser.add_argument("mt_name", help="The name of MT cache.")
    parser.add_argument("mt_dir", type=pathlib.Path, help="Path to MT cache device configurations.")
    args = parser.parse_args()

    generator = MTGenerator(args.mt_name, args.mt_dir)
    generator.generate_mt_caches("L-D1", "L-F1", "L-H1", "/home/pranav/mtc/mt_config/local")