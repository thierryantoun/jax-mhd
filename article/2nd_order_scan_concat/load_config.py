import configparser
import argparse

def load_config_and_args():
    parser = argparse.ArgumentParser(description="Run MHD simulation")
    parser.add_argument("config_file", type=str, help="Path to the .ini configuration file")
    parser.add_argument("--benchmark", action="store_true", help="Disable output for performance testing")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only execution")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    return args, config