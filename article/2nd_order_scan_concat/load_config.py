# load_config.py
import configparser
import argparse

def load_config_and_args():
    parser = argparse.ArgumentParser(description="Run MHD simulation")

    # Argument positionnel existant
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the .ini configuration file"
    )

    # Options existantes
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Disable output for performance testing"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only execution"
    )

    # 🔴 OPTIONS DE PROFILING (AJOUT)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable host-loop profiling mode (recommended for ncu)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations before profiling"
    )

    parser.add_argument(
        "--profile-steps",
        dest="profile_steps",   # <-- correspond à args.profile_steps dans main.py
        type=int,
        default=3,
        help="Number of profiled iterations"
    )

    # Parse
    args = parser.parse_args()

    # Load config
    config = configparser.ConfigParser()
    config.read(args.config_file)

    return args, config