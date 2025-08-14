# %%
"""Command-line interface for LOS Estimator."""

import argparse
import logging
import sys
from pathlib import Path

from los_estimator.config import load_configurations
from los_estimator.estimation_run import LosEstimationRun

logger = logging.getLogger("los_estimator")


def update_dict(d1, d2):
    for key, value in d2.items():
        if isinstance(value, dict):
            update_dict(d1.setdefault(key, {}), value)
        else:
            d1[key] = value


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Length of Stay Estimator for ICU data using deconvolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )
    parser.add_argument(
        "--config_file",
        help="Path to configuration file. Use only if you want to write the whole configuration",
    )
    parser.add_argument(
        "--overwrite_config_file",
        help="Path to a configuration file. Just overwrite the parameters that you want to change.",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="No showing of plots."
    )
    return parser


def main():
    parser = setup_parser()
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments passed: {unknown_args}")

    try:
        if args.config_file:
            logger.info("Custom configuration file was loaded.")
            path = Path(args.config_file)
        else:
            logger.info("Default configuration file was loaded.")
            path = Path(__file__).parent.parent / "default_config.toml"

        cfg = load_configurations(path)

        if args.overwrite_config_file:
            overwrite_cfg = load_configurations(args.overwrite_config_file)
            update_dict(cfg, overwrite_cfg)

        if args.show_plots:
            cfg["visualization_config"].show_figures = True
            cfg["animation_config"].show_figures = True
        else:
            cfg["visualization_config"].show_figures = False
            cfg["animation_config"].show_figures = False

        estimator = LosEstimationRun(
            data_config=cfg["data_config"],
            output_config=cfg["output_config"],
            model_config=cfg["model_config"],
            debug_config=cfg["debug_config"],
            visualization_config=cfg["visualization_config"],
            animation_config=cfg["animation_config"],
        )

        estimator.run_analysis()

        logger.info("LOS estimation completed successfully!")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import logging
        import sys

        logging.error("Error during execution", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
