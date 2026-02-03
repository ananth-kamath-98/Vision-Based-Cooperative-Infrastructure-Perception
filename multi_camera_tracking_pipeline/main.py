import argparse
from multi_camera_tracker.pipeline import run_pipeline
from multi_camera_tracker.config import get_dataset_config

def main():
    """
    Main function to run the multi-camera tracking pipeline.
    """
    parser = argparse.ArgumentParser(description="Multi-camera tracking pipeline.")
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        choices=['LUMPI', 'CarLA'],
        help='The dataset to process (LUMPI or CarLA).'
    )
    args = parser.parse_args()

    print(f"Loading configuration for '{args.dataset}' dataset...")
    config = get_dataset_config(args.dataset)

    print("Starting tracking pipeline...")
    run_pipeline(config)
    print("Processing complete.")

if __name__ == "__main__":
    main()
