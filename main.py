import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Small Satellite Image Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for training
    train_parser = subparsers.add_parser('train', help="Train a model for a given task")
    train_parser.add_argument('--task', choices=['horizon', 'star', 'quality'], required=True,
                              help="Task name to train (horizon/star/quality)")

    # Subparser for evaluation
    eval_parser = subparsers.add_parser('evaluate', help="Evaluate a trained model on test data")
    eval_parser.add_argument('--task', choices=['horizon', 'star', 'quality'], required=True,
                             help="Task name to evaluate")

    # Subparser for capturing images using Pi camera (or simulation)
    capture_parser = subparsers.add_parser('capture', help="Capture images from PiCamera (or webcam simulation)")
    capture_parser.add_argument('--simulate', action='store_true',
                                help="Use OpenCV webcam simulation instead of PiCamera")

    args = parser.parse_args()

    # Dispatch to the appropriate functionality
    if args.command == 'train':
        from train import train_model
        train_model(args.task)
    elif args.command == 'evaluate':
        from evaluate import evaluate_model
        evaluate_model(args.task)
    elif args.command == 'capture':
        from capture import main_menu
        main_menu()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
