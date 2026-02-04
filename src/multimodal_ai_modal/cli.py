"""Command line interface for the multimodal AI modal."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core import MultimodalProcessor


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multimodal AI Modal - Process text, images, and audio"
    )
    parser.add_argument(
        "--text", type=str, help="Text input to process"
    )
    parser.add_argument(
        "--image", type=str, help="Image file path to process"
    )
    parser.add_argument(
        "--audio", type=str, help="Audio file path to process"
    )
    parser.add_argument(
        "--output", type=str, help="Output file path for results"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/DialoGPT-medium",
        help="Model name to use for processing"
    )
    parser.add_argument(
        "--format", type=str, choices=["json", "text"], default="json",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    if not any([args.text, args.image, args.audio]):
        print("Error: At least one input type must be provided", file=sys.stderr)
        return 1
    
    try:
        processor = MultimodalProcessor(model_name=args.model)
        
        # Prepare input data
        data = {}
        if args.text:
            data["text"] = args.text
        if args.image:
            data["image"] = args.image
        if args.audio:
            data["audio"] = args.audio
        
        # Process the data
        results = processor.process_multimodal(data)
        
        # Format and output results
        if args.format == "json":
            import json
            output = json.dumps(results, indent=2, default=str)
        else:
            output = str(results)
        
        if args.output:
            Path(args.output).write_text(output)
            print(f"Results saved to {args.output}")
        else:
            print(output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())