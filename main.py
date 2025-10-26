"""
Main entry point for the Nuclear Fusion Analyzer
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.data.generator import FusionDataGenerator
from src.models.fusion_predictor import FusionPredictor
from src.visualization.plotter import FusionPlotter
from src.utils.config import load_config
from src.utils.logger import setup_logging


class FusionAnalyzer:
    """
    Main Nuclear Fusion Analyzer class that orchestrates the entire analysis pipeline.
    """
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize the Fusion Analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(self.config.get('logging', {}))
        
        self.data_generator = FusionDataGenerator(self.config.get('data', {}))
        self.predictor = FusionPredictor(self.config.get('models', {}))
        self.plotter = FusionPlotter(self.config.get('visualization', {}))
        
        logger.info("Nuclear Fusion Analyzer initialized successfully")
    
    def generate_data(self, num_samples: int = 10000, save_path: str = None):
        """
        Generate synthetic fusion data.
        
        Args:
            num_samples: Number of samples to generate
            save_path: Path to save generated data
        """
        logger.info(f"Generating {num_samples} fusion data samples")
        data = self.data_generator.generate_dataset(num_samples)
        
        if save_path:
            data.to_csv(save_path, index=False)
            logger.info(f"Data saved to {save_path}")
        
        return data
    
    def train_models(self, data_path: str = None, data=None):
        """
        Train fusion prediction models.
        
        Args:
            data_path: Path to training data
            data: Pandas DataFrame with training data
        """
        logger.info("Training fusion prediction models")
        
        if data_path:
            import pandas as pd
            data = pd.read_csv(data_path)
        
        if data is None:
            raise ValueError("Either data_path or data must be provided")
        
        results = self.predictor.train_all_models(data)
        logger.info("Model training completed")
        return results
    
    def predict(self, input_data):
        """
        Make predictions on new data.
        
        Args:
            input_data: Input data for prediction
        """
        logger.info("Making fusion predictions")
        return self.predictor.predict(input_data)
    
    def analyze_and_visualize(self, data, output_dir: str = "outputs"):
        """
        Perform comprehensive analysis and generate visualizations.
        
        Args:
            data: Data to analyze
            output_dir: Directory to save outputs
        """
        logger.info("Performing fusion data analysis and visualization")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        self.plotter.plot_plasma_parameters(data, save_path=f"{output_dir}/plasma_parameters.html")
        self.plotter.plot_fusion_efficiency(data, save_path=f"{output_dir}/fusion_efficiency.html")
        self.plotter.plot_correlation_matrix(data, save_path=f"{output_dir}/correlation_matrix.html")
        
        logger.info(f"Analysis completed. Results saved to {output_dir}")
    
    def run_full_pipeline(self, num_samples: int = 10000, output_dir: str = "outputs"):
        """
        Run the complete fusion analysis pipeline.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save outputs
        """
        logger.info("Starting full fusion analysis pipeline")
        
        # Generate data
        data = self.generate_data(num_samples)
        
        # Train models
        model_results = self.train_models(data=data)
        
        # Analyze and visualize
        self.analyze_and_visualize(data, output_dir)
        
        logger.info("Full pipeline completed successfully")
        return {
            'data': data,
            'model_results': model_results,
            'output_dir': output_dir
        }


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Nuclear Fusion Analyzer")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "train", "predict", "analyze", "full"],
        default="full",
        help="Analysis mode to run"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to input data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = FusionAnalyzer(args.config)
        
        if args.mode == "generate":
            data = analyzer.generate_data(
                num_samples=args.samples,
                save_path=f"{args.output_dir}/fusion_data.csv"
            )
            logger.info(f"Generated {len(data)} samples")
            
        elif args.mode == "train":
            if not args.data_path:
                raise ValueError("--data-path required for train mode")
            results = analyzer.train_models(data_path=args.data_path)
            logger.info("Training completed")
            
        elif args.mode == "analyze":
            if not args.data_path:
                raise ValueError("--data-path required for analyze mode")
            import pandas as pd
            data = pd.read_csv(args.data_path)
            analyzer.analyze_and_visualize(data, args.output_dir)
            
        elif args.mode == "full":
            results = analyzer.run_full_pipeline(args.samples, args.output_dir)
            logger.info("Full pipeline completed successfully")
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()