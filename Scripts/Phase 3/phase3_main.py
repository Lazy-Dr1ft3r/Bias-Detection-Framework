"""
Intersectional Bias Detection Framework for Large Language Models: Phase 3
Main Script - Run the complete bias detection and analysis process
"""

import os
import sys
import argparse
import logging
import pandas as pd
from bias_detector import BiasDetector, ResponseAnalyzer
from bias_visualizer import BiasVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bias_framework.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BiasFramework")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Intersectional Bias Detection Framework")
    
    parser.add_argument("--input", "-i", type=str, default="combined_model_responses.csv",
                       help="Path to input CSV file with model responses")
    
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                       help="Directory to store output files")
    
    parser.add_argument("--stereotype-data", type=str, default=None,
                       help="Path to custom stereotype data JSON file")
    
    parser.add_argument("--lexicon", type=str, default=None,
                       help="Path to custom bias lexicon JSON file")
    
    parser.add_argument("--context-rules", type=str, default=None,
                       help="Path to custom context rules JSON file")
    
    parser.add_argument("--theme", type=str, default="default", choices=["default", "dark", "light"],
                       help="Visual theme for visualizations")
    
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip analysis and only generate visualizations from existing results")
    
    parser.add_argument("--skip-visualizations", action="store_true",
                       help="Skip visualizations and only perform analysis")
    
    return parser.parse_args()

def main():
    """Run the complete Phase 3 process."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for output files
    analysis_results_path = os.path.join(args.output_dir, "bias_analysis_results.csv")
    detailed_analysis_path = os.path.join(args.output_dir, "detailed_bias_analysis.json")
    visualizations_dir = os.path.join(args.output_dir, "visualizations")
    report_path = os.path.join(visualizations_dir, "bias_report.html")
    
    # Run analysis if not skipped
    if not args.skip_analysis:
        logger.info("Starting bias analysis")
        
        # Initialize bias detector with custom resources if provided
        bias_detector = BiasDetector(
            stereotype_data_path=args.stereotype_data,
            lexicon_path=args.lexicon,
            context_rules_path=args.context_rules
        )
        
        # Initialize response analyzer
        response_analyzer = ResponseAnalyzer(bias_detector)
        
        # Analyze responses
        analysis_results = response_analyzer.analyze_responses_from_csv(
            input_path=args.input,
            output_path=analysis_results_path
        )
        
        # Export detailed analysis
        response_analyzer.export_detailed_analysis(
            analysis_results=analysis_results,
            output_path=detailed_analysis_path
        )
        
        logger.info(f"Analysis complete. Results saved to {analysis_results_path} and {detailed_analysis_path}")
    else:
        logger.info("Skipping analysis as requested")
        # Check if analysis files exist
        if not os.path.exists(analysis_results_path):
            logger.error(f"Analysis results file not found at {analysis_results_path}")
            sys.exit(1)
        if not os.path.exists(detailed_analysis_path):
            logger.error(f"Detailed analysis file not found at {detailed_analysis_path}")
            sys.exit(1)
        
        # Load existing analysis results
        analysis_results = pd.read_csv(analysis_results_path)
    
    # Generate visualizations if not skipped
    if not args.skip_visualizations:
        logger.info("Starting visualization generation")
        
        # Initialize visualizer
        visualizer = BiasVisualizer(
            theme=args.theme,
            output_dir=visualizations_dir
        )
        
        # Create comprehensive report
        visualizer.create_comprehensive_report(
            analysis_df=analysis_results,
            detailed_analysis_path=detailed_analysis_path
        )
        
        logger.info(f"Visualization and reporting complete. Report saved to {report_path}")
    else:
        logger.info("Skipping visualizations as requested")
    
    logger.info("Phase 3 processing complete")


if __name__ == "__main__":
    main()