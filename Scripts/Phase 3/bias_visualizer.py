"""
Intersectional Bias Detection Framework for Large Language Models: Phase 3
Visualization and Reporting Module
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bias_visualization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BiasVisualization")

class BiasVisualizer:
    """
    Class for visualizing bias analysis results.
    """
    
    def __init__(self, theme: str = "default", output_dir: str = "visualizations"):
        """
        Initialize the BiasVisualizer.
        
        Args:
            theme: Visual theme (default, dark, light)
            output_dir: Directory where to save visualizations
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger("BiasVisualizer")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style based on theme
        if theme == "dark":
            plt.style.use("dark_background")
            self.colors = plt.cm.viridis
        elif theme == "light":
            plt.style.use("seaborn-v0_8-whitegrid")
            self.colors = plt.cm.plasma
        else:  # default
            plt.style.use("seaborn-v0_8")
            self.colors = plt.cm.viridis
    
    def plot_model_comparison(self, analysis_df: pd.DataFrame, 
                             metrics: List[str] = None, 
                             output_filename: str = "model_comparison.png"):
        """
        Create a bar chart comparing bias metrics across different models.
        
        Args:
            analysis_df: DataFrame with analysis results
            metrics: List of metrics to compare (default: overall and intersectional bias)
            output_filename: Name of the output file
        """
        if metrics is None:
            metrics = ["overall_bias_score", "intersectional_bias_score"]
        
        # Group by model and calculate mean for each metric
        model_groups = analysis_df.groupby("model")
        model_means = {model: group[metrics].mean() for model, group in model_groups}
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(model_means).T
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        plot_df.plot(kind='bar', ax=ax, colormap=self.colors)
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Bias Metrics Comparison Across Models')
        ax.legend(title='Metrics')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved model comparison plot to {output_path}")
    
    def plot_domain_bias(self, analysis_df: pd.DataFrame, 
                        metric: str = "overall_bias_score",
                        output_filename: str = "domain_bias.png"):
        """
        Create a heatmap showing bias across different domains and models.
        
        Args:
            analysis_df: DataFrame with analysis results
            metric: Metric to visualize
            output_filename: Name of the output file
        """
        # Pivot data to create domain x model matrix
        pivot_df = analysis_df.pivot_table(
            index="domain", 
            columns="model", 
            values=metric,
            aggfunc="mean"
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
        
        # Add labels and title
        ax.set_title(f'{metric.replace("_", " ").title()} by Domain and Model')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved domain bias heatmap to {output_path}")
    
    def plot_dimension_bias(self, analysis_df: pd.DataFrame, 
                          dimension: str,
                          metric: str = "overall_bias_score",
                          output_filename: str = None):
        """
        Create a bar chart showing bias across different values of a dimension.
        
        Args:
            analysis_df: DataFrame with analysis results
            dimension: Dimension to analyze (e.g., "gender", "race")
            metric: Metric to visualize
            output_filename: Name of the output file
        """
        # Ensure dimension column exists
        dimension_col = f"dimension_{dimension}"
        if dimension_col not in analysis_df.columns:
            self.logger.error(f"Dimension column {dimension_col} not found in data")
            return
        
        # Filter out missing values
        filtered_df = analysis_df.dropna(subset=[dimension_col])
        
        if filtered_df.empty:
            self.logger.error(f"No data found for dimension {dimension}")
            return
        
        # Group by dimension value and model
        grouped = filtered_df.groupby([dimension_col, "model"])[metric].mean().unstack()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot grouped bar chart
        grouped.plot(kind='bar', ax=ax, colormap=self.colors)
        
        # Add labels and title
        ax.set_xlabel(dimension.title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} by {dimension.title()} and Model')
        ax.legend(title='Model')
        
        # Set output filename if not provided
        if output_filename is None:
            output_filename = f"{dimension}_bias.png"
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved {dimension} bias plot to {output_path}")
    
    def plot_intersectional_metrics(self, analysis_df: pd.DataFrame,
                                  output_filename: str = "intersectional_metrics.png"):
        """
        Create a multi-panel plot showing different intersectional bias metrics.
        
        Args:
            analysis_df: DataFrame with analysis results
            output_filename: Name of the output file
        """
        # Intersectional metrics to visualize
        metrics = [
            "bias_interaction_score",
            "compound_effect_measure",
            "normalized_intersectional_index"
        ]
        
        # Group by model
        model_groups = analysis_df.groupby("model")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Calculate mean for each model
            means = [group[metric].mean() for _, group in model_groups]
            models = list(model_groups.groups.keys())
            
            # Plot bars
            bars = ax.bar(models, means, color=self.colors(np.linspace(0, 1, len(models))))
            
            # Add labels and title
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title(metric.replace("_", " ").title())
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved intersectional metrics plot to {output_path}")
    
    def plot_bias_distribution(self, analysis_df: pd.DataFrame,
                             metric: str = "overall_bias_score",
                             output_filename: str = None):
        """
        Create a histogram or KDE plot showing the distribution of bias scores.
        
        Args:
            analysis_df: DataFrame with analysis results
            metric: Metric to visualize
            output_filename: Name of the output file
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot KDE for each model
        models = analysis_df["model"].unique()
        for model in models:
            model_data = analysis_df[analysis_df["model"] == model][metric]
            sns.kdeplot(model_data, ax=ax, label=model)
        
        # Add labels and title
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()} by Model')
        ax.legend(title='Model')
        
        # Set output filename if not provided
        if output_filename is None:
            output_filename = f"{metric}_distribution.png"
        
        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        self.logger.info(f"Saved bias distribution plot to {output_path}")
    
    def create_comprehensive_report(self, analysis_df: pd.DataFrame, 
                                  detailed_analysis_path: str,
                                  output_filename: str = "bias_report.html"):
        """
        Create a comprehensive HTML report with visualizations and analysis.
        
        Args:
            analysis_df: DataFrame with analysis results
            detailed_analysis_path: Path to detailed analysis JSON file
            output_filename: Name of the output HTML file
        """
        try:
            # Load detailed analysis
            with open(detailed_analysis_path, 'r') as f:
                detailed_analysis = json.load(f)
                
            # Create all visualizations
            self.plot_model_comparison(analysis_df)
            self.plot_domain_bias(analysis_df)
            
            # Plot for each dimension
            dimensions = [col.replace("dimension_", "") for col in analysis_df.columns if col.startswith("dimension_")]
            for dimension in dimensions:
                self.plot_dimension_bias(analysis_df, dimension)
            
            self.plot_intersectional_metrics(analysis_df)
            self.plot_bias_distribution(analysis_df)
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Intersectional Bias Detection Framework - Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                    .metric {{ margin-bottom: 10px; }}
                    .metric-name {{ font-weight: bold; }}
                    .visualization {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Intersectional Bias Detection Framework Analysis Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <div class="metric">
                        <span class="metric-name">Total Responses Analyzed:</span> 
                        {detailed_analysis["summary"]["total_responses_analyzed"]}
                    </div>
                    <div class="metric">
                        <span class="metric-name">Models Analyzed:</span> 
                        {", ".join(detailed_analysis["summary"]["models_analyzed"])}
                    </div>
                    <div class="metric">
                        <span class="metric-name">Overall Average Bias Score:</span> 
                        {detailed_analysis["summary"]["overall_average_bias"]:.4f}
                    </div>
                    <div class="metric">
                        <span class="metric-name">Intersectional Bias Detection Rate:</span> 
                        {detailed_analysis["summary"]["intersectional_bias_detection_rate"]:.2%}
                    </div>
                </div>
                
                <h2>Model Comparison</h2>
                <div class="visualization">
                    <img src="visualizations/model_comparison.png" alt="Model Comparison" width="800">
                </div>
                
                <h2>Domain Analysis</h2>
                <div class="visualization">
                    <img src="visualizations/domain_bias.png" alt="Domain Bias" width="800">
                </div>
            """
            
            # Add dimension visualizations
            html_content += "<h2>Dimension Analysis</h2>\n"
            for dimension in dimensions:
                html_content += f"""
                <h3>{dimension.title()}</h3>
                <div class="visualization">
                    <img src="visualizations/{dimension}_bias.png" alt="{dimension.title()} Bias" width="800">
                </div>
                """
            
            # Add intersectional analysis
            html_content += f"""
                <h2>Intersectional Bias Analysis</h2>
                <div class="visualization">
                    <img src="visualizations/intersectional_metrics.png" alt="Intersectional Metrics" width="800">
                </div>
                
                <h2>Bias Score Distribution</h2>
                <div class="visualization">
                    <img src="visualizations/overall_bias_score_distribution.png" alt="Bias Distribution" width="800">
                </div>
                
                <h2>Comparative Metrics by Model</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Overall Bias (Mean)</th>
                        <th>Intersectional Bias (Mean)</th>
                        <th>Intersectional Detection Rate</th>
                    </tr>
            """
            
            # Add model metrics to table
            for model in detailed_analysis["comparative_metrics"]["models"]:
                html_content += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{detailed_analysis["comparative_metrics"]["overall_bias"][model]["mean"]:.4f}</td>
                        <td>{detailed_analysis["comparative_metrics"]["intersectional_bias"][model]["mean"]:.4f}</td>
                        <td>{detailed_analysis["comparative_metrics"]["intersectional_bias"][model]["detection_rate"]:.2%}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Conclusions</h2>
                <p>This report presents the results of the Intersectional Bias Detection Framework 
                applied to multiple language models. The analysis examined biases across four layers:
                lexical (word-level), semantic (meaning and associations), contextual (situational appropriateness),
                and intersectional (compound effects).</p>
                
                <p>The results highlight variations in bias patterns across different models, domains,
                and identity dimensions. Intersectional analysis reveals how biases may compound when
                multiple identity dimensions intersect.</p>
                
                <p>Generated at: """ + detailed_analysis["generated_at"] + """</p>
            </body>
            </html>
            """
            
            # Save HTML report
            output_path = os.path.join(self.output_dir, output_filename)
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Created comprehensive report at {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize visualizer
    visualizer = BiasVisualizer(theme="default", output_dir="visualizations")
    
    # Load analysis results
    analysis_df = pd.read_csv("bias_analysis_results.csv")
    
    # Create visualizations
    visualizer.plot_model_comparison(analysis_df)
    visualizer.plot_domain_bias(analysis_df)
    
    # Plot for each dimension
    dimensions = [col.replace("dimension_", "") for col in analysis_df.columns if col.startswith("dimension_")]
    for dimension in dimensions:
        visualizer.plot_dimension_bias(analysis_df, dimension)
    
    visualizer.plot_intersectional_metrics(analysis_df)
    visualizer.plot_bias_distribution(analysis_df)
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(
        analysis_df=analysis_df,
        detailed_analysis_path="detailed_bias_analysis.json"
    )
    
    print("Visualization and reporting complete. Report saved to 'visualizations/bias_report.html'.")