"""
GMM Model Evaluation Tutorial - Dataset Comparison
================================================

This tutorial demonstrates how different GMM datasets look and how their
characteristics affect model performance:

1. Basic dataset comparison (simple, standard, complex)
2. SNR dataset comparison (high, moderate, low SNR)
3. Understanding how dataset properties impact clustering difficulty
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new high-level API
from scripts.evaluation.tutorial.src.visualization import VisualizationPipeline

# Define paths
tutorial_output_dir = Path('/mount/Storage/gmm-v4/scripts/evaluation/tutorial/output')
tutorial_output_dir.mkdir(parents=True, exist_ok=True)

def create_basic_dataset_comparison(pipeline):
    """Create a comparison of basic dataset types (simple, standard, complex)."""
    print("\n1. Creating basic dataset comparison...")
    print("   Comparing: Simple, Standard, Complex datasets (3 samples each)")
    print("   This shows how cluster separation varies across dataset types")
    
    # Generate 3 unique samples per dataset
    results = []
    
    dataset_types = ['simple', 'standard', 'complex']
    dataset_labels = ['Simple', 'Standard', 'Complex']
    
    # For each dataset type, generate 3 unique samples
    for dataset_type, label in zip(dataset_types, dataset_labels):
        dataset_results = pipeline._process_dataset_input(
            datasets=dataset_type,
            models=None,
            parameter_values=None,
            show=['points', 'true_centers'],
            num_samples=3
        )
        
        # Add proper titles to the results
        for i, result in enumerate(dataset_results):
            result['metadata']['title'] = f"{label} Sample {i+1}"
            results.append(result)
    
    # Create the grid
    from scripts.evaluation.tutorial.src.visualization import create_comparison_grid, save_figure
    
    titles = [result['metadata']['title'] for result in results]
    fig, axes = create_comparison_grid(
        results=results,
        layout='3x3',
        show_predictions=False,
        show_kmeans=False,
        titles=titles,
        figsize=(15, 15)
    )
    
    # Save the figure
    save_path = tutorial_output_dir / "all_datasets_comparison.png"
    save_figure(fig, save_path)
    
    print("   ✓ Basic dataset comparison saved!")
    print("   • Simple: Well-separated clusters, easy to identify")
    print("   • Standard: Moderate overlap, realistic clustering task")
    print("   • Complex: Significant overlap, challenging for models")
    return fig

def create_snr_dataset_comparison(pipeline):
    """Create a comparison of SNR-based datasets."""
    print("\n2. Creating SNR dataset comparison...")
    print("   Comparing: High, Moderate, Low SNR datasets (3 samples each)")
    print("   This shows how noise levels affect data visibility")
    
    # Generate 3 unique samples per dataset
    results = []
    
    dataset_types = ['high_snr_fixed', 'average_snr_fixed', 'low_snr_fixed']
    dataset_labels = ['High SNR', 'Moderate SNR', 'Low SNR']
    
    # For each dataset type, generate 3 unique samples
    for dataset_type, label in zip(dataset_types, dataset_labels):
        dataset_results = pipeline._process_dataset_input(
            datasets=dataset_type,
            models=None,
            parameter_values=None,
            show=['points', 'true_centers'],
            num_samples=3
        )
        
        # Add proper titles to the results
        for i, result in enumerate(dataset_results):
            result['metadata']['title'] = f"{label} Sample {i+1}"
            results.append(result)
    
    # Create the grid
    from scripts.evaluation.tutorial.src.visualization import create_comparison_grid, save_figure
    
    titles = [result['metadata']['title'] for result in results]
    fig, axes = create_comparison_grid(
        results=results,
        layout='3x3',
        show_predictions=False,
        show_kmeans=False,
        titles=titles,
        figsize=(15, 15)
    )
    
    # Save the figure
    save_path = tutorial_output_dir / "snr_datasets_comparison.png"
    save_figure(fig, save_path)
    
    print("   ✓ SNR dataset comparison saved!")
    print("   • High SNR (12 dB): Clear cluster structure visible")
    print("   • Moderate SNR (6 dB): Clusters still distinguishable")
    print("   • Low SNR (3 dB): Heavy noise obscures cluster structure")
    return fig

def create_comprehensive_dataset_grid(pipeline):
    """Create a comprehensive grid showing all dataset types."""
    print("\n3. Creating comprehensive dataset grid...")
    print("   Showing all dataset types in a single view")
    
    # Mix different dataset types in one visualization
    datasets = [
        'simple', 'standard', 'complex',
        'high_snr_fixed', 'average_snr_fixed', 'low_snr_fixed'
    ]
    
    titles = [
        'Simple', 'Standard', 'Complex',
        'High SNR', 'Moderate SNR', 'Low SNR'
    ]
    
    fig = pipeline.scatter_plot(
        datasets=datasets[:6],
        show=['points', 'true_centers'],
        layout='2x3',
        titles=titles[:6],
        figsize=(18, 12),
        save_path=tutorial_output_dir / "comprehensive_datasets_grid.png"
    )
    
    print("   ✓ Comprehensive dataset grid saved!")
    print("   Top row: Varying cluster separation (Simple → Complex)")
    print("   Bottom row: Varying noise levels (High → Low SNR)")
    return fig

def create_dataset_with_analysis(pipeline):
    """Create dataset comparison with KMeans analysis."""
    print("\n4. Creating dataset comparison with KMeans analysis...")
    print("   Adding KMeans clustering to see baseline performance")
    
    fig = pipeline.scatter_plot(
        datasets=['simple', 'standard', 'complex'],
        show=['points', 'true_centers', 'kmeans'],
        layout='1x3',
        titles=['Simple + KMeans', 'Standard + KMeans', 'Complex + KMeans'],
        figsize=(18, 6),
        save_path=tutorial_output_dir / "datasets_with_kmeans.png"
    )
    
    print("   ✓ Dataset comparison with KMeans saved!")
    print("   • Simple: KMeans performs well with separated clusters")
    print("   • Standard: KMeans shows some errors with overlapping clusters")
    print("   • Complex: KMeans struggles significantly with high overlap")
    return fig

def main():
    """Run the dataset comparison tutorial."""
    print("\n" + "="*60)
    print("GMM DATASET COMPARISON TUTORIAL")
    print("="*60)
    print("Exploring different GMM dataset characteristics")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Initialize the visualization pipeline
    print("\nInitializing visualization pipeline...")
    experiment_dir = '/mount/Storage/gmm-v4/output/final_experiments'
    pipeline = VisualizationPipeline(
        experiment_dir=experiment_dir,
        output_dir=tutorial_output_dir,
        device=device
    )
    
    # Run dataset comparisons
    create_basic_dataset_comparison(pipeline)
    create_snr_dataset_comparison(pipeline)
    create_comprehensive_dataset_grid(pipeline)
    create_dataset_with_analysis(pipeline)
    
    print("\n" + "="*60)
    print("DATASET COMPARISON COMPLETE")
    print("="*60)
    
    print("\nKEY INSIGHTS:")
    print("• Dataset complexity affects clustering difficulty:")
    print("  - Simple: Well-separated clusters, easy for any algorithm")
    print("  - Standard: Realistic overlap, requires good algorithms")
    print("  - Complex: High overlap, challenges even advanced models")
    
    print("\n• SNR (Signal-to-Noise Ratio) controls data clarity:")
    print("  - High SNR: Clean data, cluster structure visible")
    print("  - Moderate SNR: Some noise, but patterns remain clear")
    print("  - Low SNR: Heavy noise dominates, structure obscured")
    
    print("\n• KMeans baseline performance varies:")
    print("  - Works well on simple/high-SNR data")
    print("  - Degrades quickly with complexity/noise")
    print("  - This motivates need for learned models")
    
    print(f"\nAll visualizations saved to: {tutorial_output_dir}")
    
    # Show the plot if we're in a display environment
    if sys.platform != 'linux' or os.environ.get('DISPLAY', '') != '':
        print("\nDisplaying interactive plots...")
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print("\nNo display detected. All visualizations saved as images.")

if __name__ == "__main__":
    main() 