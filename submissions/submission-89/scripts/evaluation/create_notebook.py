import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GMM Model Evaluation (Refactored)\n",
                "\n",
                "This notebook demonstrates the evaluation of GMM models using the refactored evaluation API."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Import necessary libraries\n",
                "import torch\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import matplotlib as mpl\n",
                "from matplotlib.ticker import MultipleLocator\n",
                "from pathlib import Path\n",
                "\n",
                "# Import our refactored evaluation modules\n",
                "from scripts.evaluation.src.eval_utils import (\n",
                "    evaluate,\n",
                "    evaluate_with_snr,\n",
                "    evaluate_with_flow,\n",
                "    evaluate_with_dataset,\n",
                "    evaluate_with_kmeans,\n",
                "    get_flow_prediction,\n",
                "    results_to_dataframe,\n",
                "    compare_results\n",
                ")\n",
                "from scripts.evaluation.src.io import (\n",
                "    load_experiment,\n",
                "    save_results,\n",
                "    load_results,\n",
                "    get_cached_results,\n",
                "    create_data_loader\n",
                ")\n",
                "from scripts.evaluation.src.plots import (\n",
                "    set_gmm_template,\n",
                "    plot_flow_vs_snr,\n",
                "    plot_multi_snr_comparison,\n",
                "    plot_metrics_vs_flow,\n",
                "    plot_metrics_vs_snr,\n",
                "    plot_entropy_vs_wasserstein,\n",
                "    create_metrics_tables\n",
                ")\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.append(str(Path.cwd().parent.parent))\n",
                "\n",
                "# Set device\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f\"Using device: {device}\")\n",
                "\n",
                "# Set output directory\n",
                "output_dir = Path(\"/mount/Storage/gmm-v4/output\")\n",
                "if not output_dir.exists():\n",
                "    output_dir = Path(\"../../../output\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Setup Visualization Style\n",
                "\n",
                "Configure the plotting style for consistent visualizations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Set up visualization style\n",
                "plt.rcdefaults()\n",
                "mpl.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams.update({\n",
                "    'figure.figsize': (10, 5),\n",
                "    'figure.dpi': 100,\n",
                "    'font.size': 12,\n",
                "    'axes.titlesize': 14,\n",
                "    'axes.labelsize': 12,\n",
                "    'xtick.labelsize': 10,\n",
                "    'ytick.labelsize': 10,\n",
                "    'legend.fontsize': 10,\n",
                "    'lines.linewidth': 2,\n",
                "    'grid.alpha': 0.3\n",
                "})\n",
                "\n",
                "# Set GMM template for plotly visualizations\n",
                "set_gmm_template()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load Models\n",
                "\n",
                "Load the GMM models we want to evaluate."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Define model paths\n",
                "model_paths = {\n",
                "    \"16 Layers\": output_dir / 'final_experiments' / 'baseline_16_layers',\n",
                "    \"32 Layers\": output_dir / 'final_experiments' / 'baseline_32_layers',\n",
                "    \"64 Layers\": output_dir / 'final_experiments' / 'baseline_64_layers',\n",
                "    \"Simple 16 layers\": output_dir / 'final_experiments' / 'simple_16_layers',\n",
                "    \"Hard 16 layers\": output_dir / 'final_experiments' / 'hard_16_layers'\n",
                "}\n",
                "\n",
                "# Load models\n",
                "models = {}\n",
                "configs = {}\n",
                "for model_name, model_path in model_paths.items():\n",
                "    try:\n",
                "        model, config, _ = load_experiment(\n",
                "            exp_dir=model_path,\n",
                "            dataset_name=\"high_snr_fixed\",  # Default dataset\n",
                "            batch_size=16,\n",
                "            num_samples=512,\n",
                "            device=device,\n",
                "            load_best=False  # Use final model\n",
                "        )\n",
                "        models[model_name] = model\n",
                "        configs[model_name] = config\n",
                "        print(f\"Loaded model: {model_name}\")\n",
                "    except Exception as e:\n",
                "        print(f\"Failed to load model {model_name}: {e}\")\n",
                "\n",
                "# Check loaded models\n",
                "print(f\"\\nSuccessfully loaded {len(models)} models.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Flow Prediction Analysis\n",
                "\n",
                "Analyze how models adapt their flow speed based on SNR values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Plot flow speeds vs SNR\n",
                "snr_db_range = np.linspace(0, 20, 50)  # 0 to 20 dB\n",
                "snr_db_np = np.array(snr_db_range)\n",
                "\n",
                "# Predict flow speeds for each model\n",
                "flow_data = []\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    # Get flow predictor function\n",
                "    predictor = get_flow_prediction(model)\n",
                "    \n",
                "    # Get model config\n",
                "    model_config = {\n",
                "        \"name\": model_name,\n",
                "        \"layers\": configs[model_name].model.transformer.num_layers\n",
                "    }\n",
                "    \n",
                "    # Get flow speeds for each SNR value\n",
                "    for snr in snr_db_range:\n",
                "        flow_speed = predictor(snr).item()\n",
                "        flow_data.append({\n",
                "            \"model_name\": model_name,\n",
                "            \"snr_db\": snr,\n",
                "            \"flow_speed\": flow_speed,\n",
                "            \"layers\": model_config[\"layers\"],\n",
                "            \"total_flow\": flow_speed * model_config[\"layers\"]\n",
                "        })\n",
                "\n",
                "# Convert to DataFrame\n",
                "flow_df = pd.DataFrame(flow_data)\n",
                "\n",
                "# Create flow vs SNR plot\n",
                "fig = plot_flow_vs_snr(flow_df, show_total_flow=True)\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Evaluation With True SNR\n",
                "\n",
                "Evaluate models using the SNR values from the data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Evaluate models on high SNR dataset\n",
                "dataset_name = \"high_snr_fixed\"\n",
                "model_results = {}\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    # Evaluate model with dataset using true SNR values\n",
                "    results = evaluate_with_dataset(\n",
                "        model=model,\n",
                "        dataset_name=dataset_name,\n",
                "        batch_size=16,\n",
                "        num_samples=512,\n",
                "        metrics=[\"entropy\", \"log_wasserstein\", \"log_kmeans_wasserstein\"],\n",
                "        device=device\n",
                "    )\n",
                "    \n",
                "    # Store results (ensure it's not None or a string)\n",
                "    if results is not None and not isinstance(results, str):\n",
                "        model_results[model_name] = results\n",
                "    else:\n",
                "        print(f\"Skipping invalid results for {model_name}\")\n",
                "    \n",
                "    # Convert results to DataFrame for analysis\n",
                "    try:\n",
                "        df = results_to_dataframe(results)\n",
                "        \n",
                "        # Calculate statistics\n",
                "        stats = df.describe()\n",
                "        print(f\"\\nModel: {model_name} - Results Summary:\")\n",
                "        print(stats)\n",
                "    except Exception as e:\n",
                "        print(f\"Error processing results for {model_name}: {str(e)}\")\n",
                "\n",
                "# Create comparison dataframe if we have valid results\n",
                "if model_results:\n",
                "    # Pass a list of (name, results) pairs for clearer organization\n",
                "    model_results_list = [(name, results) for name, results in model_results.items()]\n",
                "    comparison_df = compare_results(model_results_list, id_column=\"model_name\")\n",
                "    print(\"\\nComparison DataFrame:\")\n",
                "    print(comparison_df.head())\n",
                "else:\n",
                "    print(\"\\nNo valid results to compare.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Evaluation With Custom SNR Values\n",
                "\n",
                "Evaluate models using custom SNR values to see how they behave across different noise scenarios."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Evaluate models with custom SNR values\n",
                "num_snrs = 4\n",
                "test_snr_values = torch.linspace(3, 15, num_snrs).to(device)\n",
                "print(f\"Custom SNR values: {test_snr_values.cpu().numpy()}\")\n",
                "\n",
                "# Create data loader with batch_size=num_snrs to match the number of SNR values\n",
                "data_loader = create_data_loader(\n",
                "    dataset_name=\"high_snr_fixed\",\n",
                "    batch_size=num_snrs,  # Make batch size match number of SNR values\n",
                "    num_samples=512,\n",
                "    device=device,\n",
                "    fixed_data=True\n",
                ")\n",
                "\n",
                "# Get the first batch of data\n",
                "batch = next(iter(data_loader))\n",
                "inputs = batch[0]  # Data is the first element in the tuple\n",
                "targets = batch[1] if len(batch) > 1 else None  # Targets is the second element if available\n",
                "\n",
                "# Verify input shape matches SNR values\n",
                "print(f\"Input shape: {inputs.shape}, SNR values shape: {test_snr_values.shape}\")\n",
                "if inputs.shape[0] != len(test_snr_values):\n",
                "    print(\"Warning: Batch size mismatch between inputs and SNR values.\")\n",
                "    # Method 1: Repeat the input to match SNR values size\n",
                "    if inputs.shape[0] == 1:\n",
                "        inputs = inputs.repeat(len(test_snr_values), 1, 1)\n",
                "        print(f\"Expanded input shape to: {inputs.shape}\")\n",
                "    # Method 2: If input batch is larger, use only one SNR value\n",
                "    elif len(test_snr_values) == 1:\n",
                "        test_snr_values = test_snr_values.repeat(inputs.shape[0])\n",
                "        print(f\"Expanded SNR values to: {test_snr_values.shape}\")\n",
                "    # Method 3: Use a single SNR value for all inputs\n",
                "    else:\n",
                "        # Choose a single SNR value if batch sizes don't match and can't be easily fixed\n",
                "        single_snr = test_snr_values[0:1]\n",
                "        test_snr_values = single_snr.repeat(inputs.shape[0])\n",
                "        print(f\"Using a single SNR value {single_snr.item()} expanded to shape: {test_snr_values.shape}\")\n",
                "\n",
                "# Store results for each model\n",
                "snr_results = {}\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    try:\n",
                "        # Evaluate with custom SNR values\n",
                "        results = evaluate_with_snr(\n",
                "            model=model,\n",
                "            data=inputs,\n",
                "            snr_values=test_snr_values,\n",
                "            metrics=[\"entropy\"],\n",
                "            device=device\n",
                "        )\n",
                "        \n",
                "        # Store results\n",
                "        snr_results[model_name] = results\n",
                "    except Exception as e:\n",
                "        print(f\"Error evaluating model {model_name} with custom SNR: {str(e)}\")\n",
                "\n",
                "# Visualize results with custom SNR values if we have any successful evaluations\n",
                "if snr_results:\n",
                "    # Choose up to two models to display\n",
                "    display_models = list(snr_results.keys())[:min(2, len(snr_results))]\n",
                "    fig = plot_multi_snr_comparison(snr_results, model_names=display_models, snr_values=test_snr_values.cpu().numpy())\n",
                "    fig.show()\n",
                "else:\n",
                "    print(\"No successful evaluations to visualize.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Evaluation With Custom Flow Speeds\n",
                "\n",
                "Evaluate models using custom flow speeds to analyze the impact of different flow values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Evaluate models with custom flow speeds\n",
                "num_flows = 4\n",
                "test_flow_speeds = torch.linspace(0.1, 0.9, num_flows).to(device)\n",
                "print(f\"Custom flow speeds: {test_flow_speeds.cpu().numpy()}\")\n",
                "\n",
                "# Create data loader with batch_size=num_flows to match the number of flow speeds\n",
                "data_loader = create_data_loader(\n",
                "    dataset_name=\"high_snr_fixed\",\n",
                "    batch_size=num_flows,  # Make batch size match number of flow values\n",
                "    num_samples=512,\n",
                "    device=device,\n",
                "    fixed_data=True\n",
                ")\n",
                "\n",
                "# Get the first batch of data\n",
                "batch = next(iter(data_loader))\n",
                "inputs = batch[0]  # Data is the first element in the tuple\n",
                "\n",
                "# Verify input shape matches flow speeds\n",
                "print(f\"Input shape: {inputs.shape}, Flow speeds shape: {test_flow_speeds.shape}\")\n",
                "if inputs.shape[0] != len(test_flow_speeds):\n",
                "    print(\"Warning: Batch size mismatch between inputs and flow speeds.\")\n",
                "    # Method 1: Repeat the input to match flow speeds size\n",
                "    if inputs.shape[0] == 1:\n",
                "        inputs = inputs.repeat(len(test_flow_speeds), 1, 1)\n",
                "        print(f\"Expanded input shape to: {inputs.shape}\")\n",
                "    # Method 2: If input batch is larger, use only one flow speed\n",
                "    elif len(test_flow_speeds) == 1:\n",
                "        test_flow_speeds = test_flow_speeds.repeat(inputs.shape[0])\n",
                "        print(f\"Expanded flow speeds to: {test_flow_speeds.shape}\")\n",
                "    # Method 3: Use a single flow speed for all inputs\n",
                "    else:\n",
                "        # Choose a single flow speed if batch sizes don't match and can't be easily fixed\n",
                "        single_flow = test_flow_speeds[0:1]\n",
                "        test_flow_speeds = single_flow.repeat(inputs.shape[0])\n",
                "        print(f\"Using a single flow speed {single_flow.item()} expanded to shape: {test_flow_speeds.shape}\")\n",
                "\n",
                "# Store results for each model\n",
                "flow_results = {}\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    try:\n",
                "        # Evaluate with custom flow speeds\n",
                "        results = evaluate_with_flow(\n",
                "            model=model,\n",
                "            data=inputs,\n",
                "            flow_speeds=test_flow_speeds,\n",
                "            metrics=[\"entropy\"],\n",
                "            device=device\n",
                "        )\n",
                "        \n",
                "        # Store results\n",
                "        flow_results[model_name] = results\n",
                "        \n",
                "        # Convert to DataFrame for analysis\n",
                "        df = results_to_dataframe(results)\n",
                "        \n",
                "        # Plot entropy vs flow speed\n",
                "        fig = plot_metrics_vs_flow(df)\n",
                "        fig.update_layout(title=f\"Entropy vs Flow Speed - {model_name}\")\n",
                "        fig.show()\n",
                "    except Exception as e:\n",
                "        print(f\"Error evaluating model {model_name} with custom flow speeds: {str(e)}\")\n",
                "        \n",
                "if not flow_results:\n",
                "    print(\"No successful evaluations to visualize.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## KMeans Evaluation\n",
                "\n",
                "Evaluate models using KMeans clustering to compare input and output distributions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Evaluate models with KMeans\n",
                "model_kmeans_results = {}\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    # Skip some models to save time\n",
                "    if model_name not in [\"16 Layers\", \"64 Layers\"]:\n",
                "        continue\n",
                "        \n",
                "    # Evaluate with KMeans\n",
                "    results = evaluate_with_kmeans(\n",
                "        model=model,\n",
                "        data=inputs,\n",
                "        snr=test_snr_values,\n",
                "        run_on_predictions=True,\n",
                "        run_on_inputs=True,\n",
                "        metrics=[\"entropy\", \"log_wasserstein\", \"log_kmeans_wasserstein\"],\n",
                "        device=device\n",
                "    )\n",
                "    \n",
                "    # Store results\n",
                "    model_kmeans_results[model_name] = results\n",
                "    \n",
                "    # Convert to DataFrame for analysis\n",
                "    df = results_to_dataframe(results)\n",
                "    \n",
                "    # Plot metrics vs SNR\n",
                "    fig = plot_metrics_vs_snr(df)\n",
                "    fig.update_layout(title=f\"Metrics vs SNR - {model_name}\")\n",
                "    fig.show()\n",
                "    \n",
                "    # Plot entropy vs Wasserstein distance\n",
                "    fig = plot_entropy_vs_wasserstein(df)\n",
                "    fig.update_layout(title=f\"Entropy vs Wasserstein Distance - {model_name}\")\n",
                "    fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Save and Load Results\n",
                "\n",
                "Demonstrate saving results to disk and loading them back."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Save results to disk\n",
                "cache_dir = Path(\"./cache\")\n",
                "cache_dir.mkdir(exist_ok=True)\n",
                "\n",
                "# Save example result\n",
                "model_name = list(model_kmeans_results.keys())[0]\n",
                "result_path = cache_dir / f\"{model_name.replace(' ', '_')}_kmeans.npz\"\n",
                "save_results(model_kmeans_results[model_name], result_path)\n",
                "print(f\"Saved results to {result_path}\")\n",
                "\n",
                "# Load results back\n",
                "loaded_results = load_results(result_path)\n",
                "print(\"Loaded results successfully.\")\n",
                "\n",
                "# Verify loaded results match original\n",
                "loaded_df = results_to_dataframe(loaded_results)\n",
                "original_df = results_to_dataframe(model_kmeans_results[model_name])\n",
                "\n",
                "print(\"\\nLoaded DataFrame:\")\n",
                "print(loaded_df.head())\n",
                "\n",
                "print(\"\\nOriginal DataFrame:\")\n",
                "print(original_df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Compare Multiple Models\n",
                "\n",
                "Create metrics tables to compare all models across different metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Create metrics tables for comparison\n",
                "metrics_data = []\n",
                "\n",
                "for model_name, results in model_results.items():\n",
                "    # Convert results to DataFrame\n",
                "    df = results_to_dataframe(results)\n",
                "    \n",
                "    # Aggregate metrics\n",
                "    metrics_dict = {\n",
                "        \"model_name\": model_name,\n",
                "        \"dataset\": \"high_snr_fixed\",\n",
                "        \"num_samples\": df.shape[0]\n",
                "    }\n",
                "    \n",
                "    # Add mean and std for each metric\n",
                "    for metric in [\"entropy\", \"log_wasserstein\", \"log_kmeans_wasserstein\"]:\n",
                "        if metric in df.columns:\n",
                "            metrics_dict[f\"{metric}_mean\"] = df[metric].mean()\n",
                "            metrics_dict[f\"{metric}_std\"] = df[metric].std()\n",
                "    \n",
                "    metrics_data.append(metrics_dict)\n",
                "\n",
                "# Create DataFrame for metrics\n",
                "metrics_df = pd.DataFrame(metrics_data)\n",
                "\n",
                "# Create tables\n",
                "tables = create_metrics_tables(metrics_df)\n",
                "\n",
                "# Display tables\n",
                "for i, table in enumerate(tables):\n",
                "    table.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Using Cached Results\n",
                "\n",
                "Demonstrate how to use the caching functionality to avoid expensive recomputations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Demonstrate caching functionality\n",
                "cached_results = {}\n",
                "cache_dir = Path(\"./evaluation_cache\")\n",
                "cache_dir.mkdir(exist_ok=True)\n",
                "\n",
                "# Choose a model for demonstration\n",
                "model_name = list(models.keys())[0] if models else \"16 Layers\"\n",
                "model = models.get(model_name)\n",
                "\n",
                "if model is not None:\n",
                "    print(f\"Demonstrating caching with model: {model_name}\")\n",
                "    \n",
                "    # First call should compute and cache the results\n",
                "    print(\"\\nFirst call (should compute and cache):\")\n",
                "    results1 = get_cached_results(\n",
                "        model=model,\n",
                "        data_id=\"high_snr_fixed\",  # Dataset name\n",
                "        use_data_snr=True,         # Use SNR from the dataset\n",
                "        cache_dir=str(cache_dir),\n",
                "        force_recompute=False      # Don't force recomputation\n",
                "    )\n",
                "    \n",
                "    # Convert to dataframe for display\n",
                "    if results1 is not None:\n",
                "        df1 = results_to_dataframe(results1)\n",
                "        print(f\"Results shape: {df1.shape}\")\n",
                "        print(\"First few rows:\")\n",
                "        print(df1.head())\n",
                "        \n",
                "        # Second call should load from cache\n",
                "        print(\"\\nSecond call (should load from cache):\")\n",
                "        results2 = get_cached_results(\n",
                "            model=model,\n",
                "            data_id=\"high_snr_fixed\",\n",
                "            use_data_snr=True,\n",
                "            cache_dir=str(cache_dir),\n",
                "            force_recompute=False\n",
                "        )\n",
                "        \n",
                "        # Verify the results match\n",
                "        df2 = results_to_dataframe(results2)\n",
                "        print(f\"Results shape: {df2.shape}\")\n",
                "        print(\"Results are identical:\", df1.equals(df2))\n",
                "        \n",
                "        # Force recomputation\n",
                "        print(\"\\nThird call with force_recompute=True:\")\n",
                "        results3 = get_cached_results(\n",
                "            model=model,\n",
                "            data_id=\"high_snr_fixed\",\n",
                "            use_data_snr=True,\n",
                "            cache_dir=str(cache_dir),\n",
                "            force_recompute=True  # Force recomputation\n",
                "        )\n",
                "        \n",
                "        df3 = results_to_dataframe(results3)\n",
                "        print(f\"Results shape: {df3.shape}\")\n",
                "        print(\"Results are similar (may not be identical due to randomness):\", df3.shape == df1.shape)\n",
                "else:\n",
                "    print(\"No models available for cache demonstration\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "This notebook has demonstrated the refactored evaluation API for GMM models. The key improvements include:\n",
                "\n",
                "1. **Standardized data formats** with clear documentation\n",
                "2. **Unified evaluation functions** for different scenarios\n",
                "3. **Metrics computation API** that works with both single and multi-batch results\n",
                "4. **IO and caching API** for saving and loading results\n",
                "5. **Data processing API** for converting results to DataFrames\n",
                "6. **Visualization functions** for different types of plots\n",
                "\n",
                "These improvements make the evaluation code more modular, maintainable, and easier to use."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Read the existing notebook to preserve its content
try:
    with open('/mount/Storage/gmm-v4/scripts/evaluation/refactored_evaluation.ipynb', 'r') as f:
        notebook = json.load(f)
except:
    print("Error reading notebook file. Using default template.")
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Find cells that need to be updated
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        # Look for code that accesses batch["data"]
        for i, line in enumerate(cell['source']):
            if 'batch = next(iter(data_loader))' in line:
                cell_index = notebook['cells'].index(cell)
                
                # Find the next line that accesses batch["data"]
                for j, next_line in enumerate(cell['source'][i+1:], i+1):
                    if 'batch["data"]' in next_line:
                        # Replace with tuple unpacking
                        cell['source'][j] = next_line.replace('batch["data"]', 'batch[0]')  # Assuming input is the first element
                        
                        # Also check if batch["targets"] is used and replace if found
                        for k, target_line in enumerate(cell['source']):
                            if 'batch["targets"]' in target_line or 'batch.get("targets"' in target_line:
                                cell['source'][k] = target_line.replace('batch["targets"]', 'batch[1]').replace('batch.get("targets"', 'batch[1] if len(batch) > 1 else None')

# Fix specifically the model evaluation with custom SNR values section
custom_snr_code_updated = [
    "# Evaluate models with custom SNR values\n",
    "num_snrs = 4\n",
    "test_snr_values = torch.linspace(3, 15, num_snrs).to(device)\n",
    "print(f\"Custom SNR values: {test_snr_values.cpu().numpy()}\")\n",
    "\n",
    "# Create data loader with batch_size=num_snrs to match the number of SNR values\n",
    "data_loader = create_data_loader(\n",
    "    dataset_name=\"high_snr_fixed\",\n",
    "    batch_size=num_snrs,  # Make batch size match number of SNR values\n",
    "    num_samples=512,\n",
    "    device=device,\n",
    "    fixed_data=True\n",
    ")\n",
    "\n",
    "# Get the first batch of data\n",
    "batch = next(iter(data_loader))\n",
    "inputs = batch[0]  # Data is the first element in the tuple\n",
    "targets = batch[1] if len(batch) > 1 else None  # Targets is the second element if available\n",
    "\n",
    "# Verify input shape matches SNR values\n",
    "print(f\"Input shape: {inputs.shape}, SNR values shape: {test_snr_values.shape}\")\n",
    "if inputs.shape[0] != len(test_snr_values):\n",
    "    print(\"Warning: Batch size mismatch between inputs and SNR values.\")\n",
    "    # Method 1: Repeat the input to match SNR values size\n",
    "    if inputs.shape[0] == 1:\n",
    "        inputs = inputs.repeat(len(test_snr_values), 1, 1)\n",
    "        print(f\"Expanded input shape to: {inputs.shape}\")\n",
    "    # Method 2: If input batch is larger, use only one SNR value\n",
    "    elif len(test_snr_values) == 1:\n",
    "        test_snr_values = test_snr_values.repeat(inputs.shape[0])\n",
    "        print(f\"Expanded SNR values to: {test_snr_values.shape}\")\n",
    "    # Method 3: Use a single SNR value for all inputs\n",
    "    else:\n",
    "        # Choose a single SNR value if batch sizes don't match and can't be easily fixed\n",
    "        single_snr = test_snr_values[0:1]\n",
    "        test_snr_values = single_snr.repeat(inputs.shape[0])\n",
    "        print(f\"Using a single SNR value {single_snr.item()} expanded to shape: {test_snr_values.shape}\")\n",
    "\n",
    "# Store results for each model\n",
    "snr_results = {}\n",
    "\n",
    "for model_name, model in models.items():\n",
    "    try:\n",
    "        # Evaluate with custom SNR values\n",
    "        results = evaluate_with_snr(\n",
    "            model=model,\n",
    "            data=inputs,\n",
    "            snr_values=test_snr_values,\n",
    "            metrics=[\"entropy\"],\n",
    "            device=device\n",
    "        )\n",
    "        \n",
    "        # Store results\n",
    "        snr_results[model_name] = results\n",
    "    except Exception as e:\n",
    "        print(f\"Error evaluating model {model_name} with custom SNR: {str(e)}\")\n",
    "\n",
    "# Visualize results with custom SNR values if we have any successful evaluations\n",
    "if snr_results:\n",
    "    # Choose up to two models to display\n",
    "    display_models = list(snr_results.keys())[:min(2, len(snr_results))]\n",
    "    fig = plot_multi_snr_comparison(snr_results, model_names=display_models, snr_values=test_snr_values.cpu().numpy())\n",
    "    fig.show()\n",
    "else:\n",
    "    print(\"No successful evaluations to visualize.\")"
]

# Fix also the flow speeds evaluation section
flow_speeds_code_updated = [
    "# Evaluate models with custom flow speeds\n",
    "num_flows = 4\n",
    "test_flow_speeds = torch.linspace(0.1, 0.9, num_flows).to(device)\n",
    "print(f\"Custom flow speeds: {test_flow_speeds.cpu().numpy()}\")\n",
    "\n",
    "# Create data loader with batch_size=num_flows to match the number of flow speeds\n",
    "data_loader = create_data_loader(\n",
    "    dataset_name=\"high_snr_fixed\",\n",
    "    batch_size=num_flows,  # Make batch size match number of flow values\n",
    "    num_samples=512,\n",
    "    device=device,\n",
    "    fixed_data=True\n",
    ")\n",
    "\n",
    "# Get the first batch of data\n",
    "batch = next(iter(data_loader))\n",
    "inputs = batch[0]  # Data is the first element in the tuple\n",
    "\n",
    "# Verify input shape matches flow speeds\n",
    "print(f\"Input shape: {inputs.shape}, Flow speeds shape: {test_flow_speeds.shape}\")\n",
    "if inputs.shape[0] != len(test_flow_speeds):\n",
    "    print(\"Warning: Batch size mismatch between inputs and flow speeds.\")\n",
    "    # Method 1: Repeat the input to match flow speeds size\n",
    "    if inputs.shape[0] == 1:\n",
    "        inputs = inputs.repeat(len(test_flow_speeds), 1, 1)\n",
    "        print(f\"Expanded input shape to: {inputs.shape}\")\n",
    "    # Method 2: If input batch is larger, use only one flow speed\n",
    "    elif len(test_flow_speeds) == 1:\n",
    "        test_flow_speeds = test_flow_speeds.repeat(inputs.shape[0])\n",
    "        print(f\"Expanded flow speeds to: {test_flow_speeds.shape}\")\n",
    "    # Method 3: Use a single flow speed for all inputs\n",
    "    else:\n",
    "        # Choose a single flow speed if batch sizes don't match and can't be easily fixed\n",
    "        single_flow = test_flow_speeds[0:1]\n",
    "        test_flow_speeds = single_flow.repeat(inputs.shape[0])\n",
    "        print(f\"Using a single flow speed {single_flow.item()} expanded to shape: {test_flow_speeds.shape}\")\n",
    "\n",
    "# Store results for each model\n",
    "flow_results = {}\n",
    "\n",
    "for model_name, model in models.items():\n",
    "    try:\n",
    "        # Evaluate with custom flow speeds\n",
    "        results = evaluate_with_flow(\n",
    "            model=model,\n",
    "            data=inputs,\n",
    "            flow_speeds=test_flow_speeds,\n",
    "            metrics=[\"entropy\"],\n",
    "            device=device\n",
    "        )\n",
    "        \n",
    "        # Store results\n",
    "        flow_results[model_name] = results\n",
    "        \n",
    "        # Convert to DataFrame for analysis\n",
    "        df = results_to_dataframe(results)\n",
    "        \n",
    "        # Plot entropy vs flow speed\n",
    "        fig = plot_metrics_vs_flow(df)\n",
    "        fig.update_layout(title=f\"Entropy vs Flow Speed - {model_name}\")\n",
    "        fig.show()\n",
    "    except Exception as e:\n",
    "        print(f\"Error evaluating model {model_name} with custom flow speeds: {str(e)}\")\n",
    "        \n",
    "if not flow_results:\n",
    "    # Evaluate with custom flow speeds\n",
    "    results = evaluate_with_flow(\n",
    "        model=model,\n",
    "        data=inputs,\n",
    "        flow_speeds=test_flow_speeds,\n",
    "        metrics=[\"entropy\"],\n",
    "        device=device\n",
    "    )\n",
    "    \n",
    "    # Store results\n",
    "    flow_results[model_name] = results\n",
    "    \n",
    "    # Convert to DataFrame for analysis\n",
    "    df = results_to_dataframe(results)\n",
    "    \n",
    "    # Plot entropy vs flow speed\n",
    "    fig = plot_metrics_vs_flow(df)\n",
    "    fig.update_layout(title=f\"Entropy vs Flow Speed - {model_name}\")\n",
    "    fig.show()"
]

# Find and replace the entire SNR and flow speeds sections
snr_section_index = -1
flow_section_index = -1

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        title = ''.join(cell['source'])
        if "Model Evaluation With Custom SNR Values" in title:
            snr_section_index = i
        elif "Model Evaluation With Custom Flow Speeds" in title:
            flow_section_index = i

# Update the SNR code section if found
if snr_section_index >= 0 and snr_section_index + 1 < len(notebook['cells']):
    code_cell = notebook['cells'][snr_section_index + 1]
    if code_cell['cell_type'] == 'code':
        code_cell['source'] = custom_snr_code_updated

# Update the flow code section if found
if flow_section_index >= 0 and flow_section_index + 1 < len(notebook['cells']):
    code_cell = notebook['cells'][flow_section_index + 1]
    if code_cell['cell_type'] == 'code':
        code_cell['source'] = flow_speeds_code_updated

# Also fix the evaluate function if it assumes dictionary batch
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        found = False
        for j, line in enumerate(cell['source']):
            if "evaluate(" in line and "batch['data']" in line:
                found = True
                line_num = j
                
        if found:
            # Replace the problematic lines
            for j, line in enumerate(cell['source']):
                if "batch['data']" in line:
                    cell['source'][j] = line.replace("batch['data']", "batch[0]")
                if "targets = batch.get('targets', {})" in line:
                    cell['source'][j] = line.replace("targets = batch.get('targets', {})", "targets = batch[1] if len(batch) > 1 else {}")
                if "batch.get('targets'" in line:
                    cell['source'][j] = line.replace("batch.get('targets'", "batch[1] if len(batch) > 1 else None")

# Also fix the eval_utils module if it assumes dictionary for DataLoader
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and "from scripts.evaluation.src.eval_utils import (" in ''.join(cell['source']):
        # Add a note about data loader format
        cell['source'].insert(4, "\n# Note: Our data loader returns tuples of (input, target) not dictionaries\n")

# Write the updated notebook
with open('/mount/Storage/gmm-v4/scripts/evaluation/refactored_evaluation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 