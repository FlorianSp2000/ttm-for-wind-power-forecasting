"""Channel Attention Analysis v2 for TinyTimeMixer Models

Complete analysis toolkit including cross-channel attention maps matching the original TTM paper.
This module provides tools to extract and visualize gated attention weights from TinyTimeMixer models,
specifically focusing on channel-level attention patterns in wind power forecasting applications.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset


class AttentionWeightExtractor:
    """Extract and store attention weights from TinyTimeMixer gated attention layers."""
    
    def __init__(self):
        self.attention_weights = defaultdict(list)
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks to capture attention weights.
        
        :param model: TinyTimeMixer model to hook into
        """
        
        # Hook for decoder channel feature mixer (most relevant for channel analysis)
        def hook_channel_mixer(module, input, output):
            # The gated attention computes: softmax(linear(x)) * x
            # We want the softmax weights before multiplication
            if hasattr(module, 'attn_softmax') and len(input) > 0:
                x = input[0]  # Input to the gating block
                attn_weights = module.attn_layer(x)  # Linear transformation
                attn_probs = module.attn_softmax(attn_weights)  # Softmax weights
                self.attention_weights['channel_mixer'].append(attn_probs.detach().cpu())
        
        # Register hook on decoder channel feature mixer gating blocks
        for layer_idx, mixer_layer in enumerate(model.decoder.decoder_block.mixers):
            if hasattr(mixer_layer, 'channel_feature_mixer'):
                gating_block = mixer_layer.channel_feature_mixer.gating_block
                hook = gating_block.register_forward_hook(hook_channel_mixer)
                self.hooks.append(hook)
        
        # Optional: Hook for head's forecast channel mixer
        def hook_forecast_mixer(module, input, output):
            if hasattr(module, 'attn_softmax') and len(input) > 0:
                x = input[0]
                attn_weights = module.attn_layer(x)
                attn_probs = module.attn_softmax(attn_weights)
                self.attention_weights['forecast_mixer'].append(attn_probs.detach().cpu())
        
        if hasattr(model.head, 'fcm_block') and hasattr(model.head.fcm_block, 'fcm_gating_block'):
            hook = model.head.fcm_block.fcm_gating_block.register_forward_hook(hook_forecast_mixer)
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_weights(self):
        """Clear stored attention weights."""
        self.attention_weights = defaultdict(list)


def plot_channel_attention_weights(attention_weights, channel_names=None, figsize=(12, 6)):
    """Plot channel-level attention weights.
    
    :param attention_weights: Dictionary containing captured attention weights
    :param channel_names: List of channel names for labeling
    :param figsize: Figure size tuple
    :returns: Figure and attention matrix tensor
    """
    
    if 'channel_mixer' not in attention_weights or len(attention_weights['channel_mixer']) == 0:
        return None, None
    
    # Aggregate attention weights across batches and layers
    all_weights = []
    
    for i, weights in enumerate(attention_weights['channel_mixer']):
        # Handle different tensor shapes
        if len(weights.shape) == 4:
            # Shape: [batch_size, hidden_dim, patches, num_channels]
            # Average over hidden_dim and patches dimensions
            weights = weights.mean(dim=(1, 2))  # Result: [batch_size, num_channels]
            # Process each sample in the batch
            for j in range(weights.shape[0]):
                all_weights.append(weights[j])
        elif len(weights.shape) == 3:
            # Shape: [batch_size, seq_len, num_channels]
            # Average over batch_size and seq_len
            batch_avg = weights.mean(dim=(0, 1))  # Result: [num_channels]
            all_weights.append(batch_avg)
        elif len(weights.shape) == 2:
            # Shape: [batch_size, num_channels]
            # Average over batch_size
            batch_avg = weights.mean(dim=0)  # Result: [num_channels]
            all_weights.append(batch_avg)
        else:
            continue
    
    if not all_weights:
        return None, None
    
    # Stack and compute statistics
    attention_matrix = torch.stack(all_weights)  # [num_samples, num_channels]
    
    # Create channel names if not provided
    if channel_names is None:
        channel_names = [f'Channel_{i}' for i in range(attention_matrix.shape[1])]
    elif len(channel_names) != attention_matrix.shape[1]:
        channel_names = channel_names[:attention_matrix.shape[1]]  # Truncate if needed
    
    # Plot - only 2 subplots for better visibility
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Channel Attention Weights Analysis', fontsize=16)
    
    # 1. Attention distribution (box plots)
    ax1 = axes[0]
    ax1.boxplot([attention_matrix[:, i].numpy() for i in range(attention_matrix.shape[1])],
                labels=channel_names)
    ax1.set_title('Attention Weight Distributions')
    ax1.set_xlabel('Channels')
    ax1.set_ylabel('Attention Weight')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Top attended channels
    ax2 = axes[1]
    mean_attention = attention_matrix.mean(dim=0)
    top_channels = torch.argsort(mean_attention, descending=True)[:10]
    top_names = [channel_names[i] for i in top_channels]
    top_values = mean_attention[top_channels]
    
    bars = ax2.barh(range(len(top_names)), top_values.numpy())
    ax2.set_title('Top 10 Attended Channels')
    ax2.set_xlabel('Mean Attention Weight')
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names)
    ax2.invert_yaxis()
    
    # Color bars
    norm_values = top_values / top_values.max()
    for bar, norm_val in zip(bars, norm_values):
        bar.set_color(plt.cm.viridis(norm_val))
    
    plt.tight_layout()
    return fig, attention_matrix


def create_cross_channel_attention_map(attention_matrix, channel_names, target_channels=['wind_power_offshore', 'wind_power_onshore']):
    """Create cross-channel attention map like in the original TTM paper.
    
    Shows relative importance: how much exogenous channels are attended when target attention is high.
    
    :param attention_matrix: Tensor of shape [num_samples, num_channels]
    :param channel_names: List of all channel names
    :param target_channels: List of target channel names
    :returns: Figure showing cross-channel attention heatmap
    """
    
    # Separate target and exogenous channel indices
    target_indices = [i for i, name in enumerate(channel_names) if name in target_channels]
    exogenous_indices = [i for i, name in enumerate(channel_names) if name not in target_channels]
    exogenous_names = [channel_names[i] for i in exogenous_indices]
    
    # Extract attention weights for target and exogenous channels
    target_attention = attention_matrix[:, target_indices]  # [samples, n_targets]
    exogenous_attention = attention_matrix[:, exogenous_indices]  # [samples, n_exogenous]
    mean_exogenous_attention = exogenous_attention.mean(dim=0)  # [n_exogenous]
    
    # Relative importance: how much exogenous channels are attended when target attention is high
    normalized_cross_attention = torch.zeros(len(target_indices), len(exogenous_indices))
    
    for i, target_idx in enumerate(target_indices):
        # Find samples where target attention is above median
        target_weights = attention_matrix[:, target_idx]
        high_target_mask = target_weights > target_weights.median()
        
        # Calculate mean exogenous attention when target attention is high
        if high_target_mask.sum() > 0:
            high_target_exog_attention = exogenous_attention[high_target_mask].mean(dim=0)
            # Normalize by overall exogenous attention to get relative importance
            relative_attention = high_target_exog_attention / (mean_exogenous_attention + 1e-8)
            normalized_cross_attention[i] = relative_attention
    
    # Create visualization - single plot taking full width
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # Relative importance plot
    im = ax.imshow(normalized_cross_attention.numpy(), cmap='Greens', aspect='auto', vmin=0)
    ax.set_xticks(range(len(exogenous_names)))
    ax.set_xticklabels(exogenous_names, rotation=45, ha='right')
    ax.set_yticks(range(len(target_channels)))
    ax.set_yticklabels([channel_names[i] for i in target_indices])
    ax.set_title('Cross-Channel Attention (Relative Importance)\nExogenous attention when target attention is high vs overall', fontsize=14)
    ax.set_xlabel('Exogenous Variables')
    ax.set_ylabel('Target Variables')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Attention Weight')
    
    # Add values as text
    for i in range(len(target_indices)):
        for j in range(len(exogenous_indices)):
            text = ax.text(j, i, f'{normalized_cross_attention[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    
    return fig, normalized_cross_attention


def check_channel_order(eval_dataset, model_preprocessor=None):
    """Helper function to verify the correct channel order in your model.
    
    :param eval_dataset: The evaluation dataset
    :param model_preprocessor: The model's data preprocessor (tsp object)
    :returns: List of channel names in correct order or None
    """
    
    # If you have access to the preprocessor object (tsp), check the column order
    if model_preprocessor is not None:
        # Construct correct channel order
        target_cols = getattr(model_preprocessor, 'target_columns', [])
        obs_cols = getattr(model_preprocessor, 'observable_columns', [])
        correct_order = target_cols + obs_cols
        return correct_order
    else:
        return None


def quick_attention_analysis(model, trainer, eval_dataset, channel_names):
    """Quick and simple attention analysis using trainer - most reliable approach.
    
    :param model: TinyTimeMixer model
    :param trainer: HuggingFace trainer object
    :param eval_dataset: Evaluation dataset
    :param channel_names: List of channel names in correct order
    :returns: Tuple of (attention_weights, attention_matrix) or (None, None)
    """
    
    extractor = AttentionWeightExtractor()
    extractor.register_hooks(model)
    
    # Use a small subset
    subset_dataset = Subset(eval_dataset, range(min(100, len(eval_dataset))))
    
    model.eval()
    predictions = trainer.predict(subset_dataset)
    
    extractor.remove_hooks()
    
    if 'channel_mixer' in extractor.attention_weights and len(extractor.attention_weights['channel_mixer']) > 0:
        fig, attention_matrix = plot_channel_attention_weights(extractor.attention_weights, channel_names)
        if fig is not None:
            plt.show()
        return extractor.attention_weights, attention_matrix
    else:
        return None, None


def quick_attention_analysis_with_cross_channel(model, trainer, eval_dataset, channel_names, target_channels=None):
    """Extended analysis including cross-channel attention like original paper.
    
    :param model: TinyTimeMixer model
    :param trainer: HuggingFace trainer object
    :param eval_dataset: Evaluation dataset
    :param channel_names: List of channel names in correct order
    :param target_channels: List of target channel names (auto-detected if None)
    :returns: Tuple of (attention_weights, attention_matrix, cross_attention_results)
    """
    
    # Run standard attention analysis first
    attention_weights, attention_matrix = quick_attention_analysis(model, trainer, eval_dataset, channel_names)
    
    if attention_weights is None:
        return None, None, None
    
    # Auto-detect target channels if not provided
    if target_channels is None:
        # Assume first channels with 'wind_power' in name are targets
        target_channels = [name for name in channel_names if 'wind_power' in name]
        if not target_channels:
            # Fallback: assume first 2 channels are targets
            target_channels = channel_names[:2]
    
    # Create cross-channel attention map like original paper
    fig_cross, relative_attention_matrix = create_cross_channel_attention_map(
        attention_matrix, channel_names, target_channels
    )
    
    return (attention_weights, attention_matrix, 
            {'relative_attention': relative_attention_matrix, 'figure': fig_cross})


def analyze_attention_patterns(attention_matrix, channel_names, save_path=None):
    """Analyze and save detailed attention pattern statistics.
    
    :param attention_matrix: Tensor of shape [num_samples, num_channels]
    :param channel_names: List of channel names
    :param save_path: Optional path to save results as CSV
    :returns: DataFrame containing analysis results
    """
    
    mean_attention = attention_matrix.mean(dim=0)
    std_attention = attention_matrix.std(dim=0)
    min_attention = attention_matrix.min(dim=0)[0]
    max_attention = attention_matrix.max(dim=0)[0]
    
    # Create analysis results
    results = {
        'channel': channel_names,
        'mean_attention': mean_attention.numpy(),
        'std_attention': std_attention.numpy(),
        'min_attention': min_attention.numpy(),
        'max_attention': max_attention.numpy(),
        'coefficient_of_variation': (std_attention / mean_attention).numpy()
    }
    
    # Create DataFrame for easy analysis
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('mean_attention', ascending=False)
    
    if save_path:
        df_results.to_csv(save_path, index=False)
    
    return df_results


# Example usage and constants
WIND_POWER_CHANNEL_NAMES_CORRECTED = [
    # Target columns come FIRST in the model
    'wind_power_offshore', 'wind_power_onshore',
    
    # Observable columns follow
    'u100', 'v100', 'u10', 'v10', 'msl', 'pressure_hpa', 't2m', 'temperature_c', 'cdir',
    'wind_dir_sin', 'wind_dir_cos', 'blh', 'tcc', 'tp', 'wind_speed_10m', 'wind_speed_100m', 
    'forecast_lead_hours'
]


def main_analysis_pipeline(model, trainer, eval_dataset, tsp=None, include_cross_channel=True):
    """Complete analysis pipeline for channel attention analysis.
    
    :param model: TinyTimeMixer model
    :param trainer: HuggingFace trainer
    :param eval_dataset: Evaluation dataset
    :param tsp: Model preprocessor object (optional)
    :param include_cross_channel: Whether to include cross-channel analysis
    """
    
    # Step 1: Check channel order
    if tsp is not None:
        correct_channel_names = check_channel_order(eval_dataset, tsp)
    else:
        correct_channel_names = WIND_POWER_CHANNEL_NAMES_CORRECTED
    
    # Step 2: Run attention analysis
    if include_cross_channel:
        attention_weights, attention_matrix, cross_results = quick_attention_analysis_with_cross_channel(
            model, trainer, eval_dataset, correct_channel_names
        )
    else:
        attention_weights, attention_matrix = quick_attention_analysis(
            model, trainer, eval_dataset, correct_channel_names
        )
        cross_results = None
    
    if attention_weights is not None:
        # Step 3: Detailed analysis
        results_df = analyze_attention_patterns(
            attention_matrix, correct_channel_names, 
            save_path='channel_attention_results_v2.csv'
        )
        return attention_weights, attention_matrix, results_df, cross_results
    else:
        return None, None, None, None


if __name__ == "__main__":
    # Example usage:
    # Run complete pipeline with cross-channel analysis
    # attention_weights, attention_matrix, results_df, cross_results = main_analysis_pipeline(
    #     model, trainer, eval_dataset, tsp, include_cross_channel=True
    # )
    
    # Or run individual components:
    # correct_order = check_channel_order(eval_dataset, tsp)
    # attention_weights, attention_matrix, cross_results = quick_attention_analysis_with_cross_channel(
    #     model, trainer, eval_dataset, correct_order
    # )
    pass