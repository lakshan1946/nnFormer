"""
Comparative Analysis Script for Enhanced nnFormer
================================================

This script provides comprehensive comparison between baseline and enhanced
nnFormer models with statistical analysis and visualization.

Author: 210353V
Date: October 2025
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import argparse
from datetime import datetime


class ModelComparator:
    """
    Compare performance between baseline and enhanced nnFormer models.
    """
    
    def __init__(self, baseline_dir, enhanced_dir, output_dir):
        self.baseline_dir = Path(baseline_dir)
        self.enhanced_dir = Path(enhanced_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.baseline_results = self.load_results(self.baseline_dir)
        self.enhanced_results = self.load_results(self.enhanced_dir)
        
        # Metrics of interest
        self.key_metrics = [
            'dice_WT', 'dice_TC', 'dice_ET',
            'hd95_WT', 'hd95_TC', 'hd95_ET',
            'sensitivity_class_1', 'sensitivity_class_2', 'sensitivity_class_3',
            'specificity_class_1', 'specificity_class_2', 'specificity_class_3'
        ]
    
    def load_results(self, results_dir):
        """Load results from experiment directory."""
        metrics_files = list(results_dir.glob("metrics/metrics_*.json"))
        
        if not metrics_files:
            raise FileNotFoundError(f"No metrics files found in {results_dir}/metrics/")
        
        # Use the latest metrics file
        latest_metrics = max(metrics_files, key=os.path.getctime)
        
        with open(latest_metrics, 'r') as f:
            results = json.load(f)
        
        return results
    
    def extract_case_metrics(self, results, metric_name):
        """Extract metric values across all cases."""
        values = []
        
        if 'individual_cases' in results:
            for case_id, metrics in results['individual_cases'].items():
                if metric_name in metrics:
                    values.append(metrics[metric_name])
        
        return np.array(values)
    
    def compute_statistical_comparison(self):
        """Compute statistical comparisons between models."""
        comparisons = {}
        
        for metric in self.key_metrics:
            # Extract values for both models
            baseline_values = self.extract_case_metrics(self.baseline_results, metric)
            enhanced_values = self.extract_case_metrics(self.enhanced_results, metric)
            
            if len(baseline_values) == 0 or len(enhanced_values) == 0:
                continue
            
            # Compute statistics
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            enhanced_mean = np.mean(enhanced_values)
            enhanced_std = np.std(enhanced_values)
            
            # Statistical tests
            # Paired t-test (assuming same cases)
            if len(baseline_values) == len(enhanced_values):
                t_stat, p_value = stats.ttest_rel(baseline_values, enhanced_values)
                test_type = "paired_t_test"
            else:
                t_stat, p_value = stats.ttest_ind(baseline_values, enhanced_values)
                test_type = "independent_t_test"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_std**2 + 
                                (len(enhanced_values) - 1) * enhanced_std**2) / 
                               (len(baseline_values) + len(enhanced_values) - 2))
            cohens_d = (enhanced_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Improvement percentage
            if baseline_mean != 0:
                improvement_pct = ((enhanced_mean - baseline_mean) / abs(baseline_mean)) * 100
            else:
                improvement_pct = 0
            
            comparisons[metric] = {
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(baseline_std),
                'enhanced_mean': float(enhanced_mean),
                'enhanced_std': float(enhanced_std),
                'improvement_pct': float(improvement_pct),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'test_type': test_type,
                'significant': p_value < 0.05,
                'baseline_n': int(len(baseline_values)),
                'enhanced_n': int(len(enhanced_values))
            }
        
        return comparisons
    
    def create_comparison_plots(self, comparisons):
        """Create comprehensive comparison visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Dice Score Comparison
        dice_metrics = [m for m in self.key_metrics if 'dice' in m]
        self.plot_metric_comparison(dice_metrics, comparisons, 
                                   "Dice Similarity Coefficient", 
                                   "dice_comparison.png")
        
        # 2. Hausdorff Distance Comparison  
        hd_metrics = [m for m in self.key_metrics if 'hd95' in m]
        self.plot_metric_comparison(hd_metrics, comparisons,
                                   "95th Percentile Hausdorff Distance (mm)",
                                   "hausdorff_comparison.png")
        
        # 3. Sensitivity/Specificity Comparison
        sens_spec_metrics = [m for m in self.key_metrics if 'sensitivity' in m or 'specificity' in m]
        self.plot_metric_comparison(sens_spec_metrics, comparisons,
                                   "Sensitivity and Specificity",
                                   "sensitivity_specificity_comparison.png")
        
        # 4. Statistical Significance Heatmap
        self.plot_significance_heatmap(comparisons)
        
        # 5. Effect Size Plot
        self.plot_effect_sizes(comparisons)
        
        # 6. Box plots for key metrics
        self.create_box_plots()
    
    def plot_metric_comparison(self, metrics, comparisons, ylabel, filename):
        """Create comparison plot for specific metrics."""
        if not metrics:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        baseline_means = [comparisons[m]['baseline_mean'] for m in metrics if m in comparisons]
        baseline_stds = [comparisons[m]['baseline_std'] for m in metrics if m in comparisons]
        enhanced_means = [comparisons[m]['enhanced_mean'] for m in metrics if m in comparisons]
        enhanced_stds = [comparisons[m]['enhanced_std'] for m in metrics if m in comparisons]
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, baseline_means, width, 
                      yerr=baseline_stds, label='Baseline nnFormer', 
                      alpha=0.8, capsize=5)
        bars2 = ax.bar(x_pos + width/2, enhanced_means, width,
                      yerr=enhanced_stds, label='Enhanced nnFormer',
                      alpha=0.8, capsize=5)
        
        # Add significance markers
        for i, metric in enumerate([m for m in metrics if m in comparisons]):
            if comparisons[metric]['significant']:
                y_max = max(baseline_means[i] + baseline_stds[i], 
                           enhanced_means[i] + enhanced_stds[i])
                ax.text(i, y_max * 1.05, '*', ha='center', va='bottom', 
                       fontsize=16, fontweight='bold', color='red')
        
        # Customize plot
        ax.set_xlabel('Metrics')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} - Model Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics if m in comparisons], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_significance_heatmap(self, comparisons):
        """Create heatmap showing statistical significance and effect sizes."""
        
        metrics = list(comparisons.keys())
        
        # Prepare data for heatmap
        significance_data = []
        effect_size_data = []
        
        for metric in metrics:
            comp = comparisons[metric]
            significance_data.append(1 if comp['significant'] else 0)
            effect_size_data.append(comp['cohens_d'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Significance heatmap
        sig_matrix = np.array(significance_data).reshape(-1, 1)
        sns.heatmap(sig_matrix, 
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   xticklabels=['Statistical Significance (p<0.05)'],
                   annot=True, cmap='RdYlBu_r', ax=ax1, cbar_kws={'label': 'Significant'})
        ax1.set_title('Statistical Significance')
        
        # Effect size heatmap
        effect_matrix = np.array(effect_size_data).reshape(-1, 1)
        sns.heatmap(effect_matrix,
                   yticklabels=[m.replace('_', ' ').title() for m in metrics],
                   xticklabels=['Effect Size (Cohen\'s d)'],
                   annot=True, cmap='RdBu_r', center=0, ax=ax2,
                   cbar_kws={'label': 'Effect Size'})
        ax2.set_title('Effect Size (Cohen\'s d)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_analysis_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_effect_sizes(self, comparisons):
        """Plot effect sizes with confidence intervals."""
        
        metrics = list(comparisons.keys())
        effect_sizes = [comparisons[m]['cohens_d'] for m in metrics]
        improvements = [comparisons[m]['improvement_pct'] for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Effect sizes
        colors = ['red' if es < 0 else 'green' for es in effect_sizes]
        bars1 = ax1.barh(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(metrics)))
        ax1.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.set_xlabel('Effect Size (Cohen\'s d)')
        ax1.set_title('Effect Sizes: Enhanced vs Baseline nnFormer')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add effect size interpretation lines
        ax1.axvline(x=0.2, color='orange', linestyle=':', alpha=0.7, label='Small effect')
        ax1.axvline(x=0.5, color='red', linestyle=':', alpha=0.7, label='Medium effect')
        ax1.axvline(x=0.8, color='darkred', linestyle=':', alpha=0.7, label='Large effect')
        ax1.legend()
        
        # Improvement percentages
        colors2 = ['red' if imp < 0 else 'green' for imp in improvements]
        bars2 = ax2.barh(range(len(metrics)), improvements, color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(metrics)))
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_xlabel('Improvement Percentage (%)')
        ax2.set_title('Percentage Improvement: Enhanced vs Baseline')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_sizes_and_improvements.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_box_plots(self):
        """Create box plots comparing distributions."""
        
        key_dice_metrics = ['dice_WT', 'dice_TC', 'dice_ET']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(key_dice_metrics):
            baseline_values = self.extract_case_metrics(self.baseline_results, metric)
            enhanced_values = self.extract_case_metrics(self.enhanced_results, metric)
            
            if len(baseline_values) > 0 and len(enhanced_values) > 0:
                data = [baseline_values, enhanced_values]
                labels = ['Baseline', 'Enhanced']
                
                bp = axes[i].boxplot(data, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][1].set_facecolor('lightgreen')
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Dice Coefficient')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dice_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_report(self, comparisons):
        """Generate comprehensive comparison report."""
        
        report_path = self.output_dir / 'comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced nnFormer vs Baseline Comparison Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Count significant improvements
            significant_improvements = sum(1 for m, c in comparisons.items() 
                                         if c['significant'] and c['improvement_pct'] > 0)
            total_metrics = len(comparisons)
            
            f.write(f"- **Metrics evaluated:** {total_metrics}\n")
            f.write(f"- **Significant improvements:** {significant_improvements}/{total_metrics} "
                   f"({significant_improvements/total_metrics*100:.1f}%)\n\n")
            
            # Key findings
            f.write("### Key Findings\n\n")
            
            dice_metrics = {k: v for k, v in comparisons.items() if 'dice' in k}
            if dice_metrics:
                avg_dice_improvement = np.mean([c['improvement_pct'] for c in dice_metrics.values()])
                f.write(f"- **Average Dice improvement:** {avg_dice_improvement:.2f}%\n")
            
            hd_metrics = {k: v for k, v in comparisons.items() if 'hd95' in k}
            if hd_metrics:
                avg_hd_improvement = np.mean([c['improvement_pct'] for c in hd_metrics.values()])
                f.write(f"- **Average HD95 improvement:** {avg_hd_improvement:.2f}%\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write("| Metric | Baseline | Enhanced | Improvement (%) | p-value | Effect Size | Significant |\n")
            f.write("|--------|----------|----------|------------------|---------|-------------|-------------|\n")
            
            for metric, comp in comparisons.items():
                significance_mark = "✓" if comp['significant'] else "✗"
                f.write(f"| {metric.replace('_', ' ').title()} | "
                       f"{comp['baseline_mean']:.4f}±{comp['baseline_std']:.4f} | "
                       f"{comp['enhanced_mean']:.4f}±{comp['enhanced_std']:.4f} | "
                       f"{comp['improvement_pct']:+.2f}% | "
                       f"{comp['p_value']:.4f} | "
                       f"{comp['cohens_d']:.3f} | "
                       f"{significance_mark} |\n")
            
            f.write("\n## Statistical Analysis\n\n")
            f.write("### Effect Size Interpretation (Cohen's d)\n")
            f.write("- Small effect: d ≥ 0.2\n")
            f.write("- Medium effect: d ≥ 0.5\n") 
            f.write("- Large effect: d ≥ 0.8\n\n")
            
            # Effect size summary
            large_effects = sum(1 for c in comparisons.values() if abs(c['cohens_d']) >= 0.8)
            medium_effects = sum(1 for c in comparisons.values() 
                               if 0.5 <= abs(c['cohens_d']) < 0.8)
            small_effects = sum(1 for c in comparisons.values() 
                              if 0.2 <= abs(c['cohens_d']) < 0.5)
            
            f.write(f"- **Large effects:** {large_effects} metrics\n")
            f.write(f"- **Medium effects:** {medium_effects} metrics\n")
            f.write(f"- **Small effects:** {small_effects} metrics\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("The following plots have been generated:\n\n")
            f.write("1. `dice_comparison.png` - Dice coefficient comparison\n")
            f.write("2. `hausdorff_comparison.png` - Hausdorff distance comparison\n") 
            f.write("3. `sensitivity_specificity_comparison.png` - Sensitivity/Specificity comparison\n")
            f.write("4. `statistical_analysis_heatmap.png` - Statistical significance heatmap\n")
            f.write("5. `effect_sizes_and_improvements.png` - Effect sizes and improvements\n")
            f.write("6. `dice_boxplots.png` - Distribution comparison via box plots\n\n")
            
        print(f"Comparison report saved to: {report_path}")
        
        return report_path
    
    def run_complete_analysis(self):
        """Run complete comparative analysis."""
        
        print("Starting comparative analysis...")
        
        # Compute statistical comparisons
        print("Computing statistical comparisons...")
        comparisons = self.compute_statistical_comparison()
        
        # Create visualizations
        print("Creating comparison plots...")
        self.create_comparison_plots(comparisons)
        
        # Generate report
        print("Generating comparison report...")
        report_path = self.generate_comparison_report(comparisons)
        
        # Save raw comparison data
        comparison_data_path = self.output_dir / 'comparison_data.json'
        with open(comparison_data_path, 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        print(f"Analysis complete! Results saved to: {self.output_dir}")
        print(f"Main report: {report_path}")
        
        return comparisons


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and enhanced nnFormer models')
    parser.add_argument('--baseline_dir', required=True, 
                       help='Directory containing baseline model results')
    parser.add_argument('--enhanced_dir', required=True,
                       help='Directory containing enhanced model results')
    parser.add_argument('--output_dir', default='./comparison_results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Create comparator and run analysis
    comparator = ModelComparator(args.baseline_dir, args.enhanced_dir, args.output_dir)
    comparisons = comparator.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    significant_improvements = sum(1 for c in comparisons.values() 
                                 if c['significant'] and c['improvement_pct'] > 0)
    total_metrics = len(comparisons)
    
    print(f"Total metrics evaluated: {total_metrics}")
    print(f"Significant improvements: {significant_improvements}/{total_metrics} "
          f"({significant_improvements/total_metrics*100:.1f}%)")
    
    # Show top improvements
    sorted_improvements = sorted(comparisons.items(), 
                               key=lambda x: x[1]['improvement_pct'], reverse=True)
    
    print("\nTop 5 Improvements:")
    for i, (metric, comp) in enumerate(sorted_improvements[:5]):
        significance = "*" if comp['significant'] else ""
        print(f"{i+1}. {metric}: {comp['improvement_pct']:+.2f}% {significance}")


if __name__ == "__main__":
    main()