import csv
import os
import argparse
from collections import defaultdict

def read_csv_files(csv_files):
    """Read all experiment CSV files into a list of dictionaries"""
    data = []
    for file in csv_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numerical fields
                row['gen_tokens'] = int(row['gen_tokens'])
                row['time_s'] = float(row['time_s'])
                row['power_w'] = float(row['power_w'])
                data.append(row)
    return data

def analyze_configurations(data):
    """Analyze data grouped by configuration (block_size, power_limit, config_name)"""
    config_groups = defaultdict(list)
    text_results = defaultdict(dict)
    
    for row in data:
        key = (row['block_size'], row['power_limit'], row['config_name'])
        config_groups[key].append(row)
        example_id = row['example_id']
        text_results[example_id][key] = row['gen_text']

    config_metrics = {}
    for config, entries in config_groups.items():
        total_energy = 0.0
        total_time = 0.0
        total_tokens = 0
        
        for entry in entries:
            total_energy += entry['power_w'] * entry['time_s']
            total_time += entry['time_s']
            total_tokens += entry['gen_tokens']
        
        # New energy efficiency metrics
        joules_per_token = (total_energy / total_tokens) * 1e6  # μJ/token
        tokens_per_sec = total_tokens / total_time
        
        config_metrics[config] = {
            'joules_per_token': joules_per_token,
            'tokens_per_sec': tokens_per_sec,
            'total_energy': total_energy,
            'total_tokens': total_tokens,
            'avg_power': sum(e['power_w'] for e in entries)/len(entries),
            'num_samples': len(entries)
        }
    
    return config_metrics, text_results

def check_text_consistency(text_results):
    """Check if generated texts match across configurations for same example"""
    discrepancies = []
    for example_id, config_texts in text_results.items():
        # Get all unique generated texts for this example
        texts = list(config_texts.values())
        if len(set(texts)) > 1:
            discrepant_configs = [str(c) for c, t in config_texts.items() if t != texts[0]]
            discrepancies.append({
                'example_id': int(example_id),
                'num_configs': len(config_texts),
                'discrepant_configs': discrepant_configs,
                'sample_text': texts[0][:50] + "..." if texts else ""
            })
    return sorted(discrepancies, key=lambda x: x['example_id'])

def generate_report(config_metrics, discrepancies, text_results, baseline_power=250):
    """Generate report with per-token energy metrics"""
    report = []
    
    # Configuration Efficiency Table
    report.append("=== Energy Efficiency Analysis ===")
    report.append("{:<10} {:<8} {:<12} {:<12} {:<10}".format(
        "Config", "Power(W)", "J/token (μJ)", "Tokens/s", "Samples"
    ))
    
    baseline = next((c for c in config_metrics if c[1] == str(baseline_power)), None)
    baseline_metrics = config_metrics[baseline] if baseline else None
    
    for config in sorted(config_metrics.keys(), key=lambda x: int(x[0])):
        metrics = config_metrics[config]
        report.append("{:<10} {:<8} {:<12.1f} {:<12.2f} {:<10}".format(
            config[0], config[1], metrics['joules_per_token'],
            metrics['tokens_per_sec'], metrics['num_samples']
        ))
    
    # Energy Savings Analysis
    if baseline_metrics:
        report.append("\n=== Energy/Tput vs Baseline ===")
        base_jpt = baseline_metrics['joules_per_token']
        base_tps = baseline_metrics['tokens_per_sec']
        
        report.append("{:<20} {:<12} {:<12} {:<12}".format(
            "Configuration", "ΔJ/token", "ΔThroughput", "Power Eff."
        ))
        
        for config, metrics in config_metrics.items():
            if config == baseline: continue
        
            jpt = metrics['joules_per_token']
            tps = metrics['tokens_per_sec']
            
            report.append("{:<20} {:<+12.1f}% {:<12.1f}% {:<12.1f}%".format(
                f"{config[0]}@{config[1]}W",
                ((base_jpt - jpt)/base_jpt)*100,
                (tps/base_tps)*100 - 100,
                (base_jpt/base_tps)/(jpt/tps)*100  # Energy-delay product
            ))
    
    # Text Consistency Check
    report.append("\n=== Generation Consistency ===")
    report.append(f"Affected Examples: {len(discrepancies)}/{len(text_results)}")
    for d in discrepancies[:3]:
        report.append(f"Ex.{d['example_id']}: {len(d['discrepant_configs'])} variants")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Analyze RQ4 experiment results')
    parser.add_argument('input_files', nargs='+', help='CSV files containing experiment results')
    parser.add_argument('-o', '--output', default='analysis_report.txt', help='Output file name')
    args = parser.parse_args()

    # Read and analyze data
    data = read_csv_files(args.input_files)
    config_metrics, text_results = analyze_configurations(data)
    discrepancies = check_text_consistency(text_results)
    
    # Generate and save report
    report = generate_report(config_metrics, discrepancies, text_results)
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Report generated to {args.output}")

if __name__ == '__main__':
    main()