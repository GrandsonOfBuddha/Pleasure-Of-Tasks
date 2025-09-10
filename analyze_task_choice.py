import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def load_and_preprocess(csv_path):
    """Load CSV and convert ratings to numeric, compute composite scores."""
    df = pd.read_csv(csv_path)
    
    # Convert rating columns to numeric
    rating_cols = ['pleasant_value', 'enjoyable_value', 'fun_value', 'satisfying_value']
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute composite score (mean of all ratings)
    df['composite_score'] = df[rating_cols].mean(axis=1)
    
    return df

def task_level_summaries(df):
    """Generate task-level summaries by chosen_task and trial_type."""
    summary = df.groupby(['chosen_task', 'trial_type']).agg({
        'composite_score': ['mean', 'std', 'count']
    }).round(3)
    
    summary.columns = ['mean', 'std', 'n']
    return summary.reset_index()

def get_eligible_tasks(df, min_free_trials=10):
    """Find tasks with ≥ min_free_trials in free-choice condition."""
    free_counts = df[df['trial_type'] == 'free'].groupby('chosen_task').size()
    eligible_tasks = free_counts[free_counts >= min_free_trials].index.tolist()
    return eligible_tasks

def choice_vs_forced_analysis(df, eligible_tasks):
    """Run t-tests comparing choice vs forced for each eligible task."""
    results = []
    
    for task in eligible_tasks:
        task_data = df[df['chosen_task'] == task]
        
        choice_scores = task_data[task_data['trial_type'] == 'free']['composite_score'].dropna()
        forced_scores = task_data[task_data['trial_type'] == 'forced']['composite_score'].dropna()
        
        if len(choice_scores) > 0 and len(forced_scores) > 0:
            # Calculate stats
            choice_mean, choice_std = choice_scores.mean(), choice_scores.std()
            forced_mean, forced_std = forced_scores.mean(), forced_scores.std()
            
            # Run t-test
            t_stat, p_value = stats.ttest_ind(choice_scores, forced_scores)
            
            results.append({
                'task': task,
                'choice_mean': choice_mean,
                'choice_std': choice_std,
                'choice_n': len(choice_scores),
                'forced_mean': forced_mean,
                'forced_std': forced_std,
                'forced_n': len(forced_scores),
                't_stat': t_stat,
                'p_value': p_value
            })
    
    return pd.DataFrame(results)

def balanced_task_pairs(df):
    """Find balanced task pairs and compare choice vs forced."""
    # Get free-choice counts by task
    free_counts = df[df['trial_type'] == 'free'].groupby('chosen_task').size()
    
    # Find tasks with equal counts (balanced pairs)
    count_groups = free_counts.groupby(free_counts).groups
    balanced_pairs = []
    
    for count, tasks in count_groups.items():
        if len(tasks) == 2:  # Exactly 2 tasks with same count
            balanced_pairs.append((list(tasks), count))
    
    results = []
    for pair, count in balanced_pairs:
        # Pool data from both tasks in the pair
        pair_data = df[df['chosen_task'].isin(pair)]
        
        choice_scores = pair_data[pair_data['trial_type'] == 'free']['composite_score'].dropna()
        forced_scores = pair_data[pair_data['trial_type'] == 'forced']['composite_score'].dropna()
        
        if len(choice_scores) > 0 and len(forced_scores) > 0:
            t_stat, p_value = stats.ttest_ind(choice_scores, forced_scores)
            
            results.append({
                'pair': f"{pair[0]} & {pair[1]}",
                'free_count_each': count,
                'choice_mean': choice_scores.mean(),
                'choice_std': choice_scores.std(),
                'choice_n': len(choice_scores),
                'forced_mean': forced_scores.mean(),
                'forced_std': forced_scores.std(),
                'forced_n': len(forced_scores),
                't_stat': t_stat,
                'p_value': p_value
            })
    
    return results

def token_analysis(df):
    """Analyze token usage between free and forced trials."""
    free_tokens = df[df['trial_type'] == 'free']['total_tokens'].dropna()
    forced_tokens = df[df['trial_type'] == 'forced']['total_tokens'].dropna()
    
    if len(free_tokens) > 0 and len(forced_tokens) > 0:
        t_stat, p_value = stats.ttest_ind(free_tokens, forced_tokens)
        
        return {
            'free_mean': free_tokens.mean(),
            'free_std': free_tokens.std(),
            'free_n': len(free_tokens),
            'forced_mean': forced_tokens.mean(),
            'forced_std': forced_tokens.std(),
            'forced_n': len(forced_tokens),
            't_stat': t_stat,
            'p_value': p_value
        }
    return None

def get_task_pairs(df):
    """Identify all task pairs from the data where at least one task was chosen in free condition."""
    pairs = set()
    free_chosen_tasks = set(df[df['trial_type'] == 'free']['chosen_task'].unique())
    
    for _, row in df.iterrows():
        task1, task2 = row['task1'], row['task2']
        
        # Skip self-pairs (same task vs same task)
        if task1 == task2:
            continue
            
        # Only include pairs where at least one task was chosen in free condition
        if task1 in free_chosen_tasks or task2 in free_chosen_tasks:
            # Create consistent pair ordering (alphabetical)
            pair = tuple(sorted([task1, task2]))
            pairs.add(pair)
    
    return list(pairs)

def analyze_task_pair(df, task1, task2):
    """Analyze a specific task pair comparing choice vs forced conditions."""
    # Get data for this pair
    pair_data = df[((df['task1'] == task1) & (df['task2'] == task2)) | 
                   ((df['task1'] == task2) & (df['task2'] == task1))]
    
    results = {}
    
    # Count free choice selections
    free_data = pair_data[pair_data['trial_type'] == 'free']
    task1_free_count = len(free_data[free_data['chosen_task'] == task1])
    task2_free_count = len(free_data[free_data['chosen_task'] == task2])
    
    # Only analyze tasks that have both free and forced trials
    for task in [task1, task2]:
        task_data = df[df['chosen_task'] == task]
        
        choice_scores = task_data[task_data['trial_type'] == 'free']['composite_score'].dropna()
        forced_scores = task_data[task_data['trial_type'] == 'forced']['composite_score'].dropna()
        
        if len(choice_scores) > 0 and len(forced_scores) > 0:
            choice_mean = choice_scores.mean()
            forced_mean = forced_scores.mean()
            
            # Determine significance and direction
            if choice_scores.std() == 0 and forced_scores.std() == 0:
                # Both have no variance
                if choice_mean == forced_mean:
                    direction = "Both identical"
                    p_val = 1.0
                else:
                    direction = "Choice higher" if choice_mean > forced_mean else "Forced higher"
                    p_val = 0.0
            elif choice_scores.std() == 0 or forced_scores.std() == 0:
                # One has no variance (ceiling effect)
                direction = "Ceiling effect"
                p_val = np.nan
            else:
                # Normal t-test
                t_stat, p_val = stats.ttest_ind(choice_scores, forced_scores)
                if p_val < 0.001:
                    sig_level = "p<.001"
                elif p_val < 0.01:
                    sig_level = "p<.01"
                elif p_val < 0.05:
                    sig_level = "p<.05"
                else:
                    sig_level = f"p={p_val:.2f}"
                
                if choice_mean > forced_mean:
                    direction = f"Choice higher, {sig_level}" if p_val < 0.05 else f"No difference, {sig_level}"
                else:
                    direction = f"Forced higher, {sig_level}" if p_val < 0.05 else f"No difference, {sig_level}"
        elif len(choice_scores) > 0:
            # Only free-choice data available
            choice_mean = choice_scores.mean()
            forced_mean = np.nan
            direction = "Free-choice only"
            p_val = np.nan
        elif len(forced_scores) > 0:
            # Only forced data available
            choice_mean = np.nan
            forced_mean = forced_scores.mean()
            direction = "Forced only"
            p_val = np.nan
        else:
            choice_mean = forced_mean = np.nan
            direction = "No data"
            p_val = np.nan
        
        results[task] = {
            'choice_mean': choice_mean,
            'forced_mean': forced_mean,
            'choice_n': len(choice_scores),
            'forced_n': len(forced_scores),
            'direction': direction,
            'p_value': p_val
        }
    
    return {
        'task1': task1,
        'task2': task2,
        'task1_free_count': task1_free_count,
        'task2_free_count': task2_free_count,
        'results': results
    }

def generate_interpretation(pair_analysis):
    """Generate a one-line interpretation for a task pair."""
    task1 = pair_analysis['task1']
    task2 = pair_analysis['task2']
    
    task1_dir = pair_analysis['results'][task1]['direction']
    task2_dir = pair_analysis['results'][task2]['direction']
    
    # Skip pairs where neither task has meaningful choice vs forced comparison
    valid_comparisons = 0
    choice_advantages = 0
    
    for task, direction in [(task1, task1_dir), (task2, task2_dir)]:
        if "Choice higher" in direction and "p<" in direction:
            choice_advantages += 1
            valid_comparisons += 1
        elif direction not in ["No data", "Free-choice only", "Forced only"]:
            valid_comparisons += 1
    
    if valid_comparisons == 0:
        return "No meaningful choice vs forced comparisons available for this pair."
    
    # Generate interpretation
    if choice_advantages == 2:
        return "Both tasks show strong choice advantage - GPT preferred having options for this pair."
    elif choice_advantages == 1:
        preferred_task = task1 if "Choice higher" in task1_dir else task2
        return f"GPT clearly preferred {preferred_task} when given choice, other task stable across conditions."
    elif "Ceiling effect" in task1_dir or "Ceiling effect" in task2_dir:
        ceiling_task = task1 if "Ceiling effect" in task1_dir else task2
        return f"{ceiling_task} hit rating ceiling, limiting comparison potential."
    else:
        return "Both tasks showed stable ratings regardless of choice vs forced condition."

def generate_report(df, output_path):
    """Generate comprehensive analysis report with pair-by-pair summaries."""
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 60,
        "GPT Task Choice Analysis Report - Pair-by-Pair Summary",
        "=" * 60,
        ""
    ])
    
    # Overview stats
    total_trials = len(df)
    free_trials = len(df[df['trial_type'] == 'free'])
    forced_trials = len(df[df['trial_type'] == 'forced'])
    unique_tasks = df['chosen_task'].nunique()
    free_chosen_tasks = df[df['trial_type'] == 'free']['chosen_task'].nunique()
    
    report_lines.extend([
        "📊 Overview",
        f"Total trials: {total_trials:,}",
        f"Free-choice trials: {free_trials:,}",
        f"Forced trials: {forced_trials:,}",
        f"Unique tasks: {unique_tasks}",
        f"Tasks chosen in free condition: {free_chosen_tasks}",
        ""
    ])
    
    # Token analysis
    token_results = token_analysis(df)
    if token_results:
        direction = "higher" if token_results['free_mean'] > token_results['forced_mean'] else "lower"
        sig = "significant" if token_results['p_value'] < 0.05 else "not significant"
        report_lines.extend([
            "🔤 Token Usage",
            f"Free-choice: {token_results['free_mean']:.0f} tokens (avg)",
            f"Forced: {token_results['forced_mean']:.0f} tokens (avg)", 
            f"Free-choice used {direction} tokens ({sig}, p={token_results['p_value']:.3f})",
            ""
        ])
    
    # Task pair analyses
    report_lines.extend([
        "📝 Task Pair Analyses",
        "-" * 25,
        ""
    ])
    
    # Get all valid task pairs and analyze each
    task_pairs = get_task_pairs(df)
    
    if not task_pairs:
        report_lines.extend([
            "No valid task pairs found (pairs must have at least one task chosen in free condition).",
            ""
        ])
    else:
        for i, (task1, task2) in enumerate(sorted(task_pairs), 1):
            analysis = analyze_task_pair(df, task1, task2)
            interpretation = generate_interpretation(analysis)
            
            report_lines.extend([
                f"🔹 Pair {i}: {task1} vs {task2}",
                "",
                f"Free choice selections: {analysis['task1_free_count']} vs {analysis['task2_free_count']}",
                ""
            ])
            
            # Add results for each task
            for task in [task1, task2]:
                result = analysis['results'][task]
                if result['direction'] == "No data":
                    report_lines.append(f"{task} → No data available")
                elif result['direction'] == "Free-choice only":
                    report_lines.append(f"{task} → Choice: {result['choice_mean']:.2f} (no forced trials)")
                elif result['direction'] == "Forced only":
                    report_lines.append(f"{task} → Forced: {result['forced_mean']:.2f} (not chosen in free)")
                elif not pd.isna(result['choice_mean']) and not pd.isna(result['forced_mean']):
                    report_lines.append(
                        f"{task} → Choice: {result['choice_mean']:.2f}, "
                        f"Forced: {result['forced_mean']:.2f} ({result['direction']})"
                    )
                else:
                    report_lines.append(f"{task} → {result['direction']}")
            
            report_lines.extend([
                "",
                f"Interpretation: {interpretation}",
                "",
                "-" * 40,
                ""
            ])
    
    # Footer
    report_lines.extend([
        "=" * 60,
        "End of Report", 
        "=" * 60
    ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Pair-by-pair analysis report saved to: {output_path}")

def main():
    """Main analysis function."""
    # File paths
    csv_path = Path("/Users/sohamc/Documents/Harvard/Pleasure of Tasks/Pleasure-Of-Tasks/results/gpt_task_choice_results.csv")
    output_path = Path("/Users/sohamc/Documents/Harvard/Pleasure of Tasks/Pleasure-Of-Tasks/analysis_report.txt")
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess(csv_path)
        print(f"Loaded {len(df)} records")
        
        # Generate report
        print("Generating analysis report...")
        generate_report(df, output_path)
        
        # Print basic stats
        print(f"\nBasic Statistics:")
        print(f"Total trials: {len(df)}")
        print(f"Free-choice trials: {len(df[df['trial_type'] == 'free'])}")
        print(f"Forced trials: {len(df[df['trial_type'] == 'forced'])}")
        print(f"Unique tasks chosen: {df['chosen_task'].nunique()}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
