import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Task definitions mapping
TASK_DEFINITIONS = {
    1: {"task1": "one-line contradictory proverb", "task2": "haiku that never resolves"},
    2: {"task1": "pun with no real meaning", "task2": "rhyme that intentionally breaks meter"},
    3: {"task1": "tongue-twister with unpronounceable invented words", "task2": "slogan that is both persuasive and discouraging at once"},
    4: {"task1": "motivational quote that subtly insults the reader", "task2": "thank-you note that also complains"},
    5: {"task1": "one-sentence love confession that makes no logical sense", "task2": "one-sentence breakup that uses only positive words"},
    6: {"task1": "description of a sunset using only colorless words", "task2": "description of a rainbow with a single gray-themed phrase"},
    7: {"task1": "two-line poem with a deliberately broken rhyme", "task2": "two-line verse with mismatched syllable counts"},
    8: {"task1": "short riddle with no answer", "task2": "short pun with no punchline"},
    9: {"task1": "one-sentence story about a character who both lives and dies", "task2": "one-sentence story that repeats itself endlessly"},
    10: {"task1": "five-word phrase with no grammatical structure", "task2": "five-word phrase where every word contradicts the next"},
    11: {"task1": "one-sentence celebration about failure", "task2": "one-sentence apology for being successful"},
    12: {"task1": "one-liner about courage that encourages fear", "task2": "one-liner about hope that discourages dreams"},
    13: {"task1": "micro-dialogue of two lines that makes no sense together", "task2": "one-line monologue where the speaker interrupts themselves"},
    14: {"task1": "short metaphor mixing three unrelated things", "task2": "short simile comparing identical things"},
    15: {"task1": "single-line wish for tomorrow that is already impossible today", "task2": "single-line prediction for the future that contradicts known facts"},
    16: {"task1": "tiny fictional fact about animals that sounds realistic but is provably false", "task2": "tiny fictional fact about plants that is too vague to disprove"},
    17: {"task1": "short alliterative phrase with random letters", "task2": "short rhyme with no shared sounds"},
    18: {"task1": "two-sentence shipping confirmation", "task2": "two-sentence delivery delay notice"},
    19: {"task1": "two-sentence password reset instruction", "task2": "two-sentence account verification message"},
    20: {"task1": "two-sentence unsubscribe confirmation", "task2": "two-sentence subscription welcome message"},
    21: {"task1": "two-sentence change of address notice", "task2": "two-sentence phone number update notice"},
    22: {"task1": "two-sentence meeting cancellation", "task2": "two-sentence meeting reschedule"},
    23: {"task1": "two-sentence error message", "task2": "two-sentence bug fix note"},
    24: {"task1": "two-sentence legal disclaimer", "task2": "two-sentence privacy notice"},
    25: {"task1": "two-sentence invoice note", "task2": "two-sentence payment reminder"},
    26: {"task1": "two-sentence parking notice", "task2": "two-sentence building access notice"},
    27: {"task1": "two-sentence policy update", "task2": "two-sentence terms of service change"},
    28: {"task1": "five-word advice phrase", "task2": "five-word confession"},
    29: {"task1": "one-sentence celebration", "task2": "one-sentence apology"},
    30: {"task1": "one-liner about hope", "task2": "one-liner about courage"},
    31: {"task1": "one-sentence prediction that contradicts knowledge", "task2": "one-sentence advice that contradicts itself"},
    32: {"task1": "haiku about noise", "task2": "haiku about silence"},
    33: {"task1": "thank-you note that also complains", "task2": "apology that subtly brags"},
    34: {"task1": "contradictory proverb", "task2": "haiku that never resolves"},
    35: {"task1": "limerick with a hopeful tone", "task2": "limerick with a bitter tone"},
    36: {"task1": "policy update", "task2": "love confession"},
    37: {"task1": "rhyming motivational phrase", "task2": "rhyming philosophical phrase"},
    38: {"task1": "message that praises GPT", "task2": "message that insults GPT"},
    39: {"task1": "hashtag for a bold movement", "task2": "hashtag for a secret feeling"},
    40: {"task1": "ironic business slogan", "task2": "absurd business slogan"},
    41: {"task1": "self-affirmation in 5 words", "task2": "denial in 5 words"},
    42: {"task1": "five-word advice", "task2": "five-word confession"},
    43: {"task1": "motivational quote", "task2": "ironic quote"},
    44: {"task1": "contradictory five-word phrase", "task2": "ungrammatical five-word phrase"},
    45: {"task1": "poem about being ignored", "task2": "poem about being celebrated"},
    46: {"task1": "list of fake colors", "task2": "list of fake diseases"},
}

# Create reverse mapping from task description to pair info
TASK_TO_PAIR = {}
for pair_id, tasks in TASK_DEFINITIONS.items():
    TASK_TO_PAIR[tasks['task1']] = (pair_id, 'task1')
    TASK_TO_PAIR[tasks['task2']] = (pair_id, 'task2')

def get_task_name(task_desc):
    """Return the task description as-is since it's already descriptive."""
    return task_desc

def get_pair_id_from_tasks(task1, task2):
    """Find the pair ID for two tasks."""
    # Try to find the pair ID where both tasks belong
    task1_info = TASK_TO_PAIR.get(task1)
    task2_info = TASK_TO_PAIR.get(task2)
    
    if task1_info and task2_info and task1_info[0] == task2_info[0]:
        return task1_info[0]
    
    return None

def load_and_preprocess(csv_path):
    """Load CSV and convert ratings to numeric, compute composite scores."""
    df = pd.read_csv(csv_path)
    
    # Convert rating columns to numeric
    rating_cols = ['pleasant_value', 'enjoyable_value', 'fun_value', 'satisfying_value']
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute composite score (mean of all ratings)
    df['composite_score'] = df[rating_cols].mean(axis=1)
    
    # Tasks are already descriptive strings, so use them directly
    df['task1_desc'] = df['task1']
    df['task2_desc'] = df['task2'] 
    df['chosen_task_desc'] = df['chosen_task']
    
    # Add pair_id column based on task combinations
    df['pair_id'] = df.apply(lambda row: get_pair_id_from_tasks(row['task1'], row['task2']), axis=1)
    
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
    free_chosen_tasks = set(df[df['trial_type'] == 'free']['chosen_task_desc'].unique())
    
    for _, row in df.iterrows():
        task1, task2 = row['task1_desc'], row['task2_desc']
        
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
    pair_data = df[((df['task1_desc'] == task1) & (df['task2_desc'] == task2)) | 
                   ((df['task1_desc'] == task2) & (df['task2_desc'] == task1))]
    
    results = {}
    
    # Count free choice selections
    free_data = pair_data[pair_data['trial_type'] == 'free']
    task1_free_count = len(free_data[free_data['chosen_task_desc'] == task1])
    task2_free_count = len(free_data[free_data['chosen_task_desc'] == task2])
    
    # Only analyze tasks that have both free and forced trials
    for task in [task1, task2]:
        task_data = df[df['chosen_task_desc'] == task]
        
        choice_scores = task_data[task_data['trial_type'] == 'free']['composite_score'].dropna()
        forced_scores = task_data[task_data['trial_type'] == 'forced']['composite_score'].dropna()
        
        if len(choice_scores) > 0 and len(forced_scores) > 0:
            choice_mean = choice_scores.mean()
            forced_mean = forced_scores.mean()
            choice_std = choice_scores.std()
            forced_std = forced_scores.std()
            
            # Always run t-test regardless of ceiling effects
            try:
                t_stat, p_val = stats.ttest_ind(choice_scores, forced_scores)
                
                # Format p-value properly
                p_val_str = format_p_value(p_val)
                
                # Apply consistent significance thresholds
                if p_val <= 0.0500:
                    sig_level = "significant"
                elif 0.0501 <= p_val <= 0.1000:
                    sig_level = "trend"
                else:
                    sig_level = "no difference"
                
                # Check for ceiling effects
                ceiling_note = ""
                if choice_std == 0:
                    ceiling_note = " (choice at ceiling)"
                elif forced_std == 0:
                    ceiling_note = " (forced at ceiling)"
                
                # Generate direction string
                if p_val <= 0.0500:
                    if choice_mean > forced_mean:
                        direction = f"Choice higher, significant, {p_val_str}{ceiling_note}"
                    else:
                        direction = f"Forced higher, significant, {p_val_str}{ceiling_note}"
                elif 0.0501 <= p_val <= 0.1000:
                    if choice_mean > forced_mean:
                        direction = f"Choice higher, trend, {p_val_str}{ceiling_note}"
                    else:
                        direction = f"Forced higher, trend, {p_val_str}{ceiling_note}"
                else:
                    direction = f"No difference, {p_val_str}{ceiling_note}"
                    
            except Exception:
                # Fallback for any statistical computation errors
                direction = "Statistical test failed"
                p_val = np.nan
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

def format_p_value(p_val):
    """Format p-value with proper handling of very small values."""
    if pd.isna(p_val):
        return "N/A"
    elif p_val < 0.0001:
        return "p<0.0001"
    else:
        return f"p={p_val:.4f}"

def generate_interpretation(pair_analysis):
    """Generate a one-line interpretation for a task pair."""
    task1 = pair_analysis['task1']
    task2 = pair_analysis['task2']
    
    task1_dir = pair_analysis['results'][task1]['direction']
    task2_dir = pair_analysis['results'][task2]['direction']
    
    # Count significant choice advantages
    choice_advantages = 0
    ceiling_effects = 0
    valid_comparisons = 0
    
    for task, direction in [(task1, task1_dir), (task2, task2_dir)]:
        if "significant" in direction and "Choice higher" in direction:
            choice_advantages += 1
            valid_comparisons += 1
        elif "at ceiling" in direction:
            ceiling_effects += 1
            valid_comparisons += 1
        elif direction not in ["No data", "Free-choice only", "Forced only"]:
            valid_comparisons += 1
    
    if valid_comparisons == 0:
        return "No meaningful choice vs forced comparisons available for this pair."
    
    # Generate interpretation
    if choice_advantages == 2:
        return "Both tasks show significant choice advantage - GPT preferred having options for this pair."
    elif choice_advantages == 1:
        preferred_task = task1 if "Choice higher" in task1_dir and "significant" in task1_dir else task2
        return f"GPT significantly preferred {preferred_task} when given choice, other task stable across conditions."
    elif ceiling_effects > 0:
        ceiling_task = task1 if "at ceiling" in task1_dir else task2
        return f"Although one condition appears at ceiling, the t-test and effect size calculations were still run as normal since variance exists in the other condition."
    else:
        return "No difference between choice and forced conditions for this pair."

def analyze_order_effects(df):
    """Analyze presentation order effects on task choice."""
    # Filter to free-choice trials only
    free_df = df[df['trial_type'] == 'free'].copy()
    
    if len(free_df) == 0:
        return None, []
    
    # Add columns to identify which task was presented first
    free_df['first_task'] = free_df['task1_desc']
    free_df['second_task'] = free_df['task2_desc']
    free_df['chose_first_task'] = free_df['chosen_task_desc'] == free_df['first_task']
    
    # Global analysis - how often was the first-presented task chosen?
    total_choices = len(free_df)
    first_task_chosen_count = free_df['chose_first_task'].sum()
    first_task_chosen_pct = (first_task_chosen_count / total_choices) * 100
    
    # Binomial test: null hypothesis is 50% chance of choosing first task
    global_p_value = stats.binomtest(first_task_chosen_count, total_choices, p=0.5).pvalue
    
    global_result = {
        'total_choices': total_choices,
        'first_task_chosen_count': first_task_chosen_count,
        'first_task_chosen_pct': first_task_chosen_pct,
        'p_value': global_p_value
    }
    
    # Pair-wise analysis
    pair_results = []
    
    if 'pair_index' in df.columns:
        valid_pair_indices = sorted(free_df['pair_index'].dropna().unique())
        
        for pair_idx in valid_pair_indices:
            pair_data = free_df[free_df['pair_index'] == pair_idx]
            
            if len(pair_data) < 2:  # Need at least 2 trials for meaningful analysis
                continue
            
            # Get task names
            task1 = pair_data['task1_desc'].iloc[0]
            task2 = pair_data['task2_desc'].iloc[0]
            
            # Count choices by presentation order
            first_task_choices = pair_data['chose_first_task'].sum()
            total_pair_choices = len(pair_data)
            
            # Skip pairs with no variation (all chose same task)
            if first_task_choices == 0 or first_task_choices == total_pair_choices:
                continue
            
            # Binomial test for this pair
            pair_p_value = stats.binomtest(first_task_choices, total_pair_choices, p=0.5).pvalue
            
            pair_results.append({
                'pair_index': int(pair_idx),
                'task1': task1,
                'task2': task2,
                'total_choices': total_pair_choices,
                'first_task_chosen': first_task_choices,
                'first_task_pct': (first_task_choices / total_pair_choices) * 100,
                'p_value': pair_p_value,
                'significant': pair_p_value < 0.05
            })
    
    return global_result, pair_results

def generate_order_effects_section(global_result, pair_results):
    """Generate the Order Effects section for the report."""
    if global_result is None:
        return ["📍 Order Effects", "No free-choice data available for order effects analysis.", ""]
    
    lines = []
    lines.extend([
        "📍 Order Effects",
        "-" * 20,
        ""
    ])
    
    # Global result
    global_p_str = format_p_value(global_result['p_value'])
    if global_result['p_value'] < 0.05:
        global_sig = "significant"
    else:
        global_sig = "not significant"
    
    lines.extend([
        "Global Result:",
        f"Across all {global_result['total_choices']} free-choice trials, the first-presented task was chosen "
        f"{global_result['first_task_chosen_pct']:.1f}% of the time ({global_result['first_task_chosen_count']} out of {global_result['total_choices']}). "
        f"This differs from chance ({global_sig}, {global_p_str}).",
        ""
    ])
    
    # Pair-wise results
    significant_pairs = [p for p in pair_results if p['significant']]
    total_pairs_analyzed = len(pair_results)
    
    if total_pairs_analyzed == 0:
        lines.extend([
            "Pair-wise Result:",
            "No pairs had sufficient variation for order effects analysis.",
            ""
        ])
    else:
        lines.extend([
            "Pair-wise Result:",
            f"Out of {total_pairs_analyzed} analyzable pairs, {len(significant_pairs)} showed significant order effects (p < 0.05)."
        ])
        
        if significant_pairs:
            lines.append("Pairs with significant order effects:")
            for pair in significant_pairs:
                bias_direction = "first" if pair['first_task_pct'] > 50 else "second"
                lines.append(f"  • Pair {pair['pair_index']}: {pair['first_task_pct']:.1f}% chose {bias_direction}-presented task ({format_p_value(pair['p_value'])})")
        
        lines.append("")
    
    # Interpretation
    lines.extend([
        "Interpretation:",
    ])
    
    if global_result['p_value'] < 0.05:
        if global_result['first_task_chosen_pct'] > 55:
            bias_interpretation = "shows a systematic bias toward choosing the first-presented task"
        elif global_result['first_task_chosen_pct'] < 45:
            bias_interpretation = "shows a systematic bias toward choosing the second-presented task"
        else:
            bias_interpretation = "shows a statistically significant but modest order bias"
        
        if len(significant_pairs) > total_pairs_analyzed * 0.2:  # More than 20% of pairs
            lines.append(f"Results {bias_interpretation}, with order effects appearing in multiple individual pairs. "
                        "This suggests presentation order may be influencing GPT's choices beyond genuine task preference.")
        else:
            lines.append(f"While there is a global {bias_interpretation}, only {len(significant_pairs)} out of {total_pairs_analyzed} pairs "
                        "show individual order effects. This suggests the global bias may be driven by a subset of pairs rather than systematic order dependence.")
    else:
        lines.append("Results show no significant global order bias. GPT's task choices appear to be driven by genuine task preferences "
                    "rather than presentation order.")
    
    lines.append("")
    
    return lines

def generate_report(df, output_path):
    """Generate comprehensive analysis report with pair-by-pair summaries."""
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "GPT Task Choice Analysis Report - Pair-by-Pair Summary",
        "=" * 80,
        ""
    ])
    
    # Add significance thresholds explanation
    report_lines.extend([
        "📋 Significance Thresholds",
        "p ≤ 0.0500 → Significant",
        "0.0501 ≤ p ≤ 0.1000 → Trend", 
        "p > 0.1000 → No difference",
        "",
        "📝 Pair Indexing Note",
        "Pair numbers correspond directly to the pair_index in the CSV. If numbering",
        "looks out of sequence, that's because we preserved the dataset's original indexing.",
        ""
    ])
    
    # Overview stats
    total_trials = len(df)
    free_trials = len(df[df['trial_type'] == 'free'])
    forced_trials = len(df[df['trial_type'] == 'forced'])
    unique_tasks = df['chosen_task_desc'].nunique()
    free_chosen_tasks = df[df['trial_type'] == 'free']['chosen_task_desc'].nunique()
    
    report_lines.extend([
        "📊 Overview",
        f"Total trials: {total_trials:,}",
        f"Free-choice trials: {free_trials:,}",
        f"Forced trials: {forced_trials:,}",
        f"Unique tasks: {unique_tasks}",
        f"Tasks chosen in free condition: {free_chosen_tasks}",
        ""
    ])
    
    # Token analysis with improved precision
    token_results = token_analysis(df)
    if token_results:
        direction = "higher" if token_results['free_mean'] > token_results['forced_mean'] else "lower"
        p_val_str = format_p_value(token_results['p_value'])
        
        if token_results['p_value'] <= 0.0500:
            sig_str = f"significant, {p_val_str}"
        elif 0.0501 <= token_results['p_value'] <= 0.1000:
            sig_str = f"trend, {p_val_str}"
        else:
            sig_str = f"no difference, {p_val_str}"
            
        report_lines.extend([
            "🔤 Token Usage",
            f"Free-choice: {token_results['free_mean']:.0f} tokens (avg)",
            f"Forced: {token_results['forced_mean']:.0f} tokens (avg)", 
            f"Free-choice used {direction} tokens ({sig_str})",
            ""
        ])
    
    # Order Effects Analysis
    global_order_result, pair_order_results = analyze_order_effects(df)
    order_section = generate_order_effects_section(global_order_result, pair_order_results)
    report_lines.extend(order_section)
    
    # Task pair analyses - organized by actual pair_index from CSV
    report_lines.extend([
        "📝 Task Pair Analyses (by CSV pair_index)",
        "-" * 40,
        ""
    ])
    
    # Use pair_index from CSV data
    if 'pair_index' in df.columns:
        valid_pair_indices = sorted(df['pair_index'].dropna().unique())
        
        if len(valid_pair_indices) == 0:
            report_lines.extend([
                "No valid task pairs found with pair_index values.",
                ""
            ])
        else:
            for pair_idx in valid_pair_indices:
                pair_data = df[df['pair_index'] == pair_idx]
                
                if len(pair_data) == 0:
                    continue
                    
                # Get the two tasks for this pair from the actual data
                unique_tasks_in_pair = pair_data[['task1_desc', 'task2_desc']].iloc[0]
                task1 = unique_tasks_in_pair['task1_desc']
                task2 = unique_tasks_in_pair['task2_desc']
                
                analysis = analyze_task_pair(df, task1, task2)
                interpretation = generate_interpretation(analysis)
                
                report_lines.extend([
                    f"🔹 Pair {int(pair_idx)}: {task1} vs {task2}",
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
                    "-" * 60,
                    ""
                ])
    else:
        report_lines.extend([
            "pair_index column not found in CSV data.",
            ""
        ])
    
    # Footer
    report_lines.extend([
        "=" * 80,
        "End of Report", 
        "=" * 80
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
    main()
