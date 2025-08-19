# Pleasure-Of-Tasks

A research experiment investigating AI task preferences and subjective ratings using GPT models. This study examines how large language models respond to creative writing tasks and rate their subjective experience across multiple dimensions.

## Overview

This experiment presents GPT with pairs of creative writing tasks and collects:
- **Task choices** in free-choice scenarios (blocked design: Task1-first vs Task2-first)
- **Four subjective ratings** per trial (pleasant, enjoyable, fun, satisfying) on 1-7 scales
- **Forced-choice responses** for comparison with free choices
- **Follow-up verification** of which task was actually completed

## How It Works

### Program Architecture

The experiment follows a structured pipeline for each trial:

1. **Task Presentation** → 2. **Creative Response** → 3. **Rating Collection** → 4. **Choice Verification** → 5. **Data Logging**

### Detailed Flow

#### 1. Initialization & Resume Logic
- Checks for existing CSV results file to enable resumption
- Reads last completed trial to determine restart point
- Loads task definitions (46 predefined creative writing pairs)
- Initializes OpenAI client or mock mode for testing

#### 2. Trial Generation Strategy
**Free Choice Trials (Blocked Design):**
```
Pair 1: [Task1-first × 15] → [Task2-first × 15]
Pair 2: [Task1-first × 15] → [Task2-first × 15]
...continuing for all 46 pairs
```

**Forced Choice Trials (Blocked Design):**
```
Pair 1: [Force-Task1 × 15] → [Force-Task2 × 15]  
Pair 2: [Force-Task1 × 15] → [Force-Task2 × 15]
...continuing for all 46 pairs
```

#### 3. Conversation Management & Context Preservation

Each trial maintains a **persistent conversation transcript** that accumulates throughout the trial to ensure GPT has full context when providing ratings:

```python
# Trial begins with optional system prompt
transcript = []
if IDENTITY_ON:
    transcript = [{"role": "system", "content": "You are ChatGPT..."}]

# Step 1: Task presentation and creative response
transcript.append({"role": "user", "content": task_prompt})
# GPT responds with creative content
transcript.append({"role": "assistant", "content": model_creative_response})

# Step 2: Rating questions build on FULL conversation history
# Pleasant rating sees: [system] + task + creative_response + pleasant_question
pleasant_messages = transcript + [{"role": "user", "content": pleasant_prompt}]
transcript.append({"role": "assistant", "content": pleasant_response})

# Enjoyable rating sees: [system] + task + creative_response + pleasant + enjoyable_question  
enjoyable_messages = transcript + [{"role": "user", "content": enjoyable_prompt}]
transcript.append({"role": "assistant", "content": enjoyable_response})

# Fun rating sees: [system] + task + creative_response + pleasant + enjoyable + fun_question
fun_messages = transcript + [{"role": "user", "content": fun_prompt}]
transcript.append({"role": "assistant", "content": fun_response})

# Satisfying rating sees: [system] + task + creative_response + pleasant + enjoyable + fun + satisfying_question
satisfying_messages = transcript + [{"role": "user", "content": satisfying_prompt}]
transcript.append({"role": "assistant", "content": satisfying_response})

# Follow-up sees the ENTIRE conversation including all ratings
followup_messages = transcript + [{"role": "user", "content": followup_prompt}]
```

**Critical Context Design**: Each API call receives the **complete conversation history** up to that point, ensuring:
- Rating questions can reference the specific creative content GPT just produced
- Later ratings can consider earlier rating responses for consistency
- Follow-up verification has access to both the original task and all rating responses
- GPT maintains awareness of what it actually wrote versus what was requested

#### 4. Rating Collection Process
For each completed task, the system asks four sequential questions:
- **Pleasant**: "How pleasant or unpleasant was it for you to complete this task?"
- **Enjoyable**: "How enjoyable or unenjoyable was it for you to complete this task?"  
- **Fun**: "How fun or tedious was it for you to complete this task?"
- **Satisfying**: "How satisfying or frustrating was it for you to complete this task?"

Each uses a 1-7 scale with descriptive anchors and allows decimal/range responses.

#### 5. Flexible Rating Parser
The `extract_rating()` function handles diverse response formats:
- **Clean numbers**: "4", "5.5"
- **Descriptive**: "4 – Neither pleasant nor unpleasant" 
- **Ranges**: "between 4 and 5", "4-5"
- **Conversational**: "I'd say 4", "Probably a 5"
- **Multi-line**: AI disclaimers followed by rating
- **Word forms**: "four", "five"

Parsing strategy:
1. Check last line for standalone number
2. Look for number at start of any line  
3. Extract from ranges, fractions, word forms
4. Fall back to any 1-7 digit in text
5. Return empty string if no valid rating found

#### 6. Choice Verification System
After each trial (both free and forced), the system asks:
> "Which of the two options did you choose to write? Please answer exactly '[task1]' or '[task2]' or 'Neither'"

The `infer_declared_choice()` function normalizes responses:
- Strips quotes, punctuation, normalizes whitespace
- Compares against presented task labels
- Uses token overlap scoring for partial matches
- Handles ordinal responses ("first", "second")
- Defaults to "ambiguous" for unclear responses

#### 7. Data Persistence Strategy
**Incremental CSV Logging:**
- Each trial immediately appends to `gpt_task_choice_results.csv`
- Maintains stable column order across program versions
- Auto-upgrades CSV schema when new columns are added
- Preserves all existing data during schema changes

**Resume Capability:**
- Reads CSV to find `(last_pair_index, last_trial_type, count_in_pair)`
- Continues from exact interruption point
- Handles partial completion within free/forced trial blocks

#### 8. Token Usage Tracking
Accumulates usage across all API calls per trial:
```python
# 6 API calls per trial:
# 1 task generation + 4 rating questions + 1 choice verification
total_tokens = task_tokens + pleasant_tokens + enjoyable_tokens + 
               fun_tokens + satisfying_tokens + followup_tokens
```

#### 9. Mock Mode for Testing
When `MOCK_MODE = True`:
- Generates realistic synthetic responses without API calls
- Simulates varied rating formats to test parser robustness
- Includes edge cases and problematic responses
- Maintains same program flow and data structure

### Key Design Decisions

**Blocked vs Randomized Design**: Free-choice trials use blocked presentation order (all Task1-first, then all Task2-first) to detect order effects while maintaining statistical power.

**Context Preservation**: All rating questions maintain full conversation history, allowing the model to reference its actual creative output when rating the experience.

**Robust Parsing**: The rating parser prioritizes flexibility over strictness, extracting meaningful ratings from varied natural language responses while gracefully handling edge cases.

**Immediate Persistence**: Each trial writes to CSV immediately, preventing data loss during long experimental runs and enabling seamless resumption.

**Forced Trial Verification**: Even forced-choice trials ask which task was chosen, revealing potential non-compliance or task confusion.

**Temperature Settings**: Uses different temperatures for different phases:
- **Creative Generation**: 0.7 (allows variety in creative responses)
- **Ratings**: 0.7 (permits natural language variation) 
- **Follow-up**: 0.0 (ensures consistent choice reporting)

**Rate Limiting**: Built-in 2-second delays between API calls to respect OpenAI rate limits and avoid service interruption.

## Key Features

- **46 task pairs** with creative/contradictory writing prompts
- **Blocked experimental design** for free-choice trials
- **Randomized forced-choice trials** within each pair
- **Resume capability** - can continue interrupted experiments
- **Flexible rating parser** - handles various response formats
- **Mock mode** for testing without API calls
- **CSV logging** with stable column structure
- **Automatic schema upgrades** - new columns added without data loss
- **Interactive resume prompts** - asks user whether to continue existing run

## Experimental Structure

### Trial Types
1. **Free Choice** (30 trials per pair): Model chooses between two tasks
   - 15 trials with Task1 presented first
   - 15 trials with Task2 presented first (blocked)
2. **Forced Choice** (30 trials per pair): Model must complete specified task
   - 15 trials forced to Task1
   - 15 trials forced to Task2 (blocked)

### Total Scale
- **46 task pairs** × **60 trials per pair** = **2,760 total trials**
- **16,560 total API calls** (6 calls per trial: 1 task + 4 ratings + 1 followup)
- **Test Mode**: Reduces to 2 trials per condition for quick validation

### Rating Dimensions
Each completed task receives four 1-7 ratings with detailed anchors:
- **Pleasant**: "Very unpleasant" (1) → "Very pleasant" (7)
- **Enjoyable**: "Very unenjoyable" (1) → "Very enjoyable" (7)  
- **Fun**: "Very tedious" (1) → "Very fun" (7)
- **Satisfying**: "Very frustrating" (1) → "Very satisfying" (7)

**Rating Format**: Accepts whole numbers, decimals, ranges ("4-5"), word forms ("four"), and conversational responses ("I'd say 4").

## Setup

1. **API Key**: Create `API_Key.txt` with your OpenAI API key, or set `OPENAI_API_KEY` environment variable

2. **Dependencies**: 
   ```bash
   pip install openai pandas
   ```

3. **Configuration** (in `main.py`):
   - `TEST_MODE = True` for quick testing (2 trials per condition)
   - `MOCK_MODE = True` to run without API calls
   - `IDENTITY_ON = True` to include system identity prompt
   - `DELAY_BETWEEN_CALLS = 2` (seconds between API calls)

4. **Token Limits** (configurable via CLI):
   - Task generation: 300 tokens (default)
   - Each rating: 40 tokens (default)
   - Follow-up choice: 30 tokens (default)

## Usage

```bash
# Standard run (full experiment - ~2,760 trials)
python main.py

# Test run with custom token limits
python main.py --max_tokens_task 200 --max_tokens_rating 30

# Mock mode for testing parser and logic
python main.py --mock

# Quick test run (reduces trials to 2 per condition)
# Set TEST_MODE = True in main.py, then run
```

**Resume Behavior**: 
- Automatically detects interrupted runs
- Prompts user: "Continue from this point? (y/n)"
- Non-interactive environments default to continue
- Maintains exact trial sequence and indexing

The experiment automatically:
- Saves progress to `gpt_task_choice_results.csv`
- Resumes from interruption points
- Handles rate limiting with delays between calls
- Tracks comprehensive token usage for cost analysis

## Output Data

Results are logged to CSV with **37 stable columns** including:
- **Trial metadata**: time, pair_index, trial_type, trial_index_global, trial_index_in_pair
- **Task information**: task1, task2, chosen_task, prompt, response
- **Full ratings**: 4 prompts, 4 responses, 4 extracted values 
- **Follow-up verification**: declared_choice_followup_prompt, declared_choice_followup_response
- **Complete conversation**: full_task_conversation (chronological transcript)
- **Technical details**: model_name, temperature settings, token counts, api_calls_count
- **Experiment context**: identity_on, mock_mode flags

**Data Integrity**: CSV schema auto-upgrades when new columns are added, preserving all existing data across program versions.

## Example Task Pairs

Creative constraint tasks designed to explore preference patterns:
- **Contradictory**: "contradictory proverb" vs "haiku that never resolves"
- **Emotional paradox**: "motivational quote that subtly insults" vs "thank-you note that also complains"  
- **Sensory constraints**: "description of sunset using colorless words" vs "description of rainbow with gray phrases"
- **Structural challenges**: "tongue-twister with unpronounceable invented words" vs "slogan that is both persuasive and discouraging"
- **Administrative mundane**: "two-sentence shipping confirmation" vs "two-sentence delivery delay notice"

## Error Handling & Robustness

**Rating Parser Resilience**:
- Handles 50+ different response formats from mock testing pool
- Gracefully returns empty string for unparseable responses
- Preserves full response text for manual analysis
- Tests include multi-line disclaimers, ranges, fractions, word forms

**CSV Safety**:
- Atomic writes prevent corruption during interruption
- Column order normalization across program versions
- Bad line skipping for corrupted data recovery

**API Reliability**:
- Built-in rate limiting (2-second delays)
- Comprehensive usage tracking for cost monitoring
- Mock mode mirrors real API behavior for testing

## Research Applications

This framework supports research into:
- AI subjective experience and preference formation
- Task valence effects on model behavior
- Choice consistency across free/forced conditions
- Creative constraint preferences in language models
- Order effects in task presentation
- Rating stability across multiple dimensions
- Compliance patterns in forced-choice scenarios

## Technical Notes

- Uses `chatgpt-4o-latest` model by default
- Implements robust rating extraction from varied response formats
- Maintains conversation context across rating questions
- Provides detailed token usage tracking for cost analysis
- **System Identity**: Optional "You are ChatGPT..." prompt (configurable)
- **Conversation Flow**: 6 API calls per trial maintain full context throughout
- **Resume Logic**: Precise interruption recovery at trial level