> “A smooth sea never made a skillful sailor” ~ Franklin D. Roosevelt

> I have attempted all the parts. [The metrics defined in task 3](#/blog/sed-bruyne#metrics) however are theoretical. The components needed for the metrics have been saved for the [batched gemini experiment](#/blog/sed-bruyne#batch-responses-from-gemini-25-flash-lite-for-few-shot-cot), but have not been calculated since most solutions were invalid.

The code lives on [GitHub ↗](https://github.com/PT-10/sed-bruyne).

![BIN_000 solution animation](sed-bruyne/bin-000-animation.gif)

# Generating Puzzles

## How I wanted to generate the puzzles?

The goal was to create a dataset that had a good mix of patterns, lengths of the initial strings, varying lengths of replacement sub-string, varied sizes of the transition rule-set and difficulty levels.

### Character and Length variation

```python
def get_vocab(self, level):
    if level <= 1: return list("AB")
    if level == 2: return list("ABCDE")
    if level == 3: return list("(){}<>?!#.<>$&*[]")
  return list("abcdeABCDE01234{}!#[]?")
```

![String Length vs Transitions vs Solution](sed-bruyne/scatter-3d.png)

![Top 10 Most Occurring Characters Across All Puzzle Initial Strings](sed-bruyne/character-counts.png)

![Distribution of Initial String Lengths](sed-bruyne/string-lengths.png)

## Difficulty Levels

> Try puzzles at [https://sed-alternating.web.app/](https://sed-alternating.web.app/)

### Entropy

Controls how much longer the target string can be compared to the source string in transformation rules

**Breakdown by entropy level:**

- **Entropy ≤ 1:** Target length = `[0, src_len]`
  - Example: If `src`="AB" (`len`=2), `tgt` can be 0-2 chars
  - Rules are **contractive or neutral** (always same size or smaller)
- **Entropy = 2 or 3:** Target length = `[0, src_len + 1]`
  - Example: If `src`="AB" (`len`=2), `tgt` can be 0-3 chars
  - Rules can be **slightly expansive** (+1 character)
- **Entropy ≥ 4:** Target length = `[0, src_len + 4]`
  - Example: If `src`="AB" (`len`=2), `tgt` can be 0-6 chars
  - Rules can be **highly expansive** (+4 characters)

Higher entropy allows rules like `"A" → "ABCDE"` (expanding), making it harder to predict which rules to apply and in what order.

We generate puzzles via a reverse generation process - applying rules backwards to ensure the puzzle is solvable. Since sed-puzzles are a type of semi-thue puzzles, there are essentially only  two substantial ways to ensure the construction of solvable puzzles.

- The first would be to create a random initial string, slice it based on user defined rules and the slices as `src` → `tgt` transitions.
- The second is our current method where we first create a transition ruleset with atleast one empty transition and use the ruleset to create a viable initial string.

We chose the second method, more on that later.

### Branching

Controls whether bogus/trap rules are added to mislead solvers (adds reversed sub-string rules when branching ≥ 2).

The reason we call it bogus is that it may or may not help in solving a string and in some cases potentially lead to dead-ends or inifinite loops.

**Example of a dead-end**

```json
{
  "vocab": ["A", "B"],
  "rules": [
    {"src": "AA", "tgt": ""},      // Void rule
    {"src": "AB", "tgt": "BA"},    // Reversible (bogus)
    {"src": "B", "tgt": "AAA"}     // Expanding rule
    {"src": "BA", "tgt": "AB"}
  ]
}
```

Without loss of generality, consider a sub-string: `"ABA"` of a larger string:

- If you apply `"B" → "AAA"` (when you shouldn't): `"AB"` → `"AAAAA"`

  You are now stuck - can only reduce by removing `"AA"` pairs. This results in `"AAAAA"` → `"AAA"` → `"A"` which is a dead-end since no transition now can result in `""` as you have odd number of `A`s left.
- In the same, the LLM may also be stuck in an infinite loop if it chooses to replace `"AB"` with `"BA"` and continue to do so.

### Symbols and Puzzle Types

Controls the vocabulary/character set complexity:

- Level ≤1: `"AB"` (2 chars)
- Level 2: `"ABCDE"` (5 chars)
- Level 3: `"(){}<>?!#.<>$&*[]"` (special chars)
- Level 4+: `"abcdeABCDE01234{}!#[]?"` (mixed alphanumeric + special)

**Puzzle Types**

- Alphanumeric or alphabetic puzzles using the above set
- Binary string type puzzles that exclusively use “0”s and “1”s
- NFA style puzzles

  While trying out examples on the [alternating site](https://sed-alternating.web.app/), I realized that a lot of the puzzles were similar to textual entailment of premises and deductions in a knowledge base (often studied in classical AI) and **non-deterministic finite automata (NFA)**.

  We have tried to capture the essence of state transfer and pose state changes as a string solving puzzle.

  Here is an example of an NFA style puzzle -

  ```json
  {
    "problem_id": "NFA_019",
    "initial_string": "Q0011101",
    "transitions": [
      { "src": "Q00", "tgt": "Q0" },
      { "src": "Q01", "tgt": "Q4" },
      { "src": "Q41", "tgt": "Q4" },
      { "src": "Q41", "tgt": "Q0" },
      { "src": "Q00", "tgt": "Q3" },
      { "src": "Q31", "tgt": "Q4" },
      { "src": "Q21", "tgt": "Q0" },
      { "src": "Q10", "tgt": "Q2" },
      { "src": "Q4", "tgt": "" }
    ]
  }
  ```

### The final mix

```python
all_ranges = [
((1,1,1),(2,2,2),20),
((2,2,2),(3,3,3),20),
((2,2,2),(3,3,3),20),
((3,3,3),(4,4,4),20)]
```

Each axis of difficulty has a scale from 1 to 4. A difficulty state of (1,1,1) denotes the easiest puzzle and state of (4,4,4) denotes the hardest puzzle. We create 20 puzzles in each range to have a uniform distribution of difficulty levels in puzzles.

## Method for generating semi-thue puzzles

We discuss the design choice of choosing a reverse generation method instead of **Method 1 (slicing)**, where rules are created by slicing an initial string into removable pieces.

- **Controlled difficulty:** Reverse generation lets us directly control solution depth via the number of construction steps, guaranteeing non-trivial puzzles. Method 1 offers no such guarantee and can easily produce puzzles solvable in just a few steps.
- **Strategic rule reuse:** Because the same rules may be applied repeatedly during construction, solvers must recognize patterns and decide *when* and *where* to apply them. In Method 1, each sliced rule is typically used once, making solutions mechanical.
- **Order-dependent solutions:** Reverse generation produces non-linear solution paths where rule order matters and incorrect choices can lead to dead ends. Method 1 is usually order-agnostic, apply all slices and the puzzle collapses.
- **Richer state space:** Reverse generation naturally supports expansive rules and competing transformations, creating branching that may lead to potential loops or dead-ends. Method 1 is bounded by the initial string and tends toward monotonic reduction.

The choice was made based on our understanding of the best possible way to create puzzles with varying levels of difficulty.

# Can your LLM solve it?

## dspy to the rescue (not really)

Having heard of dspy previously, my first instinct was to try prompt tuning with the help of dspy. dspy has something called optimizers that let you programmatically iterate over prompts instead of manually writing them. Three of the optimizers I used (or atleast tried to, it didn’t work) were -

Picked up verbatim from the dspy site -

- MIPROv2:

  Generates instructions *and* few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.
- SIMBA:

  Uses stochastic mini-batch sampling to identify challenging examples with high output variability, then applies the LLM to introspectively analyze failures and generate self-reflective improvement rules or add successful demonstrations.
- BootstrapFewShotWithRandomSearch:

  Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program over the optimization.

### LocalLLMs via llama.cpp

Since these require a lot of iterations and it was advised against paid APIs, I attempted to use the following models via a local GPU setup -

Llama3.2-3B

SmolLM3-128k

Qwen3-4B-128k

Qwen3-VL-8B

![It took forever :(](sed-bruyne/dspy-progress.png)

*It took forever :(*

However the iterations were painstakingly slow and I had to abandon this experiment despite setting it up and having invested substantial time.

### Gemini-2.5-Flash-lite

The gemini model has a rate limit of 20 requests per day which makes it difficult to use it for prompt optimization via dspy

## Prompting via web-interfaces

### Base questions for prompting

Three cherry picked questions -

```json
{
  "problem_id": "000",
  "initial_string": "BBBBBBBAABAA",
  "transitions": [
    { "src": "BA", "tgt": "" },
    { "src": "AA", "tgt": "BA" },
    { "src": "BAA", "tgt": "AA" },
    { "src": "B", "tgt": "A" }
  ]
}

{
  "problem_id": "BIN_025",
  "initial_string": "0000000000000101",
  "transitions": [
    { "src": "101", "tgt": "" },
    { "src": "01", "tgt": "10" },
    { "src": "100", "tgt": "01" },
    { "src": "00", "tgt": "" },
    { "src": "10", "tgt": "0" }
  ]
}

{
  "problem_id": "NFA_019",
  "initial_string": "Q0011101",
  "transitions": [
    { "src": "Q00", "tgt": "Q0" },
    { "src": "Q01", "tgt": "Q4" },
    { "src": "Q41", "tgt": "Q4" },
    { "src": "Q41", "tgt": "Q0" },
    { "src": "Q00", "tgt": "Q3" },
    { "src": "Q31", "tgt": "Q4" },
    { "src": "Q21", "tgt": "Q0" },
    { "src": "Q10", "tgt": "Q2" },
    { "src": "Q4", "tgt": "" }
  ]
}
```

## Screenshots

### Zero-shot prompt

The below are screenshots for the zero-shot prompt

````markdown
# String Rewriting Problem Solver
## Objective
Transform the given `initial_string` into an **empty string** (`""`) by applying a sequence of transitions.
---
## Rules
You are provided with:
- An `initial_string` to transform
- A list of `transitions`, where each transition has:
  - `src`: pattern to find in the string
  - `tgt`: replacement text

**Application Rules:**
1. **Pattern Matching**: You may apply a transition only if its `src` appears in the current string
2. **Leftmost Application**: Always replace the **first (leftmost)** occurrence of `src`
3. **Unlimited Reuse**: Transitions may be applied multiple times and in any order
4. **Zero-Based Indexing**: Track transitions by their 0-based index in the transitions array

**Goal**: Reach the empty string `""` (you do NOT need to find the shortest solution)
---
## Input Format
```json
{
  "problem_id": "<string>",
  "initial_string": "<string>",
  "transitions": [
    { "src": "<pattern>", "tgt": "<replacement>" },
    ...
  ]
}
```
---
## Task
Solve the problem below by finding **one valid sequence** of transition indices that reduces the initial string to `""`.

**Input:**
{{PROBLEM_JSON}}
---
## Output Format
Return **only** valid JSON matching this exact structure:
```json
{
  "problem_id": "<same_as_input>",
  "solution": [<index_0>, <index_1>, ..., <index_n>]
}
```
**Important:**
- No explanations, comments, or extra text
- Solution must be a valid sequence that produces the empty string
- All indices must be valid (0 ≤ index < number of transitions)
````

![NFA_019](sed-bruyne/zeroshot-nfa019.png)

*NFA_019*

![A zero-shot base prompt with Gemini-3 via the web interface for “problem 000”](sed-bruyne/zeroshot-000.png)

*A zero-shot base prompt with Gemini-3 via the web interface for “problem 000”*

Within the first few prompts it was clear that the models fail to answer correctly in zero-shot settings

![BIN_025 zero-shot](sed-bruyne/zeroshot-bin025.png)

![Problem 000 zero-shot](sed-bruyne/zeroshot-000-retry-1.png)

![Problem 000 follow-up](sed-bruyne/zeroshot-000-retry-2.png)

![Problem 000 step by step](sed-bruyne/zeroshot-000-stepwise-1.png)

![Problem 000 step by step, final solution](sed-bruyne/zeroshot-000-stepwise-2.png)

Despite continuing the chat and asking it to reason out load, the solution was still invalid.

Another experiment -

![Problem 000 correction](sed-bruyne/zeroshot-000-correction-1.png)

![Problem 000 infinite loop](sed-bruyne/zeroshot-000-correction-2.png)

GPT shows the same failure modes as Gemini, despite letting it proceed in a step by step manner even in a zero shot setting

![GPT zero-shot step by step](sed-bruyne/gpt-zeroshot-1.png)

![GPT miscounting](sed-bruyne/gpt-zeroshot-2.png)

### Fewshot + CoT

![Gemini few-shot CoT](sed-bruyne/fewshot-cot-gemini-1.png)

![Gemini few-shot CoT output](sed-bruyne/fewshot-cot-gemini-2.png)

The few shot + cot method works fine with gemini but gpt still shows the failure modes of improper counting which leads to reasoning failures down the line

![GPT few-shot CoT on BIN_025](sed-bruyne/fewshot-cot-gpt.png)

### I asked Gemini to improve the prompt (it did)

Gemini made a key pointer of highlighting how a better way to prompt would be to think of the initial string when applying a transition would be as shown in the below excerpt from the full prompt

````markdown
# Markov String Transformation

## Step-by-Step Reasoning Requirements
For every step, you must output the following block:
```Step [N]
Current String: [Full String]
Match: Found src "[pattern]" at index [I].
Slicing:
  Prefix (0 to I-1): [text]
  Replacement: [tgt]
  Suffix (I + src.length to end): [text]
New String: [Prefix + Replacement + Suffix]```
````

![Gemini with the Markov prompt](sed-bruyne/markov-gemini-1.png)

![Gemini with the Markov prompt, final solution](sed-bruyne/markov-gemini-2.png)

![GPT with the Markov prompt](sed-bruyne/markov-gpt.png)

The prompt works well with both GPT and Gemini and the explicit breakdown into `Prefix` + `Replacement` + `Suffix` results in less counting errors which prevents error being propogated through the reasoning chain.

## Formalizing the failure modes

**Faulty Counting → Trap States**

LLMs frequently miscount rule applications or symbol occurrences, leading to premature or incorrect transitions. These counting errors cause the model to enter trap states from which recovery is unlikely.

**Local Optimization → Global Failure**

Models often apply rules that yield immediate local reductions (e.g., shortest string or fastest apparent progress), but this greedy behavior blocks access to the globally valid solution path. The result is irreversible dead ends.

**Representation Sensitivity (Binary vs Alphabetic)**

Counting and local-optimization errors occur significantly more often in **binary string tasks** than in alphabetic (letter-based) strings.

**Oscillatory Transition Loops**

Local reasoning models may enter infinite loops by alternating between two transitions (e.g., transition 0 moves from state A→B, transition 1 moves B→A). The model fails to detect non-progress cycles and continues indefinitely. Happens more often in local LLM prompts.

## Prompting via gemini api

The major drawback of the free tier usage in the api is that the number of requests per day is limited. Since we can make only 20 requests per day, we have to batch our queries ie send as a single prompt. This greatly reduces the effectiveness of the prompt.

### Batch responses from gemini-2.5-flash-lite for few-shot CoT

Total valid 6/50

| problem_id | is_valid | llm_solution_length | baseline_solution_length | unsolved_percentage | early_stop | deadend |
| --- | --- | --- | --- | --- | --- | --- |
| BIN_016 | FALSE | 15 | 8 | 10 | TRUE | FALSE |
| BIN_012 | FALSE | 14 | 8 | 14.28571429 | TRUE | FALSE |
| BIN_018 | FALSE | 10 | 6 | 16.66666667 | TRUE | FALSE |
| 74 | FALSE | 8 | 6 | 16.66666667 | FALSE | TRUE |
| BIN_020 | FALSE | 18 | 11 | 17.85714286 | FALSE | TRUE |
| BIN_014 | FALSE | 7 | 6 | 18.75 | TRUE | FALSE |
| BIN_000 | FALSE | 6 | 5 | 23.07692308 | FALSE | TRUE |
| BIN_024 | FALSE | 13 | 12 | 23.80952381 | TRUE | FALSE |
| 43 | FALSE | 8 | 8 | 25 | TRUE | FALSE |
| BIN_033 | FALSE | 19 | 10 | 30.43478261 | TRUE | FALSE |

### Markov Prompt (Gemini generated)

The batch parser file, processes the output from the api response wherein the model is explicitly instructed to return only the json output.

````text
```json
{
  "problem_id": "NFA_006",
  "solution": [
    0,
    4,
    5,
    0
  ]
}
```

```json
{
  "problem_id": "NFA_001",
  "solution": [
    0,
    0,
    0,
    0
  ]
}
```

=== batch_response_all_50.txt ===
Valid: 0/50 | Early-stop: 47 | Deadend: 3
Done.
````

**Hypothesis**

- The issue here is that 47/50 problems were potentially on the right track but were possibly not solved due to our limitation of sending in batched inputs.
- The above failure may also be attributed to our choice of model - gemini-flash-2.5-lite. The limited capabilities might be the reason for failure as demonstrated by the example below where it gets stuck in an infinite loop of transitions eventually resulting in exhausting the max tokens limit.

![gemini-2.5-flash-lite stuck in a loop on NFA_006](sed-bruyne/flash-lite-loop.png)

Gemini-3 however works fine

![Gemini-3 solving NFA_006](sed-bruyne/markov-gemini3-nfa006.png)

# Metrics

## Validity

The first and most obvious one is validity which is checked by applying the solution returned by the LLM to see if the end state is an empty string.

### What happens on an invalid solution

We can have a qualitative metric on the invalid solution. We apply the solution on the initial string to get the partially solved string. Then we apply our baseline solver on the remaining string to see if it can reach a valid solution;

- if it can then that means our LLM was on the right path and for some reason did not return the full solution (or it supposedly did as per the LLM because of faulty counting or reasoning)
- if it cannot be solved then well the LLM failed entirely and reached a deadend

## Signed Efficiency Metric

The below metrics are used only when a valid solution is obtained.

### Optimality ratio

LLM solutions are more likely to be longer than the baseline solver solutions. As such, we may define an optimality ratio metric

- optimality_ratio = optimal_length / predicted_length
- 1.0 or greater means optimal, less than 1.0 means suboptimal

### Steps wasted

Steps wasted = predicted_length - optimal_length

The lower the better, in extremely rare cases (not come across one yet), it may be negative which would indicate that the LLM had a more efficient solution than the baseline solver

### Combining the metrics

The optimality ratio measures relative efficiency but ignores absolute overhead.

Steps wasted captures absolute inefficiency but is not comparable across problem scales.

Combining them yields a bounded, scale-invariant, and signed measure.

$$
\frac{L^* - \hat{L}}{|L^*| + |\hat{L}| + \epsilon}
$$

### Variables

- $L^*$ - Optimal or baseline solver solution length (number of steps)
- $\hat{L}$ - LLM-predicted solution length (number of steps)
- $\epsilon$ - Small positive constant to prevent division by zero (typically ε = 1)

### Interpretation

- $\text{Efficiency} \in [-1, 1]$
- $\text{Efficiency} = 0$ → LLM matches baseline efficiency
- $\text{Efficiency} > 0$ → LLM solution is more efficient (shorter)
- $\text{Efficiency} < 0$ → LLM solution is less efficient (longer)

### Response Time

The LLM response time is another paradigm that can be used as a metric.

- Response time for GPT on zero-shot prompts ~10 seconds, Gemini ~15 seconds, Local LLMs ~ 5 seconds (These were empirically measured when prompting via web-interfaces)
- The batch response time for Gemini-Flash-2.5-lite api ~ 60 seconds, while for 50 batched queries ~ 150 seconds (Note the api response times are rather unreliable as metrics because it depends on factors such as the internet connection)
- Response time for both Gemini and GPT on few-shot CoT prompts ~ 50 seconds. (empirically measured when prompting on web-interface)

# Are these base models better than humans?

The humans in reference would be me and the LLMs in question would be the local models, ChatGPT and Gemini.

## Where are humans better?

- Humans are less likely to fail on local traps or infinite loops, recognizing patterns better. This ability reduces as the string length increases
- They are also less likely to err on counting errors
- Humans tend to perform better on alphabetic puzzles, NFA as well as binary puzzles. However they struggle when a lot of symbols are involved.

## Where are LLMs better?

In our limited scope of experiments, LLMs showed limited ability to solve Sed-puzzles via prompting strategies often failing in one of the failure modes as mentioned [here](#/blog/sed-bruyne#formalizing-the-failure-modes).

- They most common fail point was their inability to count the characters correctly which would result in a faulty string despite applying a correct transition. The error would propagate through the steps which results in imminent failure.
- Locally hosted LLMs performed even worse as they confidently responded with incorrect and incomplete solutions. Their performance can be attributed to the small models sizes that were chosen due to hardware constraints (gpu poor).

## What should be done?

- Explicit post-training of base models on these style of puzzles on varying complexity, to create models with reasoning traces.
- Can we take inspiration from alpha-geometry?
  - Synthetic data generation (our puzzle generator can be used)
  - Auxiliary construction similarity:

    In AlphaGeometry, an auxiliary construction adds a new geometric object that is not part of the original problem but reveals hidden relationships needed to complete the proof.

    In the same way, an intermediate state in an SED puzzle introduces a string configuration that is not the final goal but exposes structure such as cancellations or invariants that makes further progress possible. In both cases, these additions do not change the problem’s validity but they temporarily enrich the representation so that reasoning steps that were previously inaccessible become achievable.
  - Train the LLM to propose rewrite rules of intermediate states rather than explicit steps and use an algorithmic solver as feedback loop while training the LLM to steer the model towards better reasoning on rewrite steps.
