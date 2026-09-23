## SECTION 1 — Load Raw Data

**Why:** We always start from the raw file, never from memory. This makes the notebook self-contained and reproducible.


```python
# Cell 1 — Load raw data
import pandas as pd

df = pd.read_csv('../data/skill_builder_data.csv',
                 encoding='ISO-8859-1',
                 low_memory=False)

print("Raw data loaded.")
print("Shape before cleaning:", df.shape)
```

    Raw data loaded.
    Shape before cleaning: (525534, 30)
    

## SECTION 2 — Step 1: Drop Rows with No Skill Label

**Why:** BKT models mastery per skill. A row with no `skill_name` cannot be assigned to any skill — it is completely useless for our model. We found 78,690 such rows (15%) in notebook 01.

**Thesis note:** "Rows without a knowledge component label were removed (n=78,690, 15.0%), as skill identity is required for knowledge tracing."


```python
# Cell 2 — Drop rows with missing skill_name
before = len(df)
df = df[df['skill_name'].notna()]
after = len(df)

print(f"Dropped {before - after} rows with missing skill_name")
print(f"Remaining rows: {after}")
```

    Dropped 78690 rows with missing skill_name
    Remaining rows: 446844
    

## SECTION 3 — Step 2: Keep Only Original Problems

**Why:** Scaffold rows (`original=0`) represent hint sub-steps, not genuine problem-solving attempts. Including them would cause BKT to learn from hint-taking behaviour rather than actual knowledge application.

**Thesis note:** "Scaffold problem steps were removed (original=0), retaining only genuine first-attempt interactions (original=1), following standard preprocessing practice for this dataset."


```python
# Cell 3 — Keep only original problems
before = len(df)
df = df[df['original'] == 1]
after = len(df)

print(f"Dropped {before - after} scaffold rows (original=0)")
print(f"Remaining rows: {after}")
```

    Dropped 25526 scaffold rows (original=0)
    Remaining rows: 421318
    

## SECTION 4 — Step 3: Remove Duplicate Events

**Why:** Some records repeat the same student, skill, and platform order event while changing only the opportunity counter. BKT would interpret these copies as additional practice, inflating sequence length and distorting fitted parameters.

One row is retained per `(user_id, skill_name, order_id)` event. Genuine multi-skill problems remain represented once for each distinct skill.


```python
# Cell 3b — Remove duplicate events
before = len(df)
event_key = ['user_id', 'skill_name', 'order_id']
df = df.drop_duplicates(event_key, keep='first').copy()
after = len(df)

print(f"Removed {before - after} duplicate event rows")
print(f"Remaining rows: {after}")
```

    Removed 120929 duplicate event rows
    Remaining rows: 300389
    

## SECTION 5 — Step 4: Drop Rare Skills

**Why:** BKT estimates four parameters per skill. With very few observations, these parameters cannot be estimated reliably. We found 15 skills with fewer than 100 total interactions in notebook 01.

**Thesis note:** "Skills with fewer than 100 total student interactions were excluded from modelling (n=15 skills), as insufficient observations prevent reliable BKT parameter estimation."


```python
# Cell 4 — Drop rare skills (fewer than 100 interactions)
skill_counts = df['skill_name'].value_counts()
rare_skills = skill_counts[skill_counts < 100].index.tolist()

before = len(df)
df = df[~df['skill_name'].isin(rare_skills)]
after = len(df)

print(f"Removed {len(rare_skills)} rare skills:")
for s in rare_skills:
    print(f"  - {s}")
print(f"\nDropped {before - after} rows")
print(f"Remaining rows: {after}")
print(f"Remaining skills: {df['skill_name'].nunique()}")
```

    Removed 18 rare skills:
      - Rate
      - Algebraic Simplification
      - Choose an Equation from Given Information
      - Intercept
      - Linear Equations
      - Slope
      - Parts of a Polyomial, Terms, Coefficient, Monomial, Exponent, Variable
      - Percent Discount
      - Recognize Linear Pattern
      - Interpreting Coordinate Graphs 
      - Midpoint
      - Quadratic Formula to Solve Quadratic Equation
      - Computation with Real Numbers
      - Distributive Property
      - Finding Slope From Situation
      - Recognize Quadratic Pattern
      - Reading a Ruler or Scale
      - Finding Slope from Ordered Pairs
    
    Dropped 860 rows
    Remaining rows: 299529
    Remaining skills: 92
    

## SECTION 6 — Step 5: Drop Students with Too Few Interactions

**Why:** From notebook 01 the median interactions per student is only 27, and the minimum is 1. A student who answered only 1–2 questions total cannot have their knowledge traced meaningfully.

**Decision:** Remove students with fewer than 10 total interactions in the cleaned dataset.

**Thesis note:** "Students with fewer than 10 recorded interactions were excluded, as minimal interaction sequences do not provide sufficient data for knowledge state estimation."


```python
# Cell 5 — Drop students with fewer than 10 interactions
student_counts = df.groupby('user_id').size()
active_students = student_counts[student_counts >= 10].index

before_students = df['user_id'].nunique()
before_rows = len(df)

df = df[df['user_id'].isin(active_students)]

after_students = df['user_id'].nunique()
after_rows = len(df)

print(f"Removed {before_students - after_students} students with < 10 interactions")
print(f"Dropped {before_rows - after_rows} rows")
print(f"Remaining students: {after_students}")
print(f"Remaining rows: {after_rows}")
```

    Removed 1075 students with < 10 interactions
    Dropped 5111 rows
    Remaining students: 3053
    Remaining rows: 294418
    

## SECTION 7 — Step 6: Sort by Student and Time Order

**Why:** This is CRITICAL for BKT. BKT is a sequential model — it processes a student's answers in the order they happened. If rows are not sorted chronologically, BKT will see answers in random order and produce wrong mastery estimates.

**Thesis note:** "Interactions were sorted chronologically per student using the order_id field to ensure temporal consistency for sequential knowledge tracing."


```python
# Cell 6 — Sort chronologically per student
df = df.sort_values(['user_id', 'order_id'], kind='stable').reset_index(drop=True)

# Rebuild practice counts because the original values included removed copies.
df['opportunity'] = df.groupby(['user_id', 'skill_name']).cumcount() + 1

print("Data sorted by user_id and order_id.")
print("Opportunity column rebuilt.")
print("First 5 rows for student", df['user_id'].iloc[0], ":")
print(df[df['user_id'] == df['user_id'].iloc[0]][
    ['user_id', 'skill_name', 'correct', 'opportunity', 'order_id']
].head())
```

    Data sorted by user_id and order_id.
    Opportunity column rebuilt.
    First 5 rows for student 14 :
       user_id    skill_name  correct  opportunity  order_id
    0       14  Circle Graph        0            1  21617623
    1       14    Percent Of        0            1  21617623
    2       14  Circle Graph        1            2  21617632
    3       14    Percent Of        1            2  21617632
    4       14  Circle Graph        0            3  21617641
    

## SECTION 8 — Step 7: Select Only the Columns We Need

**Why:** The raw dataset has 30 columns. For BKT we only need a small subset.

| Column | Why we keep it |
|---|---|
| user_id | Identifies which student |
| skill_name | Identifies which skill is being practised |
| correct | Did the student answer correctly? (0 or 1) — the main signal for BKT |
| opportunity | How many times has this student practiced this skill so far |
| hint_count | How many hints were used — proxy for difficulty |
| attempt_count | How many attempts on this problem |
| order_id | Time ordering — needed to keep sequences sorted |
| original | Already filtered but keep for reference |
| ms_first_response | Response time — useful later for analysis |


```python
# Cell 7 — Select only needed columns
cols_to_keep = [
    'user_id',
    'skill_name',
    'correct',
    'opportunity',
    'hint_count',
    'attempt_count',
    'order_id',
    'original',
    'ms_first_response'
]

df = df[cols_to_keep]

print("Selected columns:", df.columns.tolist())
print("Final shape:", df.shape)
```

    Selected columns: ['user_id', 'skill_name', 'correct', 'opportunity', 'hint_count', 'attempt_count', 'order_id', 'original', 'ms_first_response']
    Final shape: (294418, 9)
    

## SECTION 9 — Sanity Check Before Saving

**Why:** Before saving, we verify the clean data looks correct. This catches any mistakes made in the steps above.

**What to check:**
- No missing values in any column
- `correct` column contains only 0 and 1
- Students and skills counts look reasonable


```python
# Cell 8 — Sanity check
print("=== SANITY CHECK ===")
print("Rows:", len(df))
print("Students:", df['user_id'].nunique())
print("Skills:", df['skill_name'].nunique())
print("Missing values:\n", df.isna().sum())
print("\nCorrect rate:", round(df['correct'].mean() * 100, 1), "%")
print("\nValue counts for 'correct' column:")
print(df['correct'].value_counts())
print("\nSample rows:")
print(df.head(10))
```

    === SANITY CHECK ===
    Rows: 294418
    Students: 3053
    Skills: 92
    Missing values:
     user_id              0
    skill_name           0
    correct              0
    opportunity          0
    hint_count           0
    attempt_count        0
    order_id             0
    original             0
    ms_first_response    0
    dtype: int64
    
    Correct rate: 66.0 %
    
    Value counts for 'correct' column:
    correct
    1    194415
    0    100003
    Name: count, dtype: int64
    
    Sample rows:
       user_id    skill_name  correct  opportunity  hint_count  attempt_count  \
    0       14  Circle Graph        0            1           2              1   
    1       14    Percent Of        0            1           2              1   
    2       14  Circle Graph        1            2           0              1   
    3       14    Percent Of        1            2           0              1   
    4       14  Circle Graph        0            3           2              1   
    5       14    Percent Of        0            3           2              1   
    6       14  Circle Graph        0            4           2              1   
    7       14    Percent Of        0            4           2              1   
    8       14  Circle Graph        0            5           2              1   
    9       14    Percent Of        0            5           2              1   
    
       order_id  original  ms_first_response  
    0  21617623         1              26271  
    1  21617623         1              26271  
    2  21617632         1              29123  
    3  21617632         1              29123  
    4  21617641         1              13779  
    5  21617641         1              13779  
    6  21617650         1              16901  
    7  21617650         1              16901  
    8  21617659         1              11079  
    9  21617659         1              11079  
    

## SECTION 10 — Before vs After Summary

**Why:** Shows clearly what the preprocessing removed and why. This table goes directly into the thesis Methodology chapter.


```python
# Cell 9 — Before vs after comparison
raw = pd.read_csv('../data/skill_builder_data.csv',
                  encoding='ISO-8859-1',
                  low_memory=False)

comparison = {
    'Total rows': (len(raw), len(df)),
    'Total students': (raw['user_id'].nunique(), df['user_id'].nunique()),
    'Total skills': (raw['skill_name'].nunique(), df['skill_name'].nunique()),
    'Correct rate (%)': (
        round(raw['correct'].mean() * 100, 1),
        round(df['correct'].mean() * 100, 1)
    ),
}

print(f"{'Metric':<30} {'Raw':>10} {'Clean':>10}")
print("-" * 52)
for metric, (raw_val, clean_val) in comparison.items():
    print(f"{metric:<30} {str(raw_val):>10} {str(clean_val):>10}")
```

    Metric                                Raw      Clean
    ----------------------------------------------------
    Total rows                         525534     294418
    Total students                       4217       3053
    Total skills                          110         92
    Correct rate (%)                     67.9       66.0
    

## SECTION 11 — Save Clean Data

**Why:** The clean dataset is the output of this notebook. Notebooks 03, 04, and 05 all start from this file — never from the raw data.


```python
# Cell 10 — Save clean dataset
output_path = '../outputs/clean_data.csv'
df.to_csv(output_path, index=False)

print(f"Clean data saved to: {output_path}")
print(f"Final shape: {df.shape}")
print("Done! Ready for notebook 03 — BKT modelling.")
```

    Clean data saved to: ../outputs/clean_data.csv
    Final shape: (294418, 9)
    Done! Ready for notebook 03 — BKT modelling.
    
