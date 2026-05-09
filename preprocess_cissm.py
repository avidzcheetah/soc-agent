import pandas as pd
import json, os

df = pd.read_csv('data/raw/cissm_cyber_events.csv')

# The dataset headers have changed, we map them to the expected names
rename_map = {
    'event_type': 'attack_type',
    'organization': 'target',
    'description': 'end_effects'
}
df = df.rename(columns=rename_map)

# Keep only relevant columns
cols = ['event_date','actor','motive','target','end_effects','industry','country','attack_type','severity']
df = df[[c for c in cols if c in df.columns]]

# Drop rows with missing critical fields
df = df.dropna(subset=['attack_type','end_effects'])

# Convert structured record to text for SecBERT
def to_text(row):
    return (f"Attack: {row.get('attack_type','unknown')}. "
            f"Target: {row.get('target','unknown')}. "
            f"Industry: {row.get('industry','unknown')}. "
            f"Effects: {row.get('end_effects','unknown')}. "
            f"Motive: {row.get('motive','unknown')}")

df['text'] = df.apply(to_text, axis=1)

# Map end_effects to action labels
# 0=monitor, 1=isolate_host, 2=block_ip, 3=reset_credentials
def map_action(effect):
    e = str(effect).lower()
    if any(x in e for x in ['lateral','spread','movement']): return 1
    if any(x in e for x in ['network','ddos','traffic','access']): return 2
    if any(x in e for x in ['credential','password','account','brute']): return 3
    return 0

df['action_label'] = df['end_effects'].apply(map_action)
df.to_csv('data/processed/cissm_processed.csv', index=False)
print(f'Saved {len(df)} records')
