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

# --------------------------------------------------------------------------- #
# Map end_effects + attack_type to a 20-action discrete action space.
# Derived from CISA Incident Response Playbook containment steps (7a-7n)
# and MITRE CAR / ATT&CK technique coverage analysis.
#
# Action Space (see data/action_space.csv for full definitions):
#  0  monitor                 11 reset_credentials
#  1  enable_deep_logging     12 disable_account
#  2  create_ioc_alert        13 remove_persistence
#  3  block_source_ip         14 restore_registry
#  4  block_dest_ip           15 restore_defense_config
#  5  dns_sinkhole            16 patch_vulnerability
#  6  block_port              17 snapshot_forensics
#  7  isolate_host            18 sandbox_redirect
#  8  kill_process            19 escalate_to_human
#  9  quarantine_file
# 10  quarantine_email
# --------------------------------------------------------------------------- #

def map_action(row):
    """Map a CISSM record to one of 20 SOC response actions."""
    effect = str(row.get('end_effects', '')).lower()
    attack = str(row.get('attack_type', '')).lower()
    combined = effect + ' ' + attack

    # --- High-severity / broad-impact → escalate or isolate -----------------
    if any(x in combined for x in ['ransomware', 'wiper', 'destruct']):
        return 7   # isolate_host — CISA Step 7d: ransomware/impact containment

    if any(x in combined for x in ['lateral', 'spread', 'movement', 'pivot',
                                    'worm', 'propagat']):
        return 7   # isolate_host — CISA Step 7d,7i: stop lateral movement

    # --- Credential-focused threats -----------------------------------------
    if any(x in combined for x in ['credential', 'password', 'brute',
                                    'pass the hash', 'pass-the-hash',
                                    'authentication', 'login attempt']):
        return 11  # reset_credentials — CISA Step 7f

    if any(x in combined for x in ['account', 'privilege escalat',
                                    'unauthorized access', 'admin compromise',
                                    'hijack account']):
        return 12  # disable_account — CISA Step 7f (stronger measure)

    # --- Malware / malicious execution on host ------------------------------
    if any(x in combined for x in ['trojan', 'malware', 'virus', 'dropper',
                                    'payload', 'backdoor download',
                                    'malicious file', 'infected file']):
        return 9   # quarantine_file — isolate malicious binary

    if any(x in combined for x in ['cryptominer', 'crypto mine', 'rat',
                                    'remote access tool', 'process inject',
                                    'code execution', 'exploit execution',
                                    'script execution', 'miner']):
        return 8   # kill_process — terminate malicious PID

    # --- Phishing / email-based attacks ------------------------------------
    if any(x in combined for x in ['phishing', 'spearphish', 'spam',
                                    'malicious email', 'email compromise',
                                    'bec', 'business email']):
        return 10  # quarantine_email — purge from mailboxes

    # --- Network / external attacks ----------------------------------------
    if any(x in combined for x in ['ddos', 'denial of service', 'dos attack',
                                    'flood', 'scan', 'probe', 'reconnaissance',
                                    'port scan']):
        return 3   # block_source_ip — perimeter firewall rule

    if any(x in combined for x in ['c2', 'command and control', 'callback',
                                    'beacon', 'command-and-control']):
        return 4   # block_dest_ip — block outbound C2 traffic

    if any(x in combined for x in ['dns', 'domain', 'watering hole',
                                    'redirect', 'typosquat']):
        return 5   # dns_sinkhole — redirect malicious domain resolution

    if any(x in combined for x in ['exploit', 'remote code', 'rce',
                                    'vulnerability', 'cve', 'zero-day',
                                    '0day', 'unpatched']):
        return 16  # patch_vulnerability — trigger CVE remediation

    if any(x in combined for x in ['exfiltrat', 'data theft', 'data leak',
                                    'data breach', 'information stolen',
                                    'data loss']):
        return 4   # block_dest_ip — cut egress exfiltration channel

    # --- Persistence mechanisms --------------------------------------------
    if any(x in combined for x in ['persistence', 'backdoor', 'rootkit',
                                    'scheduled task', 'registry', 'autorun',
                                    'webshell', 'web shell']):
        return 13  # remove_persistence — eradicate persistence mechanism

    # --- Defense evasion ---------------------------------------------------
    if any(x in combined for x in ['disable security', 'antivirus',
                                    'tamper', 'defense evasion',
                                    'log delet', 'log clear']):
        return 15  # restore_defense_config — re-enable defenses

    # --- Port / service specific -------------------------------------------
    if any(x in combined for x in ['rdp', 'smb', 'ssh', 'telnet',
                                    'remote service', 'open port']):
        return 6   # block_port — close exposed service port

    # --- Suspicious but unclear — gather more intel ------------------------
    if any(x in combined for x in ['suspicious', 'anomal', 'unusual',
                                    'unknown', 'unidentified']):
        return 1   # enable_deep_logging — increase telemetry

    # --- Espionage / APT / state-sponsored → human review ------------------
    if any(x in combined for x in ['espionage', 'apt', 'state-sponsored',
                                    'nation state', 'critical infrastructure',
                                    'scada', 'ics', 'nuclear']):
        return 19  # escalate_to_human — Tier-3 analyst review

    # --- Default: monitor (low confidence / informational) -----------------
    return 0       # monitor — log and close

df['action_label'] = df.apply(map_action, axis=1)
df.to_csv('data/processed/cissm_processed.csv', index=False)

# Print distribution summary
action_names = [
    'monitor', 'enable_deep_logging', 'create_ioc_alert',
    'block_source_ip', 'block_dest_ip', 'dns_sinkhole',
    'block_port', 'isolate_host', 'kill_process',
    'quarantine_file', 'quarantine_email', 'reset_credentials',
    'disable_account', 'remove_persistence', 'restore_registry',
    'restore_defense_config', 'patch_vulnerability', 'snapshot_forensics',
    'sandbox_redirect', 'escalate_to_human'
]
print(f'Saved {len(df)} records to data/processed/cissm_processed.csv')
print(f'\nAction label distribution ({len(action_names)} actions):')
counts = df['action_label'].value_counts().sort_index()
for idx, count in counts.items():
    pct = count / len(df) * 100
    print(f'  {idx:2d}  {action_names[idx]:25s}  {count:5d}  ({pct:5.1f}%)')
