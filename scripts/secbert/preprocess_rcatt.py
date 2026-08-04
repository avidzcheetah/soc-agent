import pandas as pd
import re

# ---------------------------------------------------------------------------
# Preprocess the rcATT dataset for the SOC-Agent RL pipeline.
#
# The rcATT dataset (https://github.com/vlegoy/rcATT) contains ~1490 cyber
# threat intelligence reports with multi-label MITRE ATT&CK tactic and
# technique annotations.
#
# This script:
#   1. Truncates the raw report text to ~2000 chars (SecBERT 512-token limit)
#   2. Cleans boilerplate/navigation text from scraped web pages
#   3. Maps MITRE ATT&CK tactics+techniques → one of 20 SOC response actions
#   4. For multi-tactic reports, selects the highest-priority action
#   5. Outputs data/processed/rcatt_processed.csv
# ---------------------------------------------------------------------------

# MITRE ATT&CK Tactic IDs (as column names in the dataset)
# TA0001 = Initial Access     TA0002 = Execution
# TA0003 = Persistence        TA0004 = Privilege Escalation
# TA0005 = Defense Evasion    TA0006 = Credential Access
# TA0007 = Discovery          TA0008 = Lateral Movement
# TA0009 = Collection         TA0010 = Exfiltration
# TA0011 = Command & Control  TA0040 = Impact

TACTIC_NAMES = {
    'TA0001': 'Initial Access',
    'TA0002': 'Execution',
    'TA0003': 'Persistence',
    'TA0004': 'Privilege Escalation',
    'TA0005': 'Defense Evasion',
    'TA0006': 'Credential Access',
    'TA0007': 'Discovery',
    'TA0008': 'Lateral Movement',
    'TA0009': 'Collection',
    'TA0010': 'Exfiltration',
    'TA0011': 'Command and Control',
    'TA0040': 'Impact',
}

# Key technique-to-action mappings (MITRE ATT&CK technique ID → action index)
# These override the tactic-level defaults when a specific technique is present.
TECHNIQUE_ACTION_MAP = {
    # Credential Access techniques → reset_credentials (11)
    'T1003': 11,  # Credential Dumping
    'T1110': 11,  # Brute Force
    'T1081': 11,  # Credentials in Files
    'T1214': 11,  # Credentials in Registry
    'T1212': 11,  # Exploitation for Credential Access
    'T1187': 11,  # Forced Authentication
    'T1208': 11,  # Kerberoasting
    'T1142': 11,  # Keychain
    'T1171': 11,  # LLMNR/NBT-NS Poisoning
    'T1075': 11,  # Pass the Hash
    'T1097': 11,  # Pass the Ticket
    'T1145': 11,  # Private Keys
    'T1111': 11,  # Two-Factor Authentication Interception
    'T1167': 11,  # Input Prompt (credential)

    # Account manipulation / privilege escalation → disable_account (12)
    'T1098': 12,  # Account Manipulation
    'T1136': 12,  # Create Account
    'T1078': 12,  # Valid Accounts (compromised)
    'T1134': 12,  # Access Token Manipulation
    'T1178': 12,  # SID-History Injection

    # Persistence mechanisms → remove_persistence (13)
    'T1060': 13,  # Registry Run Keys / Startup Folder
    'T1053': 13,  # Scheduled Task
    'T1050': 13,  # New Service
    'T1004': 13,  # Winlogon Helper DLL
    'T1137': 13,  # Office Application Startup
    'T1100': 13,  # Web Shell
    'T1501': 13,  # Systemd Service
    'T1168': 13,  # Local Job Scheduling
    'T1165': 13,  # Startup Items
    'T1131': 13,  # Authentication Package
    'T1128': 13,  # Netsh Helper DLL
    'T1013': 13,  # Port Monitors
    'T1084': 13,  # WMI Event Subscription
    'T1176': 13,  # Browser Extensions
    'T1159': 13,  # Launch Agent
    'T1160': 13,  # Launch Daemon

    # Registry-specific persistence/evasion → restore_registry (14)
    'T1112': 14,  # Modify Registry
    'T1103': 14,  # AppInit DLLs
    'T1183': 14,  # Image File Execution Options Injection
    'T1182': 14,  # AppCert DLLs

    # Defense evasion / disabling security → restore_defense_config (15)
    'T1089': 15,  # Disabling Security Tools
    'T1054': 15,  # Indicator Blocking
    'T1070': 15,  # Indicator Removal on Host
    'T1099': 15,  # Timestomp
    'T1198': 15,  # SIP and Trust Provider Hijacking
    'T1130': 15,  # Install Root Certificate

    # Exploitation / vulnerability → patch_vulnerability (16)
    'T1190': 16,  # Exploit Public-Facing Application
    'T1203': 16,  # Exploitation for Client Execution
    'T1210': 16,  # Exploitation of Remote Services
    'T1068': 16,  # Exploitation for Privilege Escalation
    'T1211': 16,  # Exploitation for Defense Evasion

    # Malicious execution / process-level → kill_process (8)
    'T1059': 8,   # Command-Line Interface
    'T1086': 8,   # PowerShell
    'T1064': 8,   # Scripting
    'T1047': 8,   # Windows Management Instrumentation
    'T1106': 8,   # Execution through API
    'T1129': 8,   # Execution through Module Load
    'T1035': 8,   # Service Execution
    'T1055': 8,   # Process Injection
    'T1186': 8,   # Process Doppelgänging
    'T1093': 8,   # Process Hollowing

    # File-based malware → quarantine_file (9)
    'T1204': 9,   # User Execution (malicious file)
    'T1105': 9,   # Remote File Copy (dropped payload)
    'T1140': 9,   # Deobfuscate/Decode Files
    'T1027': 9,   # Obfuscated Files or Information
    'T1500': 9,   # Compile After Delivery
    'T1223': 9,   # Compiled HTML File
    'T1221': 9,   # Template Injection

    # Phishing / email → quarantine_email (10)
    'T1192': 10,  # Spearphishing Link
    'T1193': 10,  # Spearphishing Attachment
    'T1194': 10,  # Spearphishing via Service

    # Network scanning / inbound attacks → block_source_ip (3)
    'T1046': 3,   # Network Service Scanning
    'T1498': 3,   # Network Denial of Service
    'T1499': 3,   # Endpoint Denial of Service

    # C2 / outbound → block_dest_ip (4)
    'T1071': 4,   # Standard Application Layer Protocol (C2)
    'T1095': 4,   # Standard Non-Application Layer Protocol
    'T1094': 4,   # Custom C2 Protocol
    'T1043': 4,   # Commonly Used Port (C2)
    'T1065': 4,   # Uncommonly Used Port
    'T1104': 4,   # Multi-Stage Channels
    'T1219': 4,   # Remote Access Tools
    'T1102': 4,   # Web Service (C2)
    'T1090': 4,   # Connection Proxy
    'T1008': 4,   # Fallback Channels
    'T1132': 4,   # Data Encoding (C2)

    # DNS / domain-based → dns_sinkhole (5)
    'T1483': 5,   # Domain Generation Algorithms
    'T1189': 5,   # Drive-by Compromise (watering hole)

    # Port/service-specific → block_port (6)
    'T1076': 6,   # Remote Desktop Protocol
    'T1021': 6,   # Remote Services
    'T1077': 6,   # Windows Admin Shares

    # Lateral movement host isolation → isolate_host (7)
    'T1080': 7,   # Taint Shared Content
    'T1175': 7,   # Distributed Component Object Model
    'T1210': 7,   # Exploitation of Remote Services (also 16, but lateral context)

    # Exfiltration → block_dest_ip (4)
    'T1041': 4,   # Exfiltration Over C2 Channel
    'T1048': 4,   # Exfiltration Over Alternative Protocol
    'T1020': 4,   # Automated Exfiltration

    # Impact / ransomware → isolate_host (7)
    'T1486': 7,   # Data Encrypted for Impact (ransomware)
    'T1485': 7,   # Data Destruction
    'T1487': 7,   # Disk Structure Wipe
    'T1488': 7,   # Disk Content Wipe
    'T1489': 7,   # Service Stop
    'T1490': 7,   # Inhibit System Recovery
    'T1491': 7,   # Defacement
    'T1492': 7,   # Stored Data Manipulation
    'T1493': 7,   # Transmitted Data Manipulation

    # Deep logging / discovery → enable_deep_logging (1)
    'T1082': 1,   # System Information Discovery
    'T1083': 1,   # File and Directory Discovery
    'T1057': 1,   # Process Discovery
    'T1012': 1,   # Query Registry
    'T1497': 1,   # Virtualization/Sandbox Evasion (recon)
}

# Tactic-level fallback: if no specific technique match, use the primary tactic
# Priority order: higher index = more severe → preferred action
TACTIC_ACTION_FALLBACK = {
    'TA0040': 7,   # Impact → isolate_host
    'TA0008': 7,   # Lateral Movement → isolate_host
    'TA0006': 11,  # Credential Access → reset_credentials
    'TA0004': 12,  # Privilege Escalation → disable_account
    'TA0003': 13,  # Persistence → remove_persistence
    'TA0005': 15,  # Defense Evasion → restore_defense_config
    'TA0002': 8,   # Execution → kill_process
    'TA0010': 4,   # Exfiltration → block_dest_ip
    'TA0009': 17,  # Collection → snapshot_forensics
    'TA0011': 4,   # Command & Control → block_dest_ip
    'TA0001': 3,   # Initial Access → block_source_ip
    'TA0007': 1,   # Discovery → enable_deep_logging
}

# Action severity ranking (higher = more severe intervention)
ACTION_SEVERITY = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 10,
    8: 7, 9: 8, 10: 8, 11: 9, 12: 9, 13: 8, 14: 7,
    15: 7, 16: 8, 17: 6, 18: 5, 19: 11,
}


def clean_text(text):
    """Strip boilerplate nav/footer from scraped report text and truncate."""
    if not isinstance(text, str):
        return ''
    # Remove common boilerplate patterns
    for marker in [
        'Subscribe To Our Feed', 'Blog Archive', 'Posted by',
        'Share This Post', 'Get updates from', 'Sign up to receive',
        'Copyright ©', 'Privacy Policy', 'Terms of Use',
        'Cookie', 'CAPEC List', 'CAPEC Content Team',
    ]:
        idx = text.find(marker)
        if idx > 500:  # Only cut if enough content remains
            text = text[:idx]
            break
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate to ~2000 chars (roughly 400-500 tokens for SecBERT)
    if len(text) > 2000:
        text = text[:2000]
        # Try to end at a sentence boundary
        last_period = text.rfind('. ')
        if last_period > 1500:
            text = text[:last_period + 1]
    return text


def map_action(row, tactic_cols, technique_cols):
    """Map a rcATT row (with ATT&CK labels) to one of 20 SOC response actions.

    Strategy:
      1. Check which specific techniques are flagged → look up direct mapping
      2. If multiple techniques map to different actions, pick highest severity
      3. Fall back to tactic-level mapping if no technique match
    """
    active_techniques = [t for t in technique_cols if int(row.get(t, 0)) == 1]
    active_tactics = [t for t in tactic_cols if int(row.get(t, 0)) == 1]

    # Step 1: Collect actions from technique-level mapping
    candidate_actions = []
    for tech in active_techniques:
        if tech in TECHNIQUE_ACTION_MAP:
            candidate_actions.append(TECHNIQUE_ACTION_MAP[tech])

    # Step 2: If we found technique-level actions, pick highest severity
    if candidate_actions:
        return max(candidate_actions, key=lambda a: ACTION_SEVERITY.get(a, 0))

    # Step 3: Fall back to tactic-level mapping (highest severity tactic wins)
    tactic_actions = []
    for tac in active_tactics:
        if tac in TACTIC_ACTION_FALLBACK:
            tactic_actions.append(TACTIC_ACTION_FALLBACK[tac])

    if tactic_actions:
        return max(tactic_actions, key=lambda a: ACTION_SEVERITY.get(a, 0))

    # Step 4: Default — monitor
    return 0


def build_context_text(row, tactic_cols, technique_cols):
    """Build a structured context string summarizing active ATT&CK labels."""
    active_tactics = [TACTIC_NAMES.get(t, t) for t in tactic_cols
                      if int(row.get(t, 0)) == 1]
    active_techniques = [t for t in technique_cols if int(row.get(t, 0)) == 1]

    parts = []
    if active_tactics:
        parts.append(f"Tactics: {', '.join(active_tactics)}")
    if active_techniques:
        parts.append(f"Techniques: {', '.join(active_techniques[:10])}")
    return '. '.join(parts)


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------
print('Loading rcATT dataset...')
df = pd.read_csv('data/raw/rcATT/training_data_original.csv')
print(f'Loaded {len(df)} rows, {len(df.columns)} columns')

tactic_cols = [c for c in df.columns if c.startswith('TA')]
technique_cols = [c for c in df.columns if c.startswith('T1')]

# Clean and truncate the report text
print('Cleaning text...')
df['text'] = df['Text'].apply(clean_text)

# Drop rows where cleaning left no meaningful text
df = df[df['text'].str.len() > 50].copy()
print(f'After text cleaning: {len(df)} rows')

# Build structured ATT&CK context string
df['attack_context'] = df.apply(
    lambda row: build_context_text(row, tactic_cols, technique_cols), axis=1
)

# Map to 20-action space
print('Mapping actions...')
df['action_label'] = df.apply(
    lambda row: map_action(row, tactic_cols, technique_cols), axis=1
)

# Build final output columns
output_cols = ['text', 'attack_context', 'action_label']
# Also keep tactic columns for analysis
output_cols += tactic_cols
df_out = df[output_cols].copy()

# Save
df_out.to_csv('data/processed/rcatt_processed.csv', index=False)

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
print(f'\nSaved {len(df_out)} records to data/processed/rcatt_processed.csv')
print(f'\nAction label distribution ({len(action_names)} actions):')
counts = df_out['action_label'].value_counts().sort_index()
for idx, count in counts.items():
    pct = count / len(df_out) * 100
    print(f'  {idx:2d}  {action_names[idx]:25s}  {count:5d}  ({pct:5.1f}%)')

# Show tactic coverage
print(f'\nTactic coverage:')
for t in tactic_cols:
    cnt = df_out[t].astype(int).sum()
    print(f'  {t} ({TACTIC_NAMES.get(t, "?"):25s}): {cnt:4d} reports')
