import pandas as pd
import random
import os

# Base unaugmented data
train_path = 'data/processed/train.csv'
out_path = 'data/processed/train_augmented_v2.csv'

df = pd.read_csv(train_path)

# 2: create_ioc_alert
# 6: block_port
# 14: restore_registry
# 17: snapshot_forensics
# 18: sandbox_redirect
# 19: escalate_to_human

synthetic_data = []

def generate_samples(label, templates, count=50):
    for i in range(count):
        text = random.choice(templates).format(i=i, r1=random.randint(100,999), r2=random.randint(1000,9999))
        synthetic_data.append({'text': text, 'action_label': label})

# Class 2: create_ioc_alert
templates_2 = [
    "Threat intel match for {r1}.{r1}.{r1}.{r1}. Extracting IOCs and pushing to Wazuh.",
    "New C2 domain detected from CrowdStrike CTI. Updating IOC alert rules with ID {r2}.",
    "Suspicious hash {r1}ab{r2} matched VirusTotal. Creating new IOC alert.",
    "Ingesting fresh IOC feeds from MISP. Generating correlation alert for IP {r1}.{r2}.10.5.",
    "Detected unauthorized file signature. Adding IOC to enterprise SIEM blocklist rule {r2}."
]
generate_samples(2, templates_2, 50)

# Class 6: block_port
templates_6 = [
    "Unauthorized SMB scan detected. Blocking port 445 on firewall rule {r2}.",
    "Excessive RDP traffic observed from external IP. Blocking port 3389 at perimeter.",
    "Closing port {r2} after detecting exploit attempt on non-standard service.",
    "Firewall updated: deny all inbound to port 22 on host 10.0.{r1}.{r1}.",
    "Blocking port 1433 due to suspected SQL injection scanning."
]
generate_samples(6, templates_6, 50)

# Class 14: restore_registry
templates_14 = [
    "Malicious AutoRun key detected. Reverting registry HKLM/Software/Microsoft/Windows/CurrentVersion/Run.",
    "Registry tampering on security center settings. Restoring known-good baseline config {r2}.",
    "Deleted malicious Image File Execution Options (IFEO) registry key.",
    "Restoring altered registry values under HKLM/System/CurrentControlSet/Services after suspected malware execution.",
    "Rollback of suspicious registry modification involving PowerShell execution policy."
]
generate_samples(14, templates_14, 50)

# Class 17: snapshot_forensics
templates_17 = [
    "High severity ransomware incident detected. Triggering memory dump for forensic analysis on host {r2}.",
    "Preserving disk image for forensic evidence on compromised server {r1}.",
    "Initiating forensic snapshot of EC2 instance i-0abcd{r2} following C2 beaconing alert.",
    "Capturing volatile memory state for host {r1} due to suspected advanced persistent threat.",
    "Freezing host and performing forensic data acquisition per incident response playbook."
]
generate_samples(17, templates_17, 50)

# Class 18: sandbox_redirect
templates_18 = [
    "Advanced adversary detected. Redirecting traffic to isolated honeypot environment {r2}.",
    "Shifting suspicious session to sandbox VLAN for TTP extraction and observation.",
    "Deception network activated. Routing attacker IP {r1}.{r1}.x to sandbox.",
    "Suspected zero-day exploit attempt. Redirecting payload to detonation sandbox.",
    "Engaging active defense: rerouting attacker connection to decoy server."
]
generate_samples(18, templates_18, 50)

# Class 19: escalate_to_human
templates_19 = [
    "Critical asset compromised. Freezing automation and escalating to Tier-3 analyst.",
    "High-uncertainty incident involving domain controller. Flagging for immediate human review.",
    "Major incident declared. Initiating CISA notification and escalating to incident response lead.",
    "Automated containment failed. Escalating incident ticket {r2} to senior SOC personnel.",
    "Complex multi-stage attack detected on VIP account. Escalate to human immediately."
]
generate_samples(19, templates_19, 50)

df_synthetic = pd.DataFrame(synthetic_data)
df_augmented = pd.concat([df, df_synthetic], ignore_index=True)

df_augmented.to_csv(out_path, index=False)
print(f"Augmented dataset saved to {out_path} with {len(df_augmented)} rows (added {len(synthetic_data)} synthetic samples).")
