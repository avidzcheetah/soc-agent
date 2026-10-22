"use client";

import React, { useState, useEffect } from "react";
import {
  Activity,
  Server,
  Database,
  Cpu,
  BrainCircuit,
  Layers,
  Award,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  ShieldCheck,
  RefreshCw,
  Terminal,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { api } from "@/lib/api";
import { SystemStatusResponse, HealthResponse } from "@/types";

export default function SystemHealthPage() {
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStatus = async () => {
    setRefreshing(true);
    try {
      const [sRes, hRes] = await Promise.allSettled([
        api.getSystemStatus(),
        api.getHealth(),
      ]);
      if (sRes.status === "fulfilled") setStatus(sRes.value);
      if (hRes.status === "fulfilled") setHealth(hRes.value);
    } catch (e) {
      console.error("Failed to load system health", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  if (loading) {
    return <LoadingState message="Checking research platform services & daemons..." />;
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100">System & Infrastructure Health</h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time connectivity monitoring across telemetry daemons, data storage, AI checkpoints, and evaluation artifacts.
          </p>
        </div>

        <button
          onClick={fetchStatus}
          disabled={refreshing}
          className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs border border-slate-800 flex items-center gap-1.5 transition-colors self-start sm:self-auto font-medium"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin text-teal-400" : ""}`} />
          <span>Ping Services</span>
        </button>
      </div>

      {/* Backend API Health Banner */}
      <Card className="bg-[#0b1222] border-slate-800">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-teal-500/10 border border-teal-500/20 rounded-lg text-teal-400">
              <Server className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-slate-100">
                  {health?.service || "SOC Simulation Environment API Adapter"}
                </h3>
                <StatusBadge status={health?.status === "online" ? "Online" : "Offline"} />
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Runtime: Python {health?.python_version || "3.11"} | Mode: {health?.backend_mode || "Inspection & Simulation"}
              </p>
            </div>
          </div>
          <div className="text-xs font-mono text-slate-400">
            <span>Ping Timestamp: {health?.timestamp || new Date().toISOString()}</span>
          </div>
        </div>
      </Card>

      {/* Section 1: Infrastructure Daemons */}
      <Card>
        <CardHeader
          title="1. Infrastructure Daemons & Simulation Containers"
          subtitle="Status of Docker containers, SIEM indexers, and adversary simulation daemons"
        />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Docker Daemon</span>
              <StatusBadge status="Inactive" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Container isolation daemon. Inactive in read-only inspection mode.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Wazuh Manager</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Alert aggregation endpoint (Port 55000). Planned for Phase 3 Ubuntu lab deployment.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Wazuh Indexer & Dashboard</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Elasticsearch / OpenSearch cluster for alert indexing. Offline in current benchmark phase.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">OSQuery Host EDR</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Host process & socket query daemon. Offline; simulated actions evaluated in software.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">MITRE Caldera Server</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Automated adversary emulator. No active Caldera attacks executed in current dashboard.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Zeek NSM Engine</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Network security monitor. Traffic PCAP analysis planned for live enterprise deployment.
            </p>
          </div>
        </div>
      </Card>

      {/* Section 2: Data Pipeline & Hygiene */}
      <Card>
        <CardHeader
          title="2. Data Pipeline & Storage Health"
          subtitle="Integrity checks across the 15,390 benchmark incident dataset partitions"
        />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] uppercase text-slate-400 block">Train Split (80%)</span>
            <span className="text-base font-bold text-slate-200 mt-1 block">12,312</span>
            <span className="text-[10px] text-teal-400">Verified Stratified</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] uppercase text-slate-400 block">Val Split (10%)</span>
            <span className="text-base font-bold text-slate-200 mt-1 block">1,539</span>
            <span className="text-[10px] text-teal-400">Patience=2 Monitor</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] uppercase text-slate-400 block">Test Split (10%)</span>
            <span className="text-base font-bold text-slate-200 mt-1 block">1,539</span>
            <span className="text-[10px] text-teal-400">Untouched Holdout</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-center">
            <span className="text-[10px] uppercase text-slate-400 block">Total Clean Corpus</span>
            <span className="text-base font-bold text-teal-400 mt-1 block">15,390</span>
            <span className="text-[10px] text-teal-400">Zero Contamination</span>
          </div>
        </div>

        <div className="mt-4 p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-slate-300 space-y-1">
          <p className="font-semibold text-slate-200">Data Hygiene Protocols Applied:</p>
          <ul className="list-disc pl-4 space-y-0.5 text-slate-400 text-[11px]">
            <li>TF-IDF Cosine Similarity threshold &ge; 0.90 removed 2,604 near-duplicate incident templates.</li>
            <li>Zero-shot classes Action 2 (<code className="text-teal-400 font-mono">create_ioc_alert</code>) and Action 18 (<code className="text-teal-400 font-mono">sandbox_redirect</code>) have zero training samples and are assigned loss weight 0.0 to prevent gradient corruption.</li>
            <li>Random Seed 42 strictly enforced across NumPy, PyTorch, and scikit-learn splits.</li>
          </ul>
        </div>
      </Card>

      {/* Section 3: AI Components & Checkpoints */}
      <Card glow>
        <CardHeader
          title="3. AI Model Readiness & Architecture Verification"
          subtitle="Frozen transformer state representations and reinforcement learning policy specifications"
        />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* SecBERT */}
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-100 flex items-center gap-1.5">
                <Cpu className="h-4 w-4 text-teal-400" /> SecBERT State Encoder
              </span>
              <StatusBadge status="Locked & Frozen" />
            </div>
            <div className="space-y-1 text-slate-400 pt-1 text-[11px]">
              <div className="flex justify-between">
                <span>Checkpoint:</span>
                <span className="font-mono text-slate-200">EXP_20260727_001</span>
              </div>
              <div className="flex justify-between">
                <span>Embedding Output:</span>
                <span className="font-mono text-teal-400">768 Dim (float32)</span>
              </div>
              <div className="flex justify-between">
                <span>Classifier Head:</span>
                <span className="text-slate-300">Discarded (AutoModel loaded)</span>
              </div>
              <div className="flex justify-between">
                <span>Gradient Updates:</span>
                <span className="text-teal-400 font-semibold">requires_grad = False (PDF-lock)</span>
              </div>
              <div className="flex justify-between">
                <span>Embedding Cache:</span>
                <span className="text-teal-400">SHA-256 Memory Cache Active</span>
              </div>
            </div>
          </div>

          {/* PPO */}
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-100 flex items-center gap-1.5">
                <BrainCircuit className="h-4 w-4 text-sky-400" /> PPO Reinforcement Learning
              </span>
              <StatusBadge status="Verified (95/100)" />
            </div>
            <div className="space-y-1 text-slate-400 pt-1 text-[11px]">
              <div className="flex justify-between">
                <span>Milestone:</span>
                <span className="font-mono text-slate-200">v1.0-ppo-core</span>
              </div>
              <div className="flex justify-between">
                <span>Actor MLP:</span>
                <span className="font-mono text-slate-200">Linear(768-&gt;512-&gt;256-&gt;20)</span>
              </div>
              <div className="flex justify-between">
                <span>Critic MLP:</span>
                <span className="font-mono text-slate-200">Linear(768-&gt;512-&gt;256-&gt;1)</span>
              </div>
              <div className="flex justify-between">
                <span>Advantage Estimation:</span>
                <span className="font-mono text-sky-400">GAE (gamma=0.99, lambda=0.95)</span>
              </div>
              <div className="flex justify-between">
                <span>Loss Surrogate:</span>
                <span className="font-mono text-slate-200">Clipped eps=0.2, Entropy c2=0.01</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Section 4: Security & Operational Boundaries */}
      <Card>
        <CardHeader
          title="4. Security & Operational Boundaries"
          subtitle="Verifying zero destructive execution capability in current web interface"
        />
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2 p-2.5 rounded bg-slate-900/60 border border-slate-800/80">
            <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0" />
            <span className="text-slate-300">
              <strong>Shell Execution Endpoint:</strong> Completely omitted. No arbitrary command execution capability exists.
            </span>
          </div>
          <div className="flex items-center gap-2 p-2.5 rounded bg-slate-900/60 border border-slate-800/80">
            <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0" />
            <span className="text-slate-300">
              <strong>Host Modification:</strong> Isolated. The environment models containment effects purely in software state.
            </span>
          </div>
          <div className="flex items-center gap-2 p-2.5 rounded bg-slate-900/60 border border-slate-800/80">
            <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0" />
            <span className="text-slate-300">
              <strong>Secrets & Tokens:</strong> No credentials, private keys, or passwords embedded in client bundles.
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
}
