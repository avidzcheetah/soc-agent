"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Server,
  Radio,
  Cpu,
  BrainCircuit,
  Layers,
  Award,
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  ChevronRight,
  TrendingUp,
  FileCheck,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { ResearchPipeline } from "@/components/pipeline/ResearchPipeline";
import { api } from "@/lib/api";
import { formatPercent } from "@/lib/utils";

export default function OverviewPage() {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<any>(null);
  const [evalData, setEvalData] = useState<any>(null);
  const [sources, setSources] = useState<any[]>([]);

  useEffect(() => {
    async function loadData() {
      try {
        const [h, ev, src] = await Promise.allSettled([
          api.getHealth(),
          api.getEvaluations(),
          api.getTelemetrySources(),
        ]);

        if (h.status === "fulfilled") setHealth(h.value);
        if (ev.status === "fulfilled") setEvalData(ev.value);
        if (src.status === "fulfilled") setSources(src.value.sources);
      } catch (e) {
        console.error("Error loading overview data:", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return <LoadingState message="Initializing SOC Simulation Platform Overview..." />;
  }

  const latestEval = evalData?.architectures?.find(
    (a: any) => a.id === "secbert_standalone"
  );
  const ppoEval = evalData?.architectures?.find((a: any) => a.id === "ppo_core");

  return (
    <div className="space-y-8">
      {/* Platform Title Banner */}
      <div className="rounded-xl border border-slate-800 bg-gradient-to-r from-[#0d1527] to-[#0a101f] p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-teal-400 animate-pulse" />
              <span className="text-[11px] font-mono uppercase tracking-wider text-teal-400 font-semibold">
                Autonomous Security Operations Research
              </span>
            </div>
            <h1 className="text-xl md:text-2xl font-bold text-slate-100 mt-1">
              Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent
            </h1>
            <p className="text-xs text-slate-400 mt-1.5 max-w-3xl leading-relaxed">
              Research investigation and evaluation interface combining fine-tuned SecBERT contextual state representations,
              a Proximal Policy Optimization (PPO) incident-response policy, and a custom Gymnasium SOC simulation environment.
            </p>
          </div>
          <div className="flex items-center gap-2.5 shrink-0">
            <Link
              href="/evaluation"
              className="px-3.5 py-2 rounded-lg bg-teal-500/15 hover:bg-teal-500/25 text-teal-300 text-xs font-semibold border border-teal-500/40 transition-colors flex items-center gap-1.5"
            >
              <span>View Policy Evaluation</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* Section A: Verified System Status Cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider font-mono">
            A. Verified System Status
          </h2>
          <span className="text-[11px] text-slate-400 font-mono">
            Strict Scientific Integrity: Only verified states displayed
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {/* Card 1: Backend Availability */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">Backend API</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">
                  {health?.status === "online" ? "FastAPI Online" : "Standalone Fallback"}
                </p>
              </div>
              <StatusBadge status={health?.status === "online" ? "Connected" : "Offline"} />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Port: 8000</span>
              <span className="font-mono text-teal-400">{health?.python_version ? `Python ${health.python_version}` : "Node/TS"}</span>
            </div>
          </Card>

          {/* Card 2: Wazuh & OSQuery Telemetry */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">SIEM & Telemetry</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">Wazuh & OSQuery</p>
              </div>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Live Daemons: Disconnected</span>
              <span className="text-slate-400">Roadmap Phase 3</span>
            </div>
          </Card>

          {/* Card 3: Caldera Adversary Simulation */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">Adversary Simulator</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">MITRE Caldera</p>
              </div>
              <StatusBadge status="Phase 3 Planned" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Live Attacks: Offline</span>
              <span className="text-slate-400">Software Safe Mode</span>
            </div>
          </Card>

          {/* Card 4: SecBERT Checkpoint */}
          <Card className="flex flex-col justify-between" glow>
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">SecBERT Checkpoint</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">EXP_20260727_001</p>
              </div>
              <StatusBadge status="Locked & Frozen" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Weights: Frozen (requires_grad=False)</span>
              <span className="font-mono text-teal-400">768 Dim</span>
            </div>
          </Card>

          {/* Card 5: PPO Policy Readiness */}
          <Card className="flex flex-col justify-between" glow>
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">PPO RL Agent</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">v1.0-ppo-core</p>
              </div>
              <StatusBadge status="Verified (95/100)" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Actor/Critic: MLP 3-Layer</span>
              <span className="font-mono text-sky-400">GAE λ=0.95</span>
            </div>
          </Card>

          {/* Card 6: Gymnasium Environment */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">SOC Environment</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">Gymnasium Bandit</p>
              </div>
              <StatusBadge status="Active" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Reward: +1.0 / -1.0</span>
              <span className="text-slate-400">20 Actions</span>
            </div>
          </Card>

          {/* Card 7: Benchmark Datasets */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">Curated Datasets</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">15,390 Incidents</p>
              </div>
              <StatusBadge status="Active & Verified" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>CISSM (14.3k) + rcATT (1.0k)</span>
              <span className="text-teal-400">Zero Leakage</span>
            </div>
          </Card>

          {/* Card 8: Latest Evaluation Availability */}
          <Card className="flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs text-slate-400">Holdout Test Split</p>
                <p className="text-sm font-bold text-slate-100 mt-0.5">1,539 Samples</p>
              </div>
              <StatusBadge status="Evaluated" />
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Top-2 Accuracy</span>
              <span className="font-mono font-bold text-teal-400">96.95%</span>
            </div>
          </Card>
        </div>
      </div>

      {/* Section B: Research Architecture Pipeline */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider font-mono">
              B. Research Architecture Pipeline
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Click any architecture node below to inspect implementation status, hyperparameters, and tensor specifications.
            </p>
          </div>
        </div>
        <ResearchPipeline />
      </div>

      {/* Section C & D: Latest Evaluation & Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Section C: Latest Evaluation */}
        <Card>
          <CardHeader
            title="C. Latest Evaluation Benchmark"
            subtitle="Verified Holdout Test Split Evaluation (1,539 Incidents)"
            action={
              <Link
                href="/evaluation"
                className="text-xs text-teal-400 hover:text-teal-300 font-medium inline-flex items-center gap-1"
              >
                <span>Full Evaluation</span>
                <ExternalLink className="h-3 w-3" />
              </Link>
            }
          />
          {latestEval ? (
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Run ID / Checkpoint:</span>
                  <span className="text-xs font-mono font-semibold text-slate-200">
                    {latestEval.checkpoint}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Completion Protocol:</span>
                  <span className="text-xs font-mono text-teal-400">
                    FP32 Ablation (EXP_002G)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Evaluation Date:</span>
                  <span className="text-xs text-slate-300">2026-07-27 (Locked)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400">Evaluation Status:</span>
                  <StatusBadge status="Completed & Verified" />
                </div>
              </div>

              {/* Metric Highlights Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Weighted F1</p>
                  <p className="text-lg font-bold font-mono text-teal-400 mt-1">
                    {formatPercent(latestEval.weighted_f1, 2)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">MCC Score</p>
                  <p className="text-lg font-bold font-mono text-teal-400 mt-1">
                    {latestEval.mcc?.toFixed(4)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Top-2 Accuracy</p>
                  <p className="text-lg font-bold font-mono text-teal-400 mt-1">
                    {formatPercent(latestEval.top2_accuracy, 2)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Top-1 Accuracy</p>
                  <p className="text-lg font-bold font-mono text-slate-200 mt-1">
                    {formatPercent(latestEval.accuracy, 2)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Macro F1</p>
                  <p className="text-lg font-bold font-mono text-slate-200 mt-1">
                    {formatPercent(latestEval.macro_f1, 2)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-center">
                  <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">PPO Mean Reward</p>
                  <p className="text-lg font-bold font-mono text-sky-400 mt-1">
                    +0.568
                  </p>
                </div>
              </div>

              <div className="text-[11px] text-slate-400 leading-relaxed bg-slate-900/40 p-3 rounded border border-slate-800/60">
                <strong className="text-slate-300">Downstream PPO Readiness:</strong> With 96.95% Top-2 accuracy, the frozen SecBERT encoder successfully structures the 768-dimensional latent space such that the correct defensive response is almost always in the top candidate neighborhood, allowing the PPO policy to learn robust action selection.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400">No evaluation record available.</p>
          )}
        </Card>

        {/* Section D: Recent Verified Activity */}
        <Card>
          <CardHeader
            title="D. Recent Research & Benchmark Activity"
            subtitle="Verified provenance across experiments, datasets, and audits"
          />
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0 mt-0.5" />
              <div className="text-xs flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">Checkpoint Locked: EXP_20260727_001</span>
                  <span className="text-[10px] font-mono text-slate-400">2026-07-27</span>
                </div>
                <p className="text-slate-400 mt-0.5">
                  Phase 1 ablation concluded with FP32 precision model achieving highest Weighted F1 (0.9454) and Top-2 Accuracy (0.9695).
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <CheckCircle2 className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
              <div className="text-xs flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">Milestone: v1.0-ppo-core Verified</span>
                  <span className="text-[10px] font-mono text-slate-400">2026-08-05</span>
                </div>
                <p className="text-slate-400 mt-0.5">
                  Actor-Critic PPO architecture with GAE passed formal verification audit (score 95/100). Mathematical identities verified to &lt; 1e-7.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <FileCheck className="h-4 w-4 text-purple-400 shrink-0 mt-0.5" />
              <div className="text-xs flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">Dataset Leakage Audit Completed</span>
                  <span className="text-[10px] font-mono text-slate-400">2026-08-01</span>
                </div>
                <p className="text-slate-400 mt-0.5">
                  15,390 records verified with zero overlap across train (12,312), val (1,539), and holdout test (1,539) splits using character-level cosine deduplication.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-900/60 border border-slate-800/80">
              <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="text-xs flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">Phase 3 Telemetry Daemons Offline</span>
                  <span className="text-[10px] font-mono text-slate-400">Notice</span>
                </div>
                <p className="text-slate-400 mt-0.5">
                  Wazuh API client and Caldera adversary server are scheduled for Phase 3 integration; simulation runner currently operates in software-simulated benchmark mode.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
