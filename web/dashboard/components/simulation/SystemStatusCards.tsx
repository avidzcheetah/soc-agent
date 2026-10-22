"use client";

import React, { useState, useEffect } from "react";
import {
  Cpu,
  BrainCircuit,
  Layers,
  Network,
  Server,
  AlertCircle,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";

interface StatusCardProps {
  title: string;
  subtitle: string;
  status: string;
  icon: React.ElementType;
  details: string;
  badgeColor?: "amber" | "slate" | "teal" | "rose" | "emerald";
  footerText?: string;
}

function StatusCard({
  title,
  subtitle,
  status,
  icon: Icon,
  details,
  badgeColor = "amber",
  footerText = "Verified Status",
}: StatusCardProps) {
  const badgeStyles = {
    amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    slate: "bg-slate-800 text-slate-300 border-slate-700",
    teal: "bg-teal-500/10 text-teal-300 border-teal-500/30",
    rose: "bg-rose-500/10 text-rose-300 border-rose-500/30",
    emerald: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
  }[badgeColor];

  return (
    <Card className="p-4 flex flex-col justify-between border-slate-800/80 bg-slate-900/60 hover:border-slate-700/80 transition-colors">
      <div className="space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
              <Icon className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-xs font-semibold text-slate-100">{title}</h3>
              <p className="text-[10px] text-slate-400 font-mono">{subtitle}</p>
            </div>
          </div>
          <span
            className={`inline-flex items-center gap-1 text-[10px] font-mono font-medium px-2 py-0.5 rounded border ${badgeStyles}`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
            {status}
          </span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">{details}</p>
      </div>

      <div className="pt-2.5 mt-2.5 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-400">
        <span>{footerText}</span>
        <span className="font-mono text-slate-400">Step 3 Verified</span>
      </div>
    </Card>
  );
}

export function SystemStatusCards() {
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [torchReady, setTorchReady] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    api
      .getHealth()
      .then((h) => {
        if (mounted) {
          setBackendOnline(h.status === "online");
          setTorchReady(Boolean(h.torch_available));
        }
      })
      .catch(() => {
        if (mounted) setBackendOnline(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const cards: StatusCardProps[] = [
    {
      title: "SecBERT Encoder",
      subtitle: "jackaduma/SecBERT",
      status: "Missing Checkpoint",
      icon: Cpu,
      details: "Trained weights directory 'models/secbert_finetuned' is uncommitted on disk. PyTorch runtime not installed in environment.",
      badgeColor: "amber",
      footerText: "Real Inference Unavailable",
    },
    {
      title: "PPO Policy",
      subtitle: "v1.0-ppo-core",
      status: "Missing Checkpoint",
      icon: BrainCircuit,
      details: "Actor-Critic code verified (95/100 audit score), but trained weights 'models/ppo/best_ppo_policy.pt' are missing from disk.",
      badgeColor: "amber",
      footerText: "Real Inference Unavailable",
    },
    {
      title: "PPO+ATGP Policy",
      subtitle: "Teacher Guided (EXP4D)",
      status: "Missing Checkpoint",
      icon: Layers,
      details: "Experimental distillation checkpoint missing from 'models/ppo_exp4d_atgp/'. Algorithm specified in protocol.",
      badgeColor: "amber",
      footerText: "Real Inference Unavailable",
    },
    {
      title: "SOC Environment",
      subtitle: "Gymnasium Contextual Bandit",
      status: "Historical Replay Ready",
      icon: Network,
      details: "Contextual bandit verified over 1,539 holdout test incidents (data/processed/test.csv). Evaluates ground-truth CISA action rewards (+1.0 / -1.0).",
      badgeColor: "teal",
      footerText: "1,539 Test Incidents Ready",
    },
    {
      title: "Backend API Service",
      subtitle: "Python / FastAPI",
      status: backendOnline ? "Online (Port 8000)" : "Disconnected",
      icon: Server,
      details: backendOnline
        ? "FastAPI inspection & historical replay service online. SSE streaming active on port 8000."
        : "FastAPI server offline. Start with 'python -m uvicorn api.server:app'.",
      badgeColor: backendOnline ? "emerald" : "rose",
      footerText: backendOnline ? "127.0.0.1:8000 Connected" : "Connection Failed",
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Server className="h-3.5 w-3.5 text-teal-400" />
          <span>System & Model Readiness Status (Scientific Integrity Audit)</span>
        </h2>
        <span className="text-[11px] text-amber-300/90 flex items-center gap-1 font-mono bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
          <AlertTriangle className="h-3 w-3 text-amber-400" />
          <span>Checkpoints Missing — Engine running in Benchmark Ground-Truth Replay Mode</span>
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {cards.map((card) => (
          <StatusCard key={card.title} {...card} />
        ))}
      </div>
    </div>
  );
}
