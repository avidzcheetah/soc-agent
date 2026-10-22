"use client";

import React from "react";
import Link from "next/link";
import {
  Network,
  ArrowLeft,
  Cpu,
  Layers,
  BrainCircuit,
  Award,
  ShieldCheck,
  Zap,
  Info,
} from "lucide-react";
import { ArchitectureFlow } from "@/components/simulation/ArchitectureFlow";
import { Card } from "@/components/ui/Card";

export default function ArchitecturePage() {
  return (
    <div className="space-y-8 pb-16">
      {/* Header */}
      <div className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-900/90 via-[#0a1122]/80 to-slate-900/90 p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Link
                href="/"
                className="inline-flex items-center gap-1 text-xs text-teal-400 hover:text-teal-300 font-medium"
              >
                <ArrowLeft className="h-3.5 w-3.5" /> Back to Simulation Control
              </Link>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
              System Architecture &amp; Execution Pipeline
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 max-w-3xl leading-relaxed">
              Interactive 8-stage execution flow connecting incident preprocessing, frozen SecBERT 768-dimensional representation learning, and PPO / PPO+ATGP policy decision-making in a Gymnasium environment.
            </p>
          </div>

          <span className="px-2.5 py-1 rounded bg-teal-500/10 border border-teal-500/30 text-teal-300 font-mono text-xs self-start sm:self-auto">
            React Flow (@xyflow/react)
          </span>
        </div>
      </div>

      {/* Main React Flow Canvas */}
      <ArchitectureFlow />

      {/* Deep-Dive Architectural Specifications */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Layers className="h-4 w-4 text-teal-400" />
            <span>Architectural Layer Specifications</span>
          </h2>
          <span className="text-[11px] font-mono text-slate-400">Department of Computer Engineering</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <Card className="p-4 space-y-2 border-slate-800 bg-slate-900/60">
            <div className="flex items-center gap-2 text-teal-400 font-semibold">
              <Cpu className="h-4 w-4" />
              <span>Layer 1: SecBERT Representation Learning (768-dim)</span>
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Fine-tuned on 18,218 cybersecurity incidents from CISSM and rcATT benchmarks. Pre-trained with semantic CISA phase tags (<code className="text-teal-300">[Detection]</code>, <code className="text-teal-300">[Containment]</code>, <code className="text-teal-300">[Eradication]</code>) which resolved severe class ambiguity and boosted Macro F1 to 0.706. The classification head is discarded during RL, and the frozen encoder produces dense, rich 768-dimensional state vectors.
            </p>
          </Card>

          <Card className="p-4 space-y-2 border-slate-800 bg-slate-900/60">
            <div className="flex items-center gap-2 text-teal-400 font-semibold">
              <BrainCircuit className="h-4 w-4" />
              <span>Layer 2: PPO &amp; PPO+ATGP Decision Policy</span>
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Actor network (<code className="text-teal-300">Linear(768→512→256→20)</code>) maps state representations to categorical response action distributions. Critic network (<code className="text-teal-300">Linear(768→512→256→1)</code>) estimates baseline expected return. GAE (<code className="text-teal-300">γ=0.99, λ=0.95</code>) and clipped surrogate loss (<code className="text-teal-300">ε=0.2</code>) guarantee stable policy optimization.
            </p>
          </Card>

          <Card className="p-4 space-y-2 border-slate-800 bg-slate-900/60">
            <div className="flex items-center gap-2 text-amber-400 font-semibold">
              <ShieldCheck className="h-4 w-4" />
              <span>Layer 3: 20-Action Canonical Space</span>
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Derived from the CISA Federal Incident Response Playbooks (containment steps 7a–7n, eradication steps 8a–8m). Actions are ordered by increasing intervention severity to enable proportional response reward shaping (monitor, deep logging, block IP, isolate host, reset credentials, patch vulnerability).
            </p>
          </Card>

          <Card className="p-4 space-y-2 border-slate-800 bg-slate-900/60">
            <div className="flex items-center gap-2 text-amber-400 font-semibold">
              <Award className="h-4 w-4" />
              <span>Layer 4: Gymnasium Contextual Bandit Environment</span>
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Formulated as an independent decision point for each alert (reward +1.0 on match, -1.0 on mismatch) without dictating the arrival distribution or semantic content of subsequent uncorrelated alerts. Enforces a strict software containment boundary with no destructive live host modifications.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
