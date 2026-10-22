"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Radio,
  FileCode,
  Tag,
  Cpu,
  Layers,
  Sparkles,
  Zap,
  RotateCcw,
  Award,
  ArrowRight,
  Info,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusBadge } from "@/components/ui/StatusBadge";

interface PipelineStep {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  icon: React.ElementType;
  status: "Implemented" | "Frozen" | "Phase 3 Planned" | "Active";
  description: string;
  technicalDetails: { label: string; value: string }[];
  route?: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: "telemetry-sources",
    number: 1,
    title: "Telemetry Sources",
    subtitle: "Wazuh, OSQuery, Caldera",
    icon: Radio,
    status: "Phase 3 Planned",
    description:
      "Enterprise security sensors and automated adversary simulation. Currently disconnected; benchmark incidents are drawn from historical CISSM & rcATT corpora.",
    technicalDetails: [
      { label: "Wazuh SIEM", value: "Phase 3 Planned (Port 55000)" },
      { label: "OSQuery EDR", value: "Phase 3 Planned (/var/osquery)" },
      { label: "MITRE Caldera", value: "Phase 3 Planned (Port 8888)" },
    ],
    route: "/telemetry",
  },
  {
    id: "aggregation",
    number: 2,
    title: "Telemetry Aggregation",
    subtitle: "Deduplication & Normalization",
    icon: FileCode,
    status: "Implemented",
    description:
      "Standardizes incident fields to ['text', 'action_label', 'source_dataset']. Enforces TF-IDF near-duplicate removal (>= 0.90 cosine similarity) to prevent train/test data leakage.",
    technicalDetails: [
      { label: "Pruned Near-Duplicates", value: "2,604 records removed" },
      { label: "Stratified Split", value: "80% Train, 10% Val, 10% Test" },
      { label: "Total Clean Corpus", value: "15,390 records" },
    ],
    route: "/telemetry",
  },
  {
    id: "text-prep",
    number: 3,
    title: "Incident Text Preparation",
    subtitle: "CISA Phase Prefixing",
    icon: Tag,
    status: "Implemented",
    description:
      "Prepends the CISA incident response phase (e.g., [Detection], [Containment], [Eradication]) to alert texts. EXP_002C demonstrated this breakthrough elevated Macro F1 from 0.632 to 0.706.",
    technicalDetails: [
      { label: "Experiment", value: "EXP_20260712_006 (EXP_002C)" },
      { label: "Format", value: "[<CISA Phase>] <Alert Text>" },
      { label: "Performance Gain", value: "+0.074 Macro F1 Jump" },
    ],
    route: "/decisions",
  },
  {
    id: "secbert-encoder",
    number: 4,
    title: "Frozen SecBERT Encoder",
    subtitle: "jackaduma/SecBERT",
    icon: Cpu,
    status: "Frozen",
    description:
      "The fine-tuned Transformer encoder (EXP_20260727_001 / FP32 Ablation) loaded via AutoModel without classifier head. Parameters locked with requires_grad=False to serve as state extractor.",
    technicalDetails: [
      { label: "Checkpoint", value: "EXP_20260727_001 (FP32)" },
      { label: "Weights Status", value: "requires_grad = False (PDF-lock)" },
      { label: "Sequence Length", value: "512 tokens max" },
    ],
    route: "/evaluation",
  },
  {
    id: "state-rep",
    number: 5,
    title: "768-Dim State Embedding",
    subtitle: "[CLS] Hidden State Vector",
    icon: Layers,
    status: "Active",
    description:
      "The 768-dimensional contextual vector extracted from SecBERT's final hidden layer. Captures cybersecurity domain semantics without human-engineered feature heuristics.",
    technicalDetails: [
      { label: "Embedding Dim", value: "768 dimensions (float32)" },
      { label: "Top-2 Semantic Accuracy", value: "96.95% on test split" },
      { label: "In-Memory Cache", value: "SHA-256 Text Hashed" },
    ],
    route: "/decisions",
  },
  {
    id: "ppo-policy",
    number: 6,
    title: "PPO Policy (Actor-Critic)",
    subtitle: "MLP Policy Network",
    icon: Sparkles,
    status: "Implemented",
    description:
      "Actor network maps 768-dim state to 20 unnormalized action logits; Critic estimates scalar state value V(s). Trained with clipped surrogate objective (eps=0.2) and entropy bonus (c2=0.01).",
    technicalDetails: [
      { label: "Actor Architecture", value: "768 -> 512 -> 256 -> 20" },
      { label: "Critic Architecture", value: "768 -> 512 -> 256 -> 1" },
      { label: "Audit Verification", value: "Score: 95/100 (v1.0-ppo-core)" },
    ],
    route: "/decisions",
  },
  {
    id: "action-selection",
    number: 7,
    title: "Selected Response Action",
    subtitle: "1 of 20 Discrete Actions",
    icon: Zap,
    status: "Active",
    description:
      "Categorical distribution sampling over 20 authoritative actions derived from CISA Incident Response Playbooks and MITRE CAR (e.g., isolate_host, kill_process, dns_sinkhole).",
    technicalDetails: [
      { label: "Action Space Size", value: "20 discrete classes" },
      { label: "Playbook Source", value: "CISA Federal Response Playbooks" },
      { label: "Zero-Shot Classes", value: "Action 2 & Action 18" },
    ],
    route: "/decisions",
  },
  {
    id: "soc-env",
    number: 8,
    title: "SOC Simulation Environment",
    subtitle: "Gymnasium Contextual Bandit",
    icon: Layers,
    status: "Implemented",
    description:
      "Custom Gymnasium environment modeling incident response as a contextual bandit. Evaluates decisions in software state without destructive live network commands.",
    technicalDetails: [
      { label: "Environment Class", value: "SOCEnvironment (src/environment.py)" },
      { label: "Formulation", value: "Contextual Bandit (Per-alert step)" },
      { label: "Sequential SOC Env", value: "Phase 3 Planned" },
    ],
    route: "/simulations",
  },
  {
    id: "state-transition",
    number: 9,
    title: "Simulated State Transition",
    subtitle: "Software-State Advance",
    icon: RotateCcw,
    status: "Active",
    description:
      "Environment updates internal pointers and presents the next incident observation. Does not modify live host configuration or isolate physical nodes.",
    technicalDetails: [
      { label: "Execution Mode", value: "Software Simulated Only" },
      { label: "Live Containment", value: "Disabled (Research Safety)" },
      { label: "Episode Advancement", value: "Step-wise Contextual Bandit" },
    ],
    route: "/simulations",
  },
  {
    id: "reward-outcome",
    number: 10,
    title: "Reward & Outcome",
    subtitle: "+1.0 Match / -1.0 Mismatch",
    icon: Award,
    status: "Active",
    description:
      "Deterministic scalar feedback: +1.0 for selecting the authoritative CISA playbook response, -1.0 for operational mismatch. Recorded in PPOMemory for GAE advantage updates.",
    technicalDetails: [
      { label: "Correct Action Reward", value: "+1.0" },
      { label: "Incorrect Action Penalty", value: "-1.0" },
      { label: "Mean Rollout Reward", value: "+0.568 (PPO Test Split)" },
    ],
    route: "/evaluation",
  },
];

export function ResearchPipeline() {
  const [selectedStep, setSelectedStep] = useState<PipelineStep>(PIPELINE_STEPS[3]);

  return (
    <div className="space-y-4">
      {/* Visual Pipeline Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
        {PIPELINE_STEPS.map((step) => {
          const Icon = step.icon;
          const isSelected = selectedStep.id === step.id;

          return (
            <button
              key={step.id}
              onClick={() => setSelectedStep(step)}
              className={cn(
                "relative flex flex-col items-start p-3 rounded-lg border text-left transition-all",
                isSelected
                  ? "bg-teal-50 border-teal-500 ring-1 ring-teal-500/40 shadow-sm"
                  : "bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50"
              )}
            >
              <div className="flex items-center justify-between w-full mb-2">
                <span className="flex items-center justify-center h-5 w-5 rounded-full bg-slate-100 text-[10px] font-mono text-slate-700">
                  {step.number}
                </span>
                <span
                  className={cn("text-[9px] px-1.5 py-0.2 rounded font-medium border", {
                    "bg-teal-50 text-teal-800 border-teal-200":
                      step.status === "Implemented" || step.status === "Active",
                    "bg-sky-50 text-sky-800 border-sky-200": step.status === "Frozen",
                    "bg-slate-100 text-slate-600 border-slate-200":
                      step.status === "Phase 3 Planned",
                  })}
                >
                  {step.status}
                </span>
              </div>
              <Icon className="h-4 w-4 text-teal-600 mb-1.5" />
              <p className="text-xs font-semibold text-slate-900 line-clamp-1">{step.title}</p>
              <p className="text-[10px] text-slate-500 line-clamp-1">{step.subtitle}</p>
            </button>
          );
        })}
      </div>

      {/* Selected Component Technical Detail Panel */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-teal-50 border border-teal-200 rounded-lg text-teal-700">
              {React.createElement(selectedStep.icon, { className: "h-5 w-5" })}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-bold text-slate-900">
                  Step {selectedStep.number}: {selectedStep.title}
                </h4>
                <StatusBadge status={selectedStep.status} />
              </div>
              <p className="text-xs text-slate-500">{selectedStep.subtitle}</p>
            </div>
          </div>
          {selectedStep.route && (
            <Link
              href={selectedStep.route}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-50 hover:bg-teal-100 text-teal-800 text-xs font-medium border border-teal-200 transition-colors self-start sm:self-auto"
            >
              <span>Explore Section</span>
              <ExternalLink className="h-3 w-3" />
            </Link>
          )}
        </div>

        <p className="text-xs text-slate-600 leading-relaxed mb-4">{selectedStep.description}</p>

        {/* Technical Specification Badges */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {selectedStep.technicalDetails.map((item, idx) => (
            <div
              key={idx}
              className="p-3 rounded-lg bg-slate-50 border border-slate-200"
            >
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono block">
                {item.label}
              </span>
              <span className="text-xs font-semibold text-slate-800 mt-0.5 block font-mono">
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
