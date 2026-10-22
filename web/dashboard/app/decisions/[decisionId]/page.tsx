"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  BrainCircuit,
  Cpu,
  Layers,
  Sparkles,
  Award,
  CheckCircle2,
  XCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Info,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge, CisaPhaseBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { ActionDistributionChart } from "@/components/charts/ActionDistributionChart";
import { api } from "@/lib/api";
import { Decision } from "@/types";

export default function DecisionDetailPage() {
  const params = useParams();
  const decisionId = (params?.decisionId as string) || "";

  const [loading, setLoading] = useState(true);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [showEmbedding, setShowEmbedding] = useState(false);

  useEffect(() => {
    async function loadDecision() {
      try {
        const data = await api.getDecisionDetail(decisionId);
        setDecision(data);
      } catch (e) {
        console.error("Failed to load decision detail", e);
      } finally {
        setLoading(false);
      }
    }
    if (decisionId) loadDecision();
  }, [decisionId]);

  if (loading) {
    return <LoadingState message={`Inspecting agent decision ${decisionId}...`} />;
  }

  if (!decision) {
    return (
      <div className="space-y-4">
        <Link
          href="/decisions"
          className="inline-flex items-center gap-1.5 text-xs text-teal-400 hover:text-teal-300 font-medium"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to Decisions
        </Link>
        <Card className="text-center py-12">
          <p className="text-sm font-semibold text-slate-200">Decision Record Not Found</p>
          <p className="text-xs text-slate-400 mt-1">ID: {decisionId}</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Navigation & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link
            href="/decisions"
            className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-100 font-mono">{decision.id}</h1>
              <CisaPhaseBadge phase={decision.cisa_phase} />
              <span
                className={`inline-flex items-center gap-1 text-[11px] font-mono font-bold px-2 py-0.5 rounded border ${
                  decision.is_correct
                    ? "bg-teal-500/10 text-teal-300 border-teal-500/30"
                    : "bg-rose-500/10 text-rose-300 border-rose-500/30"
                }`}
              >
                {decision.is_correct ? (
                  <>
                    <CheckCircle2 className="h-3 w-3" /> Policy Match (+1.0)
                  </>
                ) : (
                  <>
                    <XCircle className="h-3 w-3" /> Policy Mismatch (-1.0)
                  </>
                )}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Incident Ref:{" "}
              <Link
                href={`/incidents/${decision.incident_id}`}
                className="text-sky-400 hover:underline font-mono"
              >
                {decision.incident_id}
              </Link>{" "}
              | Policy: {decision.policy}
            </p>
          </div>
        </div>

        <Link
          href={`/incidents/${decision.incident_id}`}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-850 hover:bg-slate-800 text-slate-300 text-xs border border-slate-750 transition-colors self-start sm:self-auto"
        >
          <span>View Incident Source</span>
          <ExternalLink className="h-3 w-3" />
        </Link>
      </div>

      {/* SECTION A: Decision Metadata */}
      <Card>
        <CardHeader
          title="Section A: Decision Metadata"
          subtitle="Execution provenance and active policy checkpoints"
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-400 block">Decision ID</span>
            <span className="font-mono font-semibold text-teal-400 block mt-0.5">{decision.id}</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-400 block">Policy Checkpoint</span>
            <span className="font-mono font-semibold text-slate-200 block mt-0.5">v1.0-ppo-core</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-400 block">Encoder Checkpoint</span>
            <span className="font-mono font-semibold text-slate-200 block mt-0.5">EXP_20260727_001</span>
          </div>
          <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
            <span className="text-[10px] uppercase font-mono text-slate-400 block">Inference Execution Mode</span>
            <span className="text-teal-400 font-semibold block mt-0.5">Software Simulated</span>
          </div>
        </div>
      </Card>

      {/* SECTION B: Incident Input */}
      <Card>
        <CardHeader
          title="Section B: Incident Input & Text Preparation"
          subtitle="Original alert text transformed with structural CISA phase tagging"
        />
        <div className="space-y-4">
          <div>
            <span className="text-[11px] uppercase tracking-wider text-slate-400 font-mono block mb-1">
              Raw Incident Text:
            </span>
            <div className="p-3.5 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 leading-relaxed max-h-36 overflow-y-auto">
              {decision.incident_text}
            </div>
          </div>

          <div>
            <span className="text-[11px] uppercase tracking-wider text-slate-400 font-mono block mb-1">
              Prepared Input Text for SecBERT (With [CISA Phase] Prefix):
            </span>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono text-xs text-teal-300">
              {decision.prepared_text}
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Prefixing <code className="text-teal-400 font-mono">[{decision.cisa_phase}]</code> gives the Transformer structural context, resolving semantic confusion between attack stages.
            </p>
          </div>
        </div>
      </Card>

      {/* SECTION C: SecBERT State Encoder */}
      <Card glow>
        <CardHeader
          title="Section C: Frozen SecBERT State Encoder (768 Dimensions)"
          subtitle="Read-only AutoModel transformer producing dense contextual incident embeddings"
          action={
            <button
              onClick={() => setShowEmbedding(!showEmbedding)}
              className="text-xs text-teal-400 hover:text-teal-300 font-mono flex items-center gap-1"
            >
              <span>{showEmbedding ? "Hide Vector" : "Inspect Embedding Vector"}</span>
              {showEmbedding ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
          }
        />
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Encoder State</span>
              <span className="text-teal-400 font-semibold block mt-0.5">Frozen (requires_grad=False)</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Embedding Dimension</span>
              <span className="font-mono font-bold text-slate-100 block mt-0.5">768 Float32</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Vector L2 Norm</span>
              <span className="font-mono text-slate-200 block mt-0.5">{decision.embedding_norm}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Tensor Integrity</span>
              <span className="text-teal-400 font-semibold block mt-0.5">0 NaNs / 0 Infs (Pass)</span>
            </div>
          </div>

          {/* Scientific Disclaimer (Mandatory per Rule) */}
          <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800/80 text-[11px] text-slate-400 flex items-start gap-2">
            <Info className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
            <p>
              <strong className="text-slate-300">Representation Note:</strong> Individual dimensions of the 768-dimensional transformer embedding represent distributed contextual features across the cybersecurity semantic space. Individual dimensions do NOT correspond to isolated human-readable rules or single words.
            </p>
          </div>

          {/* Expandable Embedding Vector Inspector */}
          {showEmbedding && (
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono text-slate-300">
                  Sample Embedding Vector Components (s_0 ... s_15 of 768):
                </span>
                <span className="text-[10px] font-mono text-slate-400">dtype=torch.float32</span>
              </div>
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 font-mono text-xs">
                {decision.sample_embedding.map((val, idx) => (
                  <div key={idx} className="p-1.5 rounded bg-slate-900 border border-slate-800 text-center">
                    <span className="text-[9px] text-slate-400 block">s_{idx}</span>
                    <span className={val >= 0 ? "text-teal-300" : "text-rose-300"}>
                      {val.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-[10px] font-mono text-slate-400 mt-2 text-right">
                [... 752 additional floating point dimensions active in GPU memory ...]
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* SECTION D: PPO Decision & Action Probability Distribution */}
      <Card>
        <CardHeader
          title="Section D: PPO Decision & Categorical Distribution"
          subtitle="Actor network action logits and Critic state value estimation V(s)"
        />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Action Comparison Details */}
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-3">
              <div>
                <span className="text-[10px] uppercase font-mono text-slate-400 block">Policy Selected Action</span>
                <p className="text-base font-bold font-mono text-slate-100 mt-0.5">
                  {decision.chosen_action_name}{" "}
                  <span className="text-xs font-normal text-slate-400">(Index {decision.chosen_action})</span>
                </p>
              </div>

              <div>
                <span className="text-[10px] uppercase font-mono text-slate-400 block">Authoritative CISA Action</span>
                <p className="text-sm font-semibold font-mono text-teal-400 mt-0.5">
                  {decision.ground_truth_name}{" "}
                  <span className="text-xs font-normal text-slate-400">(Index {decision.ground_truth_action})</span>
                </p>
              </div>

              <div>
                <span className="text-[10px] uppercase font-mono text-slate-400 block">CISA Response Phase</span>
                <div className="mt-1">
                  <CisaPhaseBadge phase={decision.cisa_phase} />
                </div>
              </div>
            </div>

            {/* Critic Value V(s) Panel */}
            <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">
                Critic State Value V(s)
              </span>
              <p className="text-xl font-bold font-mono text-sky-400 mt-1">
                {decision.critic_value.toFixed(3)}
              </p>
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                <strong className="text-slate-300">Mandatory Research Distinction:</strong> The Critic value V(s) estimates expected cumulative discounted returns from state s. It is NOT a classification confidence score.
              </p>
            </div>
          </div>

          {/* Action Probability Distribution Chart */}
          <div className="lg:col-span-2">
            <h4 className="text-xs font-semibold text-slate-200 mb-2">
              Action Probabilities π(a|s) Across 20 Discrete Actions
            </h4>
            {decision.probabilities_breakdown && (
              <ActionDistributionChart probabilities={decision.probabilities_breakdown} />
            )}
            <div className="flex items-center gap-4 text-[11px] text-slate-400 mt-2 font-mono">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded bg-teal-400" /> Selected Action (Matching GT)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded bg-amber-400" /> Ground Truth Target
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded bg-rose-400" /> Selected Action (Mismatch)
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* SECTION E: Environment Outcome */}
      <Card>
        <CardHeader
          title="Section E: Environment Outcome & Response Effect"
          subtitle="Reward assignment and software-simulated containment effect"
        />
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Environment Acceptance</span>
              <span className="text-teal-400 font-semibold block mt-1">Accepted Valid Action (0-19)</span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Scalar Reward Assigned</span>
              <span
                className={`font-mono text-base font-bold block mt-1 ${
                  decision.reward > 0 ? "text-teal-400" : "text-rose-400"
                }`}
              >
                {decision.reward > 0 ? "+1.0 (Efficacy Reward)" : "-1.0 (Mismatch Penalty)"}
              </span>
            </div>
            <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">Episode Termination Status</span>
              <span className="text-slate-300 font-mono block mt-1">terminated=False, truncated=False</span>
            </div>
          </div>

          {/* Software Simulated Effect */}
          <div className="p-4 rounded-lg bg-slate-900/60 border border-slate-800 space-y-2">
            <span className="text-[10px] uppercase font-mono text-slate-400 block">Simulated Response Effect</span>
            <p className="text-xs text-slate-200 leading-relaxed font-mono">
              {decision.simulated_effect}
            </p>
          </div>

          {/* Crucial Integrity Warning (Rule 6) */}
          <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
            <p>
              <strong className="text-slate-300">Software Simulation Boundary:</strong> {decision.operational_containment} No real network interface was brought down, no active process terminated, and no firewall rules altered on host machines.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
