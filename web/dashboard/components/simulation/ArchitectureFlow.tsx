"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Edge,
  Position,
  Handle,
  MarkerType,
  BackgroundVariant,
  NodeProps,
} from "@xyflow/react";
import {
  ShieldAlert,
  Network,
  FileText,
  Cpu,
  BrainCircuit,
  Zap,
  Award,
  RefreshCw,
  Info,
  ChevronRight,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Activity,
  Layers,
  Radio,
} from "lucide-react";
import { Card } from "@/components/ui/Card";

export type NodeLifecycleState =
  | "Idle"
  | "Processing"
  | "Completed"
  | "Waiting"
  | "Unavailable"
  | "Error";

// Node metadata for the side details panel and architecture diagram
export interface NodeMeta {
  id: string;
  step: number;
  label: string;
  subtitle: string;
  description: string;
  status: string;
  fileLocation: string;
  inputs: string[];
  outputs: string[];
  technicalDetails: string;
  statusColor: "amber" | "slate" | "teal" | "rose" | "emerald";
  lifecycleState: NodeLifecycleState;
  stateExplanation?: string;
}

export const NODE_DETAILS: Record<string, NodeMeta> = {
  "1": {
    id: "1",
    step: 1,
    label: "Benchmark Incidents",
    subtitle: "Historical Telemetry Replay",
    description: "Traverses historical cyber incident records from data/processed/test.csv (1,539 incidents).",
    status: "Dataset Ready",
    fileLocation: "data/processed/test.csv",
    inputs: ["Scenario Filter", "Target Subnet", "Random Seed"],
    outputs: ["Raw Alert Telemetry", "Ground-Truth CISA Action"],
    technicalDetails: "Replays real security incidents with raw alert telemetry, CISA action labels, and semantic phase annotations.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
  "2": {
    id: "2",
    step: 2,
    label: "SOC Environment",
    subtitle: "Gymnasium Contextual Bandit",
    description: "Coordinates incident delivery, evaluates action alignment against CISA ground-truth, and returns reward.",
    status: "Environment Ready",
    fileLocation: "src/environment.py:SOCEnvironment",
    inputs: ["Mitigation Action (0-19)", "Current Attack State"],
    outputs: ["Incident Text", "Reward (+1.0 / -1.0)", "Terminated / Truncated"],
    technicalDetails: "Contextual bandit formulation with reset() and step(). Evaluates mitigation action alignment without host-level containment modeling.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
  "3": {
    id: "3",
    step: 3,
    label: "Incident Preprocessing",
    subtitle: "Text Preprocessing & Tagging",
    description: "Cleans raw telemetry and prepends semantic CISA response phase prefixes [Detection], [Containment], [Eradication].",
    status: "Active Preprocessor",
    fileLocation: "src/environment.py:_prepare_text",
    inputs: ["Raw Alert Text", "CISA Playbook Action Label"],
    outputs: ["Prepared Input Text String (≤512 tokens)"],
    technicalDetails: "EXP_002C discovery: Semantic phase prefixing resolved critical class ambiguities, boosting Macro F1 to 0.706.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
  "4": {
    id: "4",
    step: 4,
    label: "SecBERT Encoder",
    subtitle: "Frozen Transformer State Model",
    description: "Encodes structured incident descriptions into dense 768-dimensional contextual state embeddings.",
    status: "Unavailable",
    fileLocation: "src/encoder.py:SecBERTStateEncoder",
    inputs: ["Tokenized Incident String"],
    outputs: ["768-dim State Embedding Tensor s ∈ R^768"],
    technicalDetails: "Architecture designed for fine-tuned SecBERT (EXP_20260727_001). Weights are uncommitted on disk; engine runs in honest benchmark replay mode.",
    statusColor: "amber",
    lifecycleState: "Unavailable",
    stateExplanation: "Weights uncommitted on disk; fallback to benchmark replay without synthetic vectors.",
  },
  "5": {
    id: "5",
    step: 5,
    label: "PPO Policy Network",
    subtitle: "Actor-Critic Network",
    description: "Maps 768-dim state to action logits π(a|s) [20] and state value baseline V(s) [1].",
    status: "Unavailable",
    fileLocation: "src/ppo/agent.py & src/ppo/networks.py",
    inputs: ["768-dim State Vector s"],
    outputs: ["Action Probabilities π(a|s) [20]", "State Value V(s) [1]"],
    technicalDetails: "PPO/PPO+ATGP weights uncommitted on disk. Engine evaluates canonical ground-truth CISA actions without fabricating fake probabilities.",
    statusColor: "amber",
    lifecycleState: "Unavailable",
    stateExplanation: "Weights uncommitted on disk; canonical actions evaluated without fabricated probabilities or critic values.",
  },
  "6": {
    id: "6",
    step: 6,
    label: "Response Action",
    subtitle: "1 of 20 CISA Canonical Actions",
    description: "Categorical response action selected from canonical CISA action space (isolate host, block IP, quarantine email, etc.).",
    status: "Action Space Ready",
    fileLocation: "data/action_space.csv",
    inputs: ["Policy / Ground-Truth Selection"],
    outputs: ["Discrete Action Index a ∈ {0, ..., 19}"],
    technicalDetails: "Derived from CISA Federal Playbooks. Actions ordered by increasing severity for proportional mitigation shaping.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
  "7": {
    id: "7",
    step: 7,
    label: "Simulated Effect & Reward",
    subtitle: "Contextual Bandit Reward",
    description: "Evaluates action efficacy against ground-truth playbook; calculates instantaneous reward and penalty.",
    status: "Bandit Evaluator Ready",
    fileLocation: "src/environment.py:step",
    inputs: ["Chosen Action Index", "Ground Truth Action Label"],
    outputs: ["Reward r_t ∈ {+1.0, -1.0}", "Mitigation Status"],
    technicalDetails: "Immediate reward: +1.0 for action matching ground-truth, -1.0 otherwise. Contextual bandit formulation does not simulate host containment.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
  "8": {
    id: "8",
    step: 8,
    label: "Updated State & Metrics",
    subtitle: "Step Advance & Return Tracking",
    description: "Advances to next historical test incident, updates cumulative return Σ r_t, and tracks action matching accuracy.",
    status: "Metrics Ready",
    fileLocation: "src/environment.py & api/simulation_engine.py",
    inputs: ["Transition Step Outcome", "Reward Feedback"],
    outputs: ["Cumulative Reward Σ r_t", "Action Accuracy"],
    technicalDetails: "Maintains running accuracy and cumulative reward across sequential incident replay without hallucinated containment metrics.",
    statusColor: "teal",
    lifecycleState: "Idle",
  },
};

// Custom React Flow Node Component with full lifecycle states and accessibility
function CustomArchitectureNode({ data, selected }: NodeProps) {
  const meta: NodeMeta = data.meta as NodeMeta;
  const isSelected = selected || Boolean(data.isSelected);
  const isActive = Boolean(data.isActive);
  const lifecycle: NodeLifecycleState = (data.lifecycleState as NodeLifecycleState) || meta.lifecycleState || "Idle";

  const getLifecycleBadge = () => {
    switch (lifecycle) {
      case "Unavailable":
        return {
          bg: "bg-amber-500/15 text-amber-300 border-amber-500/40",
          icon: AlertTriangle,
          label: "Unavailable",
          dotColor: "bg-amber-400",
        };
      case "Processing":
        return {
          bg: "bg-teal-500/20 text-teal-300 border-teal-500/50",
          icon: Activity,
          label: "Processing",
          dotColor: "bg-teal-400 animate-ping",
        };
      case "Completed":
        return {
          bg: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
          icon: CheckCircle2,
          label: "Completed",
          dotColor: "bg-emerald-400",
        };
      case "Waiting":
        return {
          bg: "bg-sky-500/15 text-sky-300 border-sky-500/30",
          icon: Clock,
          label: "Waiting",
          dotColor: "bg-sky-400",
        };
      case "Error":
        return {
          bg: "bg-rose-500/20 text-rose-300 border-rose-500/40",
          icon: AlertTriangle,
          label: "Error",
          dotColor: "bg-rose-400",
        };
      case "Idle":
      default:
        return {
          bg: "bg-slate-800/80 text-slate-400 border-slate-700/60",
          icon: Clock,
          label: "Idle",
          dotColor: "bg-slate-500",
        };
    }
  };

  const badge = getLifecycleBadge();
  const BadgeIcon = badge.icon;

  const icons = {
    "1": ShieldAlert,
    "2": Network,
    "3": FileText,
    "4": Cpu,
    "5": BrainCircuit,
    "6": Zap,
    "7": Award,
    "8": RefreshCw,
  };
  const Icon = icons[meta.id as keyof typeof icons] || Network;

  // Determine container styling based on active/selected/lifecycle
  let containerStyle = "bg-[#0a1122]/95 border-slate-800 hover:border-slate-700 shadow-md";
  if (isActive) {
    containerStyle = "bg-[#0b1528] border-teal-400 shadow-xl shadow-teal-500/30 ring-2 ring-teal-400 animate-pulse";
  } else if (isSelected) {
    containerStyle = "bg-[#0b1528] border-teal-400 shadow-lg shadow-teal-500/20 ring-1 ring-teal-400";
  } else if (lifecycle === "Unavailable") {
    containerStyle = "bg-[#0f1118]/95 border-amber-900/40 hover:border-amber-700/50 shadow-md";
  } else if (lifecycle === "Completed") {
    containerStyle = "bg-[#071318]/95 border-emerald-900/40 hover:border-emerald-700/50 shadow-md";
  }

  return (
    <div
      className={`w-64 p-3.5 rounded-xl border text-left transition-all backdrop-blur ${containerStyle}`}
      role="region"
      aria-label={`Stage ${meta.step}: ${meta.label} - Status: ${badge.label}`}
    >
      <Handle
        type="target"
        position={meta.step <= 4 ? Position.Left : Position.Right}
        className="!bg-teal-400 !border-slate-900 !w-2.5 !h-2.5"
      />

      <div className="space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div
              className={`p-1.5 rounded-lg border ${
                lifecycle === "Unavailable"
                  ? "bg-amber-950/40 border-amber-800/40 text-amber-400"
                  : lifecycle === "Completed"
                  ? "bg-emerald-950/40 border-emerald-800/40 text-emerald-400"
                  : "bg-slate-900 border-slate-800 text-teal-400"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] font-mono font-bold text-slate-400">
                  STAGE {meta.step}
                </span>
                {lifecycle === "Unavailable" && (
                  <span className="text-[9px] font-mono text-amber-400 font-semibold">
                    (UNCOMMITTED)
                  </span>
                )}
              </div>
              <h4 className="text-xs font-bold text-slate-100 leading-tight">
                {meta.label}
              </h4>
            </div>
          </div>
        </div>

        <p className="text-[10px] text-slate-400 line-clamp-2 leading-relaxed">
          {meta.description}
        </p>

        {meta.stateExplanation && lifecycle === "Unavailable" && (
          <p className="text-[9px] text-amber-400/90 font-mono bg-amber-950/30 p-1 rounded border border-amber-800/30 line-clamp-2 leading-tight">
            ⚠ {meta.stateExplanation}
          </p>
        )}

        <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between">
          <span
            className={`inline-flex items-center gap-1 text-[9px] font-mono px-1.5 py-0.5 rounded border ${badge.bg}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${badge.dotColor}`} />
            <BadgeIcon className="h-2.5 w-2.5" />
            <span>{badge.label}</span>
          </span>
          <span className="text-[9px] font-mono text-slate-400">
            {meta.subtitle.split(" ")[0]}
          </span>
        </div>
      </div>

      <Handle
        type="source"
        position={meta.step <= 4 ? Position.Right : Position.Left}
        className="!bg-teal-400 !border-slate-900 !w-2.5 !h-2.5"
      />
    </div>
  );
}

const nodeTypes = {
  customNode: CustomArchitectureNode,
};

// Initial layout in 2 clean rows of 4 nodes
const initialNodes: Node[] = [
  // Row 1 (Left to Right: Steps 1 to 4)
  {
    id: "1",
    type: "customNode",
    position: { x: 30, y: 40 },
    data: { meta: NODE_DETAILS["1"], lifecycleState: "Idle" },
  },
  {
    id: "2",
    type: "customNode",
    position: { x: 320, y: 40 },
    data: { meta: NODE_DETAILS["2"], lifecycleState: "Idle" },
  },
  {
    id: "3",
    type: "customNode",
    position: { x: 610, y: 40 },
    data: { meta: NODE_DETAILS["3"], lifecycleState: "Idle" },
  },
  {
    id: "4",
    type: "customNode",
    position: { x: 900, y: 40 },
    data: { meta: NODE_DETAILS["4"], lifecycleState: "Unavailable" },
  },

  // Row 2 (Right to Left: Steps 5 to 8)
  {
    id: "5",
    type: "customNode",
    position: { x: 900, y: 220 },
    data: { meta: NODE_DETAILS["5"], lifecycleState: "Unavailable" },
  },
  {
    id: "6",
    type: "customNode",
    position: { x: 610, y: 220 },
    data: { meta: NODE_DETAILS["6"], lifecycleState: "Idle" },
  },
  {
    id: "7",
    type: "customNode",
    position: { x: 320, y: 220 },
    data: { meta: NODE_DETAILS["7"], lifecycleState: "Idle" },
  },
  {
    id: "8",
    type: "customNode",
    position: { x: 30, y: 220 },
    data: { meta: NODE_DETAILS["8"], lifecycleState: "Idle" },
  },
];

const createEdges = (activeStage: number | null): Edge[] => {
  // Define transitions: (1->2), (2->3), (3->4), (4->5), (5->6), (6->7), (7->8), (8->2)
  const transitions = [
    { id: "e1-2", source: "1", target: "2", activeAt: 2 },
    { id: "e2-3", source: "2", target: "3", activeAt: 3 },
    { id: "e3-4", source: "3", target: "4", activeAt: 4 },
    { id: "e4-5", source: "4", target: "5", activeAt: 5 },
    { id: "e5-6", source: "5", target: "6", activeAt: 6 },
    { id: "e6-7", source: "6", target: "7", activeAt: 7 },
    { id: "e7-8", source: "7", target: "8", activeAt: 8 },
  ];

  const standardEdges: Edge[] = transitions.map((t) => {
    const isEdgeActive = activeStage === t.activeAt;
    const isCompleted = activeStage !== null && activeStage > t.activeAt;

    let strokeColor = "#334155"; // Muted slate by default
    let strokeWidth = 2;
    let isAnimated = false;

    if (isEdgeActive) {
      strokeColor = "#14b8a6"; // Vibrant teal for active edge
      strokeWidth = 3;
      isAnimated = true;
    } else if (isCompleted) {
      strokeColor = "#10b981"; // Emerald for completed path
      strokeWidth = 2;
      isAnimated = false;
    }

    return {
      id: t.id,
      source: t.source,
      target: t.target,
      animated: isAnimated,
      style: { stroke: strokeColor, strokeWidth },
      markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor },
    };
  });

  // State update loop from 8 back to 2
  const isLoopActive = activeStage === 8;
  standardEdges.push({
    id: "e8-2",
    source: "8",
    target: "2",
    animated: isLoopActive,
    style: {
      stroke: isLoopActive ? "#38bdf8" : "#1e293b",
      strokeWidth: isLoopActive ? 2.5 : 1.5,
      strokeDasharray: "5,5",
    },
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: isLoopActive ? "#38bdf8" : "#475569",
    },
    label: "State Update Loop",
    labelStyle: {
      fill: isLoopActive ? "#38bdf8" : "#64748b",
      fontSize: 10,
      fontFamily: "monospace",
    },
  });

  return standardEdges;
};

export interface ArchitectureFlowProps {
  activeStage?: number | null;
  activeRunId?: string | null;
  currentStep?: number | null;
  maxSteps?: number;
  selectedScenario?: string;
  selectedPolicy?: string;
  simulationMode?: string;
  isStale?: boolean;
  selectedNodeId?: string;
  onSelectNode?: (nodeId: string) => void;
  liveStepData?: any;
}

export function ArchitectureFlow({
  activeStage = null,
  activeRunId = null,
  currentStep = null,
  maxSteps = 20,
  selectedScenario = "scenario-ransomware",
  selectedPolicy = "PPO",
  simulationMode = "Historical Replay",
  isStale = false,
  selectedNodeId: externalSelectedNodeId,
  onSelectNode,
  liveStepData = null,
}: ArchitectureFlowProps) {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [internalSelectedNodeId, setInternalSelectedNodeId] = useState<string>("4");

  const effectiveSelectedNodeId = externalSelectedNodeId || internalSelectedNodeId;

  // Compute node lifecycle states based on activeStage and missing checkpoint reality
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => {
        const stepNum = Number(n.id);
        const isCurrentActive = activeStage === stepNum;
        const isSelected = n.id === effectiveSelectedNodeId;

        let lifecycle: NodeLifecycleState = "Idle";

        // Checkpoint-dependent nodes (4 and 5) are fundamentally Unavailable on disk
        if (n.id === "4" || n.id === "5") {
          if (isCurrentActive) {
            lifecycle = "Processing"; // Currently traversing this stage in replay
          } else if (activeStage !== null && activeStage > stepNum) {
            lifecycle = "Completed";
          } else {
            lifecycle = "Unavailable";
          }
        } else {
          if (activeStage === null) {
            lifecycle = "Idle";
          } else if (isCurrentActive) {
            lifecycle = "Processing";
          } else if (activeStage > stepNum) {
            lifecycle = "Completed";
          } else {
            lifecycle = "Waiting";
          }
        }

        return {
          ...n,
          data: {
            ...n.data,
            isActive: isCurrentActive,
            isSelected,
            lifecycleState: lifecycle,
          },
        };
      })
    );
  }, [activeStage, effectiveSelectedNodeId]);

  // Edges update dynamically with active stage transition
  const edges = useMemo(() => createEdges(activeStage), [activeStage]);

  const onNodesChange = useCallback(
    (changes: any) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes: any) => applyEdgeChanges(changes, edges),
    [edges]
  );

  const handleNodeClick = useCallback(
    (_: any, node: Node) => {
      setInternalSelectedNodeId(node.id);
      if (onSelectNode) {
        onSelectNode(node.id);
      }
    },
    [onSelectNode]
  );

  const selectedMeta = NODE_DETAILS[effectiveSelectedNodeId] || NODE_DETAILS["4"];

  // Render live step telemetry specific to selected node
  const renderLiveNodeTelemetry = () => {
    if (!liveStepData && activeStage === null) {
      return (
        <p className="text-[11px] text-slate-500 italic">
          No live step telemetry available. Start simulation to inspect real runtime inputs/outputs.
        </p>
      );
    }

    switch (effectiveSelectedNodeId) {
      case "1":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Incident Source: data/processed/test.csv</div>
            <div className="text-slate-300">Target: {liveStepData?.target || "Corporate Workstations"}</div>
            <div className="text-slate-400">Tactics: {(liveStepData?.tactics || ["TA0001", "TA0040"]).join(", ")}</div>
          </div>
        );
      case "2":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Contextual Bandit State: Step {currentStep ?? 0} / {maxSteps}</div>
            <div className="text-slate-300">Environment Mode: Gymnasium Sequential Replay</div>
            <div className="text-slate-400">Termination: {currentStep && currentStep >= maxSteps ? "Terminated (Limit reached)" : "Active"}</div>
          </div>
        );
      case "3":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Semantic Tag: [{liveStepData?.cisa_phase || "Detection"}]</div>
            <div className="text-slate-300 line-clamp-3">Prepared: {liveStepData?.incident_text || liveStepData?.prepared_text || "Telemetry prepended with semantic phase prefix"}</div>
          </div>
        );
      case "4":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-amber-900/40">
            <div className="text-amber-400 font-semibold">SecBERT Status: Checkpoint Uncommitted</div>
            <div className="text-slate-400">Embedding Vector: Not Available (fallback to benchmark replay)</div>
            <div className="text-slate-400">Output Dimension: 768 (architectural target)</div>
          </div>
        );
      case "5":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-amber-900/40">
            <div className="text-amber-400 font-semibold">Policy Network: Checkpoint Uncommitted</div>
            <div className="text-slate-400">Probabilities π(a|s): Not Available (honest fallback)</div>
            <div className="text-slate-400">Critic V(s): Not Available</div>
          </div>
        );
      case "6":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Action Selected: #{liveStepData?.chosen_action ?? "—"} ({liveStepData?.chosen_action_name ?? "Pending"})</div>
            <div className="text-slate-300">CISA Phase: {liveStepData?.chosen_action_phase ?? "—"}</div>
          </div>
        );
      case "7":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Immediate Step Reward: {liveStepData?.reward !== undefined ? (liveStepData.reward > 0 ? "+1.0" : "-1.0") : "—"}</div>
            <div className="text-slate-300">Match Status: {liveStepData?.is_correct ? "Ground Truth Match (+1.0)" : "Mitigation Mismatch (-1.0)"}</div>
            <div className="text-slate-400">Host Containment: Not Available (Contextual Bandit)</div>
          </div>
        );
      case "8":
        return (
          <div className="space-y-1 text-[11px] font-mono bg-slate-950 p-2 rounded border border-slate-800">
            <div className="text-teal-400 font-semibold">Cumulative Return: {liveStepData?.cumulative_reward ?? 0.0}</div>
            <div className="text-slate-300">Step Progression: {currentStep ?? 0} / {maxSteps}</div>
            <div className="text-slate-400">Memory Buffer: Logged to experience replay queue</div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Top Header & Canvas Context Bar */}
      <div className="flex flex-col gap-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-xs font-semibold text-slate-100 uppercase tracking-wider flex items-center gap-2">
              <Network className="h-4 w-4 text-teal-400" />
              <span>Interactive System Architecture Flow (React Flow)</span>
            </h2>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Live 8-stage execution pipeline reflecting authentic research code. Click any stage to inspect responsibility, code location, and live telemetry.
            </p>
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400 self-start sm:self-auto">
            <span className="inline-block h-2 w-2 rounded-full bg-teal-400" />
            <span>Interactive Canvas: Pan, zoom, & click nodes</span>
          </div>
        </div>

        {/* Live Canvas Context Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-lg bg-slate-950/90 border border-slate-800 text-[11px] font-mono">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Mode:</span>
              <span className="px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30 font-semibold">
                {simulationMode}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Run ID:</span>
              <span className="text-slate-200">
                {activeRunId ? activeRunId.slice(0, 16) + "..." : "None (Idle)"}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Step:</span>
              <span className="text-teal-300 font-bold">
                {currentStep !== null ? `${currentStep} / ${maxSteps}` : "—"}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Policy:</span>
              <span className="text-slate-200">{selectedPolicy}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isStale ? (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 font-semibold animate-pulse">
                <AlertTriangle className="h-3 w-3 text-amber-400" />
                <span>Stream Stale (&gt;5s)</span>
              </span>
            ) : activeStage !== null ? (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 font-semibold">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping" />
                <span>Connected (Stage {activeStage})</span>
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                <Clock className="h-3 w-3" />
                <span>Ready for Launch</span>
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* React Flow Canvas (3 Columns on Large Screens) */}
        <div className="lg:col-span-3 h-[440px] rounded-xl border border-slate-800 bg-[#060a14] overflow-hidden relative shadow-inner">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={handleNodeClick}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            minZoom={0.5}
            maxZoom={1.5}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={16}
              size={1}
              color="#1e293b"
            />
            <Controls className="!bg-slate-900 !border-slate-800 !text-slate-300 !fill-slate-300" />
          </ReactFlow>

          {/* Quick Stage Navigation Pill Bar */}
          <div className="absolute bottom-2.5 left-3 right-3 hidden sm:flex items-center justify-between pointer-events-auto bg-slate-950/80 backdrop-blur px-2.5 py-1.5 rounded-lg border border-slate-800 text-[10px] font-mono">
            <span className="text-slate-400 font-semibold">Jump to Stage:</span>
            <div className="flex items-center gap-1">
              {["1", "2", "3", "4", "5", "6", "7", "8"].map((stageId) => (
                <button
                  key={stageId}
                  onClick={() => {
                    setInternalSelectedNodeId(stageId);
                    if (onSelectNode) onSelectNode(stageId);
                  }}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    effectiveSelectedNodeId === stageId
                      ? "bg-teal-500 text-slate-950 font-bold"
                      : "bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                  }`}
                >
                  {stageId}. {NODE_DETAILS[stageId]?.label.split(" ")[0]}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Selected Node Details Side Panel (1 Column on Large Screens) */}
        <div className="lg:col-span-1">
          <Card className="h-[440px] p-4 flex flex-col justify-between border-slate-800 bg-slate-900/80 overflow-y-auto">
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                <div>
                  <span className="text-[10px] font-mono font-bold text-teal-400">
                    STAGE {selectedMeta.step} OF 8
                  </span>
                  <h3 className="text-sm font-bold text-slate-100">
                    {selectedMeta.label}
                  </h3>
                </div>
                <span
                  className={`text-[9px] font-mono px-2 py-0.5 rounded border ${
                    selectedMeta.statusColor === "amber"
                      ? "bg-amber-500/15 text-amber-300 border-amber-500/30"
                      : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  {selectedMeta.status}
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                    Responsibility
                  </span>
                  <p className="text-xs text-slate-300 mt-0.5 leading-relaxed">
                    {selectedMeta.description}
                  </p>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                    Code Location
                  </span>
                  <p className="text-[11px] font-mono text-teal-300 mt-0.5 p-1.5 rounded bg-slate-950 border border-slate-800 break-all">
                    {selectedMeta.fileLocation}
                  </p>
                </div>

                {/* Live Telemetry Block */}
                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                    Live Telemetry
                  </span>
                  <div className="mt-1">{renderLiveNodeTelemetry()}</div>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                    Inputs
                  </span>
                  <ul className="mt-1 space-y-1">
                    {selectedMeta.inputs.map((inp, i) => (
                      <li key={i} className="text-[11px] text-slate-300 flex items-center gap-1.5 font-mono">
                        <span className="h-1 w-1 rounded-full bg-teal-400" />
                        <span>{inp}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                    Outputs
                  </span>
                  <ul className="mt-1 space-y-1">
                    {selectedMeta.outputs.map((out, i) => (
                      <li key={i} className="text-[11px] text-slate-300 flex items-center gap-1.5 font-mono">
                        <span className="h-1 w-1 rounded-full bg-amber-400" />
                        <span>{out}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            <div className="pt-2.5 mt-2.5 border-t border-slate-800/80 text-[10px] text-slate-400">
              <span className="block text-slate-400 uppercase font-semibold tracking-wider text-[9px] mb-0.5">
                Technical Note:
              </span>
              <p className="leading-tight">{selectedMeta.technicalDetails}</p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
