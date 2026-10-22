"use client";

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  ReactFlow,
  applyNodeChanges,
  applyEdgeChanges,
  Node,
  Edge,
  Position,
  Handle,
  MarkerType,
  Background,
  BackgroundVariant,
  NodeProps,
} from "@xyflow/react";
import { Card } from "@/components/ui/Card";
import {
  Flame,
  Shield,
  Cpu,
  BrainCircuit,
  Zap,
  X,
  Radio,
  Search,
  CheckCircle2,
  ArrowRight,
  Info,
} from "lucide-react";

export interface NodeDetailInfo {
  id: string;
  name: string;
  category: string;
  role: string;
  detectors?: string;
  description: string;
}

export const ARCHITECTURE_DETAILS: Record<string, NodeDetailInfo> = {
  attack: {
    id: "attack",
    name: "Attack Scenario",
    category: "Adversary Emulation",
    role: "MITRE Caldera",
    description:
      "Emulates multi-stage cyber adversary tactics (initial access, privilege escalation, lateral movement, C2 beaconing) against lab targets.",
  },
  environment: {
    id: "environment",
    name: "Target Environment",
    category: "Target Subnet & Host",
    role: "Host State & Telemetry Ingestion",
    detectors: "Wazuh SIEM & OSQuery Telemetry",
    description:
      "Target lab host environment where Wazuh SIEM agents and OSQuery daemons are co-located to detect malicious processes, socket connections, and system log anomalies.",
  },
  secbert: {
    id: "secbert",
    name: "SecBERT Encoder",
    category: "Contextual State Model",
    role: "768-D Transformer",
    description:
      "Transforms raw normalized alerts and CISA playbook phase tags into dense 768-dimensional contextual state embeddings for the policy agent.",
  },
  ppo: {
    id: "ppo",
    name: "PPO Policy Agent",
    category: "Response Decision Agent",
    role: "Reinforcement Learning Policy",
    description:
      "Evaluates the 768-D state embedding to select the optimal incident response mitigation out of 20 canonical CISA actions (e.g. isolate host, block IP, terminate process).",
  },
  response: {
    id: "response",
    name: "Simulated Response",
    category: "Closed-Loop Feedback",
    role: "Mitigation & Reward Feedback",
    description:
      "Executes the selected mitigation action on the environment, assesses containment or disruption, and feeds state transition (+1.0 / -1.0) back into the environment.",
  },
};

// 1. Attack Node Component
function AttackNode({ data, selected }: NodeProps) {
  const isActive = Boolean(data.isActive);
  const isSelected = selected || Boolean(data.isSelected);

  return (
    <div
      className={`relative p-3 rounded-xl text-left w-[200px] h-[114px] transition-all select-none shadow-sm flex flex-col justify-between ${
        isActive
          ? "bg-rose-50 border-2 border-rose-500 ring-2 ring-rose-200"
          : isSelected
          ? "bg-white border-2 border-slate-700 shadow"
          : "bg-white border border-slate-200 hover:border-slate-300"
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5 text-rose-700 font-bold text-[11px] uppercase tracking-wide">
            <Flame className="h-3.5 w-3.5 text-rose-600" />
            <span>Attack</span>
          </div>
          <span
            className={`h-2 w-2 rounded-full ${
              isActive ? "bg-rose-500 animate-ping" : "bg-slate-300"
            }`}
          />
        </div>
        <div className="text-xs font-bold text-slate-900 leading-tight">
          MITRE Caldera
        </div>
        <div className="text-[10px] text-slate-500 font-mono mt-0.5">
          Adversary Emulation
        </div>
      </div>

      <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500 font-mono">
        <span>4 TTP Scenarios</span>
        <span className="text-rose-600 font-semibold">Active Input</span>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="attack-out"
        className="!w-2.5 !h-2.5 !bg-rose-500 !border-2 !border-white"
      />
    </div>
  );
}

// 2. Environment Node Component (Wazuh & OSQuery detectors co-located ON environment)
function EnvironmentNode({ data, selected }: NodeProps) {
  const isActive = Boolean(data.isActive);
  const isSelected = selected || Boolean(data.isSelected);

  return (
    <div
      className={`relative p-3 rounded-xl text-left w-[270px] h-[114px] transition-all select-none shadow-sm flex flex-col justify-between ${
        isActive
          ? "bg-teal-50 border-2 border-teal-600 ring-2 ring-teal-200"
          : isSelected
          ? "bg-white border-2 border-slate-700 shadow"
          : "bg-white border border-slate-200 hover:border-slate-300"
      }`}
    >
      {/* Input from Attack */}
      <Handle
        type="target"
        position={Position.Left}
        id="env-in-attack"
        className="!w-2.5 !h-2.5 !bg-slate-400 !border-2 !border-white"
      />

      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5 text-teal-800 font-bold text-[11px] uppercase tracking-wide">
            <Shield className="h-3.5 w-3.5 text-teal-600" />
            <span>Environment</span>
          </div>
          <span className="flex items-center gap-1 text-[9px] font-mono font-semibold text-emerald-800 bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            <span>Target Lab Host</span>
          </span>
        </div>
        <div className="text-xs font-bold text-slate-900 leading-tight">
          Target Subnet &amp; Host
        </div>
      </div>

      {/* Embedded Detectors Section ON the Environment */}
      <div className="pt-1.5 border-t border-slate-100">
        <div className="text-[9px] font-mono uppercase text-slate-400 font-semibold mb-1">
          Detectors on Environment:
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-teal-50 border border-teal-200 text-teal-800 font-mono text-[10px] font-medium">
            <Radio className="h-2.5 w-2.5 text-teal-600" />
            <span>Wazuh SIEM</span>
          </span>
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-700 font-mono text-[10px] font-medium">
            <Search className="h-2.5 w-2.5 text-slate-500" />
            <span>OSQuery</span>
          </span>
        </div>
      </div>

      {/* Output to SecBERT */}
      <Handle
        type="source"
        position={Position.Right}
        id="env-out-secbert"
        className="!w-2.5 !h-2.5 !bg-teal-600 !border-2 !border-white"
      />

      {/* Input from Simulated Response feedback loop */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="env-in-feedback"
        className="!w-2.5 !h-2.5 !bg-amber-500 !border-2 !border-white"
      />
    </div>
  );
}

// 3. SecBERT Node Component
function SecBERTNode({ data, selected }: NodeProps) {
  const isActive = Boolean(data.isActive);
  const isSelected = selected || Boolean(data.isSelected);

  return (
    <div
      className={`relative p-3 rounded-xl text-left w-[200px] h-[114px] transition-all select-none shadow-sm flex flex-col justify-between ${
        isActive
          ? "bg-sky-50 border-2 border-sky-500 ring-2 ring-sky-200"
          : isSelected
          ? "bg-white border-2 border-slate-700 shadow"
          : "bg-white border border-slate-200 hover:border-slate-300"
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="secbert-in"
        className="!w-2.5 !h-2.5 !bg-slate-400 !border-2 !border-white"
      />

      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5 text-sky-800 font-bold text-[11px] uppercase tracking-wide">
            <Cpu className="h-3.5 w-3.5 text-sky-600" />
            <span>SecBERT</span>
          </div>
          <span
            className={`h-2 w-2 rounded-full ${
              isActive ? "bg-sky-500 animate-ping" : "bg-emerald-500"
            }`}
          />
        </div>
        <div className="text-xs font-bold text-slate-900 leading-tight">
          Context Encoder
        </div>
        <div className="text-[10px] text-slate-500 font-mono mt-0.5">
          768-D State Embedding
        </div>
      </div>

      <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500 font-mono">
        <span>Alert &amp; TTP Tokens</span>
        <span className="text-sky-600 font-semibold">12-Layer</span>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        id="secbert-out"
        className="!w-2.5 !h-2.5 !bg-sky-500 !border-2 !border-white"
      />
    </div>
  );
}

// 4. PPO Agent Node Component
function PPOAgentNode({ data, selected }: NodeProps) {
  const isActive = Boolean(data.isActive);
  const isSelected = selected || Boolean(data.isSelected);
  const policyLabel = data.policy || "PPO Agent";

  return (
    <div
      className={`relative p-3 rounded-xl text-left w-[200px] h-[114px] transition-all select-none shadow-sm flex flex-col justify-between ${
        isActive
          ? "bg-indigo-50 border-2 border-indigo-600 ring-2 ring-indigo-200"
          : isSelected
          ? "bg-white border-2 border-slate-700 shadow"
          : "bg-white border border-slate-200 hover:border-slate-300"
      }`}
    >
      <Handle
        type="target"
        position={Position.Left}
        id="ppo-in"
        className="!w-2.5 !h-2.5 !bg-slate-400 !border-2 !border-white"
      />

      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5 text-indigo-800 font-bold text-[11px] uppercase tracking-wide">
            <BrainCircuit className="h-3.5 w-3.5 text-indigo-600" />
            <span>PPO Agent</span>
          </div>
          <span
            className={`h-2 w-2 rounded-full ${
              isActive ? "bg-indigo-600 animate-ping" : "bg-emerald-500"
            }`}
          />
        </div>
        <div className="text-xs font-bold text-slate-900 leading-tight">
          {policyLabel}
        </div>
        <div className="text-[10px] text-slate-500 font-mono mt-0.5">
          Action Policy Optimization
        </div>
      </div>

      <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[10px] text-slate-500 font-mono">
        <span>20 CISA Actions</span>
        <span className="text-indigo-600 font-semibold">Decision</span>
      </div>

      {/* Downward source to Simulated Response */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="ppo-out-response"
        className="!w-2.5 !h-2.5 !bg-indigo-600 !border-2 !border-white"
      />
    </div>
  );
}

// 5. Simulated Response Feedback Node Component
function SimulatedResponseNode({ data, selected }: NodeProps) {
  const isActive = Boolean(data.isActive);
  const isSelected = selected || Boolean(data.isSelected);
  const actionName = data.chosenAction || "Mitigation Action";
  const rewardVal = data.reward;

  return (
    <div
      className={`relative p-3 rounded-xl text-left w-[320px] h-[80px] transition-all select-none shadow-sm flex flex-col justify-between ${
        isActive
          ? "bg-amber-50 border-2 border-amber-500 ring-2 ring-amber-200"
          : isSelected
          ? "bg-white border-2 border-slate-700 shadow"
          : "bg-white border border-slate-200 hover:border-slate-300"
      }`}
    >
      {/* Input from PPO Agent on right */}
      <Handle
        type="target"
        position={Position.Right}
        id="resp-in-ppo"
        className="!w-2.5 !h-2.5 !bg-slate-400 !border-2 !border-white"
      />

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded bg-amber-100 text-amber-800">
            <Zap className="h-3.5 w-3.5" />
          </div>
          <div>
            <div className="text-xs font-bold text-slate-900 leading-tight">
              Simulated Response
            </div>
            <div className="text-[10px] text-slate-600 font-mono truncate max-w-[170px]">
              {actionName}
            </div>
          </div>
        </div>

        <div className="text-right font-mono text-[10px]">
          <span className="text-[9px] text-slate-400 uppercase block font-semibold">
            Feedback
          </span>
          <span
            className={`font-bold ${
              rewardVal !== undefined && rewardVal !== null
                ? rewardVal > 0
                  ? "text-emerald-700"
                  : "text-rose-700"
                : "text-slate-500"
            }`}
          >
            {rewardVal !== undefined && rewardVal !== null
              ? rewardVal > 0
                ? "+1.0 Reward"
                : "-1.0 Penalty"
              : "Standby"}
          </span>
        </div>
      </div>

      <div className="text-[10px] text-slate-500 font-mono flex items-center justify-between pt-1 border-t border-slate-100">
        <span>Closed-Loop Evaluation</span>
        <span className="text-amber-700 font-semibold">Updates Environment</span>
      </div>

      {/* Output back to Environment on left */}
      <Handle
        type="source"
        position={Position.Left}
        id="resp-out-env"
        className="!w-2.5 !h-2.5 !bg-amber-500 !border-2 !border-white"
      />
    </div>
  );
}

const nodeTypes = {
  attackNode: AttackNode,
  environmentNode: EnvironmentNode,
  secbertNode: SecBERTNode,
  ppoNode: PPOAgentNode,
  responseNode: SimulatedResponseNode,
};

interface Props {
  scenario?: string;
  simulationStatus?: "Ready" | "Running" | "Stopping" | "Completed" | "Failed";
  currentStep?: number | null;
  maxSteps?: number;
  activeStage?: number | null;
  selectedPolicy?: string;
  isStale?: boolean;
  liveStepData?: any;
}

export function ArchitectureFlow({
  scenario = "C2 and data exfiltration",
  simulationStatus = "Ready",
  currentStep = null,
  maxSteps = 20,
  activeStage = null,
  selectedPolicy = "PPO",
  isStale = false,
  liveStepData = null,
}: Props) {
  const [selectedNodeKey, setSelectedNodeKey] = useState<string | null>(null);

  // Active stage determination
  const activeComponentKey = useMemo(() => {
    if (simulationStatus !== "Running" || activeStage === null) return null;
    switch (activeStage) {
      case 1:
        return "attack";
      case 2:
      case 3:
        return "environment";
      case 4:
        return "secbert";
      case 5:
        return "ppo";
      case 6:
      case 7:
        return "response";
      default:
        return null;
    }
  }, [simulationStatus, activeStage]);

  const activeStageDescription = useMemo(() => {
    if (simulationStatus !== "Running") {
      return simulationStatus === "Completed"
        ? "Simulation completed successfully"
        : "Ready for simulation trigger";
    }
    switch (activeStage) {
      case 1:
        return "Adversary executing attack scenario (Caldera)";
      case 2:
      case 3:
        return "Environment ingesting telemetry via Wazuh SIEM & OSQuery detectors";
      case 4:
        return "SecBERT generating 768-D contextual state embedding";
      case 5:
        return `PPO Agent selecting optimal response action (${selectedPolicy})`;
      case 6:
      case 7:
        return `Simulated response evaluated; feedback loop returning to environment`;
      default:
        return "Executing incident response cycle";
    }
  }, [simulationStatus, activeStage, selectedPolicy]);

  // Exact 5-node layout strictly implementing:
  // [Attack] ──> [Environment] ──> [SecBERT] ──> [PPO Agent]
  //                    ▲                                  │
  //                    │                                  ▼
  //                    └────── [Simulated Response] <─────┘
  // Top row Y: 35. Node heights: 114px. Handles centered at Y = 92px.
  // Straight horizontal arrows connecting Attack -> Environment -> SecBERT -> PPO Agent.
  // PPO Agent drops down to Simulated Response (Y: 195).
  // Simulated Response feeds left and up back into Environment.
  const initialNodes: Node[] = useMemo(() => {
    return [
      {
        id: "attack",
        type: "attackNode",
        position: { x: 20, y: 35 },
        data: {
          isActive: activeComponentKey === "attack",
          isSelected: selectedNodeKey === "attack",
        },
      },
      {
        id: "environment",
        type: "environmentNode",
        position: { x: 280, y: 35 },
        data: {
          isActive: activeComponentKey === "environment",
          isSelected: selectedNodeKey === "environment",
        },
      },
      {
        id: "secbert",
        type: "secbertNode",
        position: { x: 610, y: 35 },
        data: {
          isActive: activeComponentKey === "secbert",
          isSelected: selectedNodeKey === "secbert",
        },
      },
      {
        id: "ppo",
        type: "ppoNode",
        position: { x: 870, y: 35 },
        data: {
          policy: `${selectedPolicy} Policy`,
          isActive: activeComponentKey === "ppo",
          isSelected: selectedNodeKey === "ppo",
        },
      },
      {
        id: "response",
        type: "responseNode",
        position: { x: 535, y: 195 },
        data: {
          isActive: activeComponentKey === "response",
          isSelected: selectedNodeKey === "response",
          chosenAction: liveStepData?.chosen_action_name,
          reward: liveStepData?.reward,
        },
      },
    ];
  }, [activeComponentKey, selectedNodeKey, selectedPolicy, liveStepData]);

  // Edges connecting exactly per architecture specification
  const initialEdges: Edge[] = useMemo(() => {
    const isRunning = simulationStatus === "Running";

    return [
      // 1. [Attack] ──> [Environment]
      {
        id: "e-attack-env",
        source: "attack",
        sourceHandle: "attack-out",
        target: "environment",
        targetHandle: "env-in-attack",
        type: "smoothstep",
        animated: isRunning && activeComponentKey === "attack",
        style: {
          stroke: activeComponentKey === "attack" ? "#e11d48" : "#94a3b8",
          strokeWidth: activeComponentKey === "attack" ? 2.5 : 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: activeComponentKey === "attack" ? "#e11d48" : "#94a3b8",
        },
      },

      // 2. [Environment] ──> [SecBERT]
      {
        id: "e-env-secbert",
        source: "environment",
        sourceHandle: "env-out-secbert",
        target: "secbert",
        targetHandle: "secbert-in",
        type: "smoothstep",
        animated:
          isRunning &&
          (activeComponentKey === "environment" || activeComponentKey === "secbert"),
        style: {
          stroke: activeComponentKey === "secbert" ? "#0284c7" : "#94a3b8",
          strokeWidth: activeComponentKey === "secbert" ? 2.5 : 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: activeComponentKey === "secbert" ? "#0284c7" : "#94a3b8",
        },
      },

      // 3. [SecBERT] ──> [PPO Agent]
      {
        id: "e-secbert-ppo",
        source: "secbert",
        sourceHandle: "secbert-out",
        target: "ppo",
        targetHandle: "ppo-in",
        type: "smoothstep",
        animated: isRunning && activeComponentKey === "ppo",
        style: {
          stroke: activeComponentKey === "ppo" ? "#6366f1" : "#94a3b8",
          strokeWidth: activeComponentKey === "ppo" ? 2.5 : 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: activeComponentKey === "ppo" ? "#6366f1" : "#94a3b8",
        },
      },

      // 4. [PPO Agent] ──> [Simulated Response] (downward orthogonal)
      {
        id: "e-ppo-response",
        source: "ppo",
        sourceHandle: "ppo-out-response",
        target: "response",
        targetHandle: "resp-in-ppo",
        type: "smoothstep",
        animated: isRunning && activeComponentKey === "response",
        style: {
          stroke: activeComponentKey === "response" ? "#d97706" : "#94a3b8",
          strokeWidth: activeComponentKey === "response" ? 2.5 : 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: activeComponentKey === "response" ? "#d97706" : "#94a3b8",
        },
      },

      // 5. [Simulated Response] ──> [Environment] (feedback loop leftward and upward)
      {
        id: "e-response-env",
        source: "response",
        sourceHandle: "resp-out-env",
        target: "environment",
        targetHandle: "env-in-feedback",
        type: "smoothstep",
        animated: isRunning && activeComponentKey === "response",
        style: {
          stroke: activeComponentKey === "response" ? "#0d9488" : "#94a3b8",
          strokeWidth: activeComponentKey === "response" ? 2.5 : 1.5,
          strokeDasharray: "5,5",
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: activeComponentKey === "response" ? "#0d9488" : "#94a3b8",
        },
      },
    ];
  }, [simulationStatus, activeComponentKey]);

  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);

  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges]);

  const onNodesChange = useCallback(
    (changes: any) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  const onEdgesChange = useCallback(
    (changes: any) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onNodeClick = (_: any, node: Node) => {
    setSelectedNodeKey((prev) => (prev === node.id ? null : node.id));
  };

  const selectedDetails = selectedNodeKey ? ARCHITECTURE_DETAILS[selectedNodeKey] : null;

  return (
    <Card className="p-5 border-slate-200 bg-white shadow-sm space-y-4">
      {/* Header with live scenario and step info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-teal-600" />
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Live System Architecture
            </h2>
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5">
            Closed-loop reinforcement learning pipeline with co-located Wazuh &amp; OSQuery host detectors
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700">
            <span className="text-slate-400">Status:</span>
            <strong
              className={
                simulationStatus === "Running"
                  ? "text-emerald-700"
                  : simulationStatus === "Completed"
                  ? "text-teal-700"
                  : "text-slate-800"
              }
            >
              {isStale ? "Stale / Reconnecting" : simulationStatus}
            </strong>
          </span>

          <span className="px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700">
            <span className="text-slate-400">Step: </span>
            <strong className="text-slate-900">
              {currentStep !== null ? `${currentStep} / ${maxSteps}` : `0 / ${maxSteps}`}
            </strong>
          </span>
        </div>
      </div>

      {/* Visual Canvas: Closed-Loop Architecture */}
      <div className="h-[290px] w-full bg-slate-50/60 rounded-xl border border-slate-200 overflow-hidden relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={true}
          panOnDrag={false}
          zoomOnScroll={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#cbd5e1" />
        </ReactFlow>

        <div className="absolute top-2.5 right-3 text-[10px] font-mono text-slate-500 bg-white/95 px-2.5 py-1 rounded-md border border-slate-200 shadow-xs pointer-events-none">
          Closed-Loop Response Pipeline
        </div>
      </div>

      {/* Progress & Current Stage Status Banner */}
      <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-700">Current Phase:</span>
          <span className="font-mono text-teal-900 font-semibold">
            {activeStageDescription}
          </span>
        </div>

        <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
          {simulationStatus === "Running" ? (
            <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Step Active</span>
            </span>
          ) : (
            <span>Feedback loop armed</span>
          )}
        </div>
      </div>

      {/* Detail panel on node click */}
      {selectedDetails && (
        <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-xs flex items-start justify-between gap-3 animate-in fade-in duration-200">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-bold text-slate-900">{selectedDetails.name}</span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-200 text-slate-700">
                {selectedDetails.category}
              </span>
              {selectedDetails.detectors && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-teal-50 text-teal-800 border border-teal-200">
                  {selectedDetails.detectors}
                </span>
              )}
            </div>
            <p className="text-slate-600 leading-relaxed">
              {selectedDetails.description}
            </p>
          </div>

          <button
            type="button"
            onClick={() => setSelectedNodeKey(null)}
            className="text-slate-400 hover:text-slate-700 p-1 rounded transition-colors"
            title="Close details"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </Card>
  );
}
