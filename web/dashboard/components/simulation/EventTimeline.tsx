"use client";

import React, { useRef, useEffect, useState, useMemo } from "react";
import {
  Clock,
  Activity,
  AlertCircle,
  CheckCircle2,
  ListFilter,
  PlayCircle,
  Cpu,
  BrainCircuit,
  Award,
  Zap,
  ShieldAlert,
  Radio,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { SimulationEvent } from "@/types";

export type EventCategory =
  | "all"
  | "scenario"
  | "incident"
  | "secbert"
  | "policy"
  | "mitigation"
  | "state";

interface Props {
  events?: SimulationEvent[];
  isRunning?: boolean;
  activeRunId?: string | null;
  selectedNodeId?: string | null;
  onSelectNode?: (nodeId: string) => void;
}

export const EVENT_NODE_MAP: Record<string, string> = {
  simulation_started: "1",
  attack_stage_updated: "1",
  incident_observed: "3", // Ingested & preprocessed
  embedding_generated: "4", // SecBERT encoder
  action_selected: "5", // PPO Policy & action
  simulated_effect_applied: "6", // Response Action executed
  reward_calculated: "7", // Simulated effect & reward
  state_updated: "8", // Updated state & metrics
  simulation_completed: "8",
  simulation_stopped: "8",
};

export const EVENT_CATEGORY_MAP: Record<string, EventCategory> = {
  simulation_started: "scenario",
  attack_stage_updated: "scenario",
  incident_observed: "incident",
  embedding_generated: "secbert",
  action_selected: "policy",
  simulated_effect_applied: "mitigation",
  reward_calculated: "mitigation",
  state_updated: "state",
  simulation_completed: "scenario",
  simulation_stopped: "scenario",
};

export function EventTimeline({
  events = [],
  isRunning = false,
  activeRunId = null,
  selectedNodeId = null,
  onSelectNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedCategory, setSelectedCategory] = useState<EventCategory>("all");

  // Auto-scroll on new event arrival when at or near bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events.length]);

  const categories: { id: EventCategory; label: string; count: number }[] = useMemo(() => {
    const counts: Record<EventCategory, number> = {
      all: events.length,
      scenario: 0,
      incident: 0,
      secbert: 0,
      policy: 0,
      mitigation: 0,
      state: 0,
    };

    events.forEach((evt) => {
      const cat = EVENT_CATEGORY_MAP[evt.event_type] || "scenario";
      counts[cat] = (counts[cat] || 0) + 1;
    });

    return [
      { id: "all", label: "All Events", count: counts.all },
      { id: "scenario", label: "Scenario Progression", count: counts.scenario },
      { id: "incident", label: "Incident Context", count: counts.incident },
      { id: "secbert", label: "SecBERT Encoder", count: counts.secbert },
      { id: "policy", label: "Policy Decisions", count: counts.policy },
      { id: "mitigation", label: "Mitigation & Rewards", count: counts.mitigation },
      { id: "state", label: "State Transitions", count: counts.state },
    ];
  }, [events]);

  const filteredEvents = useMemo(() => {
    if (selectedCategory === "all") return events;
    return events.filter(
      (evt) => EVENT_CATEGORY_MAP[evt.event_type] === selectedCategory
    );
  }, [events, selectedCategory]);

  const getEventBadge = (type: string) => {
    switch (type) {
      case "simulation_started":
        return {
          icon: PlayCircle,
          color: "text-teal-400 bg-teal-500/10 border-teal-500/30",
          label: "Simulation Started",
          stage: "1",
        };
      case "attack_stage_updated":
        return {
          icon: ShieldAlert,
          color: "text-rose-400 bg-rose-500/10 border-rose-500/30",
          label: "Attack Stage",
          stage: "1",
        };
      case "incident_observed":
        return {
          icon: AlertCircle,
          color: "text-amber-400 bg-amber-500/10 border-amber-500/30",
          label: "Incident Ingested",
          stage: "3",
        };
      case "embedding_generated":
        return {
          icon: Cpu,
          color: "text-sky-400 bg-sky-500/10 border-sky-500/30",
          label: "SecBERT State",
          stage: "4",
        };
      case "action_selected":
        return {
          icon: BrainCircuit,
          color: "text-teal-300 bg-teal-500/15 border-teal-500/40",
          label: "Policy Decision",
          stage: "5",
        };
      case "simulated_effect_applied":
        return {
          icon: Zap,
          color: "text-indigo-400 bg-indigo-500/10 border-indigo-500/30",
          label: "Effect Applied",
          stage: "6",
        };
      case "reward_calculated":
        return {
          icon: Award,
          color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
          label: "Reward Feedback",
          stage: "7",
        };
      case "state_updated":
        return {
          icon: Activity,
          color: "text-cyan-400 bg-cyan-500/10 border-cyan-500/30",
          label: "State Transition",
          stage: "8",
        };
      case "simulation_completed":
      case "simulation_stopped":
        return {
          icon: CheckCircle2,
          color: "text-emerald-300 bg-emerald-500/20 border-emerald-500/50",
          label: type === "simulation_completed" ? "Simulation Completed" : "Simulation Stopped",
          stage: "8",
        };
      default:
        return {
          icon: Radio,
          color: "text-slate-400 bg-slate-800 border-slate-700",
          label: type,
          stage: "1",
        };
    }
  };

  const renderEventSummary = (evt: SimulationEvent) => {
    const d = evt.data || {};
    switch (evt.event_type) {
      case "simulation_started":
        return `Rollout initialized for ${d.scenario} using ${d.policy} (${d.max_steps} steps, seed=${d.seed})`;
      case "attack_stage_updated":
        return `Step ${d.step}: ${d.stage} targeting ${d.target} [${(d.tactics || []).join(", ")}]`;
      case "incident_observed":
        return `Step ${d.step}: [${d.cisa_phase}] ${d.text ? d.text.slice(0, 95) + "..." : "Incident alert ingested"}`;
      case "embedding_generated": {
        if (d.status === "unavailable" || d.norm === null || d.norm === undefined) {
          return `Step ${d.step}: SecBERT checkpoint uncommitted on disk; proceeding with benchmark incident text`;
        }
        return `Step ${d.step}: 768-dim state embedding produced (L2 norm: ${d.norm})`;
      }
      case "action_selected": {
        const confStr = d.confidence !== null && d.confidence !== undefined ? `${(d.confidence * 100).toFixed(1)}% confidence` : "Ground-truth replay";
        const valStr = d.critic_value !== null && d.critic_value !== undefined ? `, V(s)=${d.critic_value}` : "";
        return `Step ${d.step}: Action #${d.chosen_action} (${d.chosen_action_name}) selected [${confStr}${valStr}]`;
      }
      case "simulated_effect_applied":
        return `Step ${d.step}: Simulated mitigation '${d.action}' applied [${d.implementation}]`;
      case "reward_calculated":
        return `Step ${d.step}: Assigned reward ${d.step_reward > 0 ? "+1.0" : "-1.0"} (Cumulative: ${d.cumulative_reward}) - ${d.is_correct ? "Ground truth match" : "Mitigation mismatch"}`;
      case "state_updated":
        return `Step ${d.step}: Environment state transition logged into experience buffer (buffer size: ${d.memory_buffer_size})`;
      case "simulation_completed":
        return `Rollout finished: ${d.total_steps} steps executed with ${(d.accuracy * 100).toFixed(1)}% accuracy and return ${d.cumulative_reward}`;
      case "simulation_stopped":
        return `Execution stopped by user: ${d.reason || "Manual halt"} at step ${d.step}`;
      default:
        return JSON.stringify(d);
    }
  };

  return (
    <Card className="p-5 border-slate-800/80 bg-slate-900/60 space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h2 className="text-xs font-semibold text-slate-100 uppercase tracking-wider flex items-center gap-2">
            <Clock className="h-4 w-4 text-teal-400" />
            <span>Simulation Event Timeline (SSE Stream)</span>
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Click any event to highlight and inspect its corresponding architecture node.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && (
            <span className="flex items-center gap-1.5 text-[10px] font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
              <span className="h-1.5 w-1.5 rounded-full bg-teal-400 animate-ping" />
              <span>LIVE</span>
            </span>
          )}
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
            {events.length} Events
          </span>
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="flex flex-wrap items-center gap-1.5 pb-1">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`px-2 py-1 rounded text-[10px] font-mono flex items-center gap-1 transition-all ${
              selectedCategory === cat.id
                ? "bg-teal-500 text-slate-950 font-bold shadow-sm"
                : "bg-slate-950/80 text-slate-400 border border-slate-800/80 hover:border-slate-700 hover:text-slate-200"
            }`}
          >
            <span>{cat.label}</span>
            <span
              className={`text-[9px] px-1 rounded-full ${
                selectedCategory === cat.id
                  ? "bg-slate-950/20 text-slate-950"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              {cat.count}
            </span>
          </button>
        ))}
      </div>

      {/* Event Stream List */}
      {filteredEvents.length === 0 ? (
        <div className="py-10 px-4 text-center rounded-lg bg-slate-950/60 border border-slate-800/60 space-y-3">
          <div className="h-10 w-10 mx-auto rounded-full bg-slate-800/80 border border-slate-700 text-slate-400 flex items-center justify-center">
            <Activity className="h-5 w-5" />
          </div>
          <div className="max-w-md mx-auto space-y-1">
            <h4 className="text-xs font-semibold text-slate-200">
              {events.length === 0
                ? "Awaiting Simulation Trigger"
                : "No Events for Selected Filter"}
            </h4>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              {events.length === 0
                ? "Click 'Start Simulation' above to launch a managed background run. Real-time events will stream live into this timeline."
                : "Select 'All Events' above to view the full execution trace."}
            </p>
          </div>
        </div>
      ) : (
        <div
          ref={containerRef}
          className="space-y-2 max-h-[380px] overflow-y-auto pr-1 text-xs"
        >
          {filteredEvents.map((evt) => {
            const badge = getEventBadge(evt.event_type);
            const Icon = badge.icon;
            const timeStr = evt.timestamp
              ? evt.timestamp.split("T")[1]?.slice(0, 8)
              : "";
            const mappedNodeId = EVENT_NODE_MAP[evt.event_type] || "1";
            const isStageSelected = selectedNodeId === mappedNodeId;

            return (
              <div
                key={`${evt.run_id}-${evt.seq}`}
                onClick={() => {
                  if (onSelectNode) {
                    onSelectNode(mappedNodeId);
                  }
                }}
                className={`p-2.5 rounded-lg text-left space-y-1 cursor-pointer transition-all ${
                  isStageSelected
                    ? "bg-[#0b1528] border-teal-500/60 ring-1 ring-teal-500/40 shadow-md"
                    : "bg-slate-950/90 border border-slate-800/80 hover:border-slate-700 hover:bg-slate-900/60"
                }`}
                title={`Click to inspect Stage ${mappedNodeId} in Architecture Flow`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border ${badge.color}`}
                    >
                      <Icon className="h-3 w-3" />
                      <span>{badge.label}</span>
                    </span>
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-teal-400">
                      Stage {badge.stage}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">
                      seq #{evt.seq}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-400">
                    <span>{timeStr}</span>
                    <ChevronRight className="h-3 w-3 text-slate-500" />
                  </div>
                </div>
                <p className="text-[11px] text-slate-300 font-sans leading-relaxed">
                  {renderEventSummary(evt)}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}
