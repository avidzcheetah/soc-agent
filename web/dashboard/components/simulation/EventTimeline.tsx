"use client";

import React, { useRef, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { SimulationEvent } from "@/types";

interface Props {
  events?: SimulationEvent[];
  isRunning?: boolean;
  activeRunId?: string | null;
}

export function EventTimeline({
  events = [],
  isRunning = false,
  activeRunId = null,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll on new event
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length]);

  const formatEventName = (type: string) => {
    switch (type) {
      case "simulation_started":
        return "Simulation Started";
      case "attack_stage_updated":
        return "Adversary Attack";
      case "incident_observed":
        return "Alert Telemetry";
      case "embedding_generated":
        return "SecBERT Encoding";
      case "action_selected":
        return "Policy Action";
      case "simulated_effect_applied":
        return "Response Applied";
      case "reward_calculated":
        return "Reward Feedback";
      case "state_updated":
        return "State Updated";
      case "simulation_completed":
        return "Completed";
      case "simulation_stopped":
        return "Stopped";
      default:
        return type.replace(/_/g, " ");
    }
  };

  const renderEventDetails = (evt: SimulationEvent) => {
    const d = evt.data || {};
    switch (evt.event_type) {
      case "simulation_started":
        return `Initialized ${d.scenario || "scenario"} with ${d.policy || "policy"} (${d.max_steps || 20} steps)`;
      case "attack_stage_updated":
        return `Step ${d.step}: ${d.stage} targeting ${d.target || "host"}`;
      case "incident_observed":
        return `Step ${d.step}: ${d.text ? d.text.slice(0, 80) + "..." : "Alert ingested"}`;
      case "embedding_generated":
        return `Step ${d.step}: 768-dim contextual embedding computed`;
      case "action_selected":
        return `Step ${d.step}: Selected #${d.chosen_action} (${d.chosen_action_name || "Action"})`;
      case "simulated_effect_applied":
        return `Step ${d.step}: Applied mitigation '${d.action || d.chosen_action_name}'`;
      case "reward_calculated":
        return `Step ${d.step}: Reward ${d.step_reward > 0 ? "+1.0" : "-1.0"} (Cumulative: ${d.cumulative_reward || 0})`;
      case "state_updated":
        return `Step ${d.step}: Transition logged`;
      case "simulation_completed":
        return `Completed: ${d.total_steps || 0} steps, return ${d.cumulative_reward || 0}`;
      case "simulation_stopped":
        return `Stopped by user at step ${d.step || 0}`;
      default:
        return typeof d === "string" ? d : JSON.stringify(d);
    }
  };

  return (
    <Card className="p-4 border-slate-200 bg-white shadow-sm space-y-3 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
          Simulation Event Log
        </h2>
        <div className="flex items-center gap-2">
          {isRunning && (
            <span className="flex items-center gap-1.5 text-[11px] font-mono text-teal-700">
              <span className="h-1.5 w-1.5 rounded-full bg-teal-600 animate-pulse" />
              <span>Streaming</span>
            </span>
          )}
          <span className="text-[11px] font-mono text-slate-500">
            {events.length} events
          </span>
        </div>
      </div>

      {/* Chronological Event Log Table */}
      <div className="border border-slate-200 rounded-md overflow-hidden bg-slate-50">
        <div className="grid grid-cols-12 gap-2 px-3 py-1.5 bg-slate-100 text-[11px] font-semibold text-slate-700 border-b border-slate-200">
          <div className="col-span-3 sm:col-span-2">Timestamp</div>
          <div className="col-span-4 sm:col-span-3">Event</div>
          <div className="col-span-5 sm:col-span-7">Details</div>
        </div>

        {events.length === 0 ? (
          <div className="py-12 px-4 text-center text-xs text-slate-500">
            No events recorded yet. Start a simulation above to stream events.
          </div>
        ) : (
          <div
            ref={scrollRef}
            className="divide-y divide-slate-100 max-h-[300px] overflow-y-auto font-mono text-xs bg-white"
          >
            {events.map((evt, idx) => {
              const timeStr = evt.timestamp
                ? evt.timestamp.split("T")[1]?.slice(0, 8)
                : "--:--:--";
              return (
                <div
                  key={`${evt.run_id}-${evt.seq || idx}`}
                  className="grid grid-cols-12 gap-2 px-3 py-1.5 items-center hover:bg-slate-50 transition-colors"
                >
                  <div className="col-span-3 sm:col-span-2 text-[11px] text-slate-500">
                    {timeStr}
                  </div>
                  <div className="col-span-4 sm:col-span-3 text-[11px] font-semibold text-slate-800 truncate">
                    {formatEventName(evt.event_type)}
                  </div>
                  <div className="col-span-5 sm:col-span-7 text-[11px] text-slate-600 truncate font-sans">
                    {renderEventDetails(evt)}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="pt-2 text-[11px] text-slate-400 font-mono border-t border-slate-100 flex items-center justify-between">
        <span>Server-Sent Events</span>
        <span>SSE: active</span>
      </div>
    </Card>
  );
}
