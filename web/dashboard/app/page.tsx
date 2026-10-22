"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { SystemStatusCards } from "@/components/simulation/SystemStatusCards";
import {
  SimulationConfigForm,
  SimulationConfig,
  ATTACK_SCENARIOS,
} from "@/components/simulation/SimulationConfigForm";
import { ArchitectureFlow } from "@/components/simulation/ArchitectureFlow";
import { CurrentDecisionPanel } from "@/components/simulation/CurrentDecisionPanel";
import { EventTimeline } from "@/components/simulation/EventTimeline";
import { ResultsSummaryCards } from "@/components/simulation/ResultsSummaryCards";
import { api } from "@/lib/api";
import { SimulationEvent, SimulationResults } from "@/types";

export default function SOCSimulationPage() {
  const [config, setConfig] = useState<SimulationConfig>({
    scenario: ATTACK_SCENARIOS[0].id,
    policy: "PPO",
    target: ATTACK_SCENARIOS[0].targetDefault,
    maxSteps: 20,
    seed: 42,
  });

  const [isStarting, setIsStarting] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [currentDecision, setCurrentDecision] = useState<any | null>(null);
  const [currentStep, setCurrentStep] = useState<number | null>(null);
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [results, setResults] = useState<SimulationResults | null>(null);
  const [isStale, setIsStale] = useState<boolean>(false);

  const eventSourceRef = useRef<EventSource | null>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const lastSeenSeqRef = useRef<number>(-1);
  const staleTimerRef = useRef<NodeJS.Timeout | null>(null);

  const resetStaleWatchdog = useCallback(() => {
    setIsStale(false);
    if (staleTimerRef.current) {
      clearTimeout(staleTimerRef.current);
    }
    staleTimerRef.current = setTimeout(() => {
      // 5s elapsed without an SSE event during active run
      setIsStale(true);
    }, 5000);
  }, []);

  const clearStaleWatchdog = useCallback(() => {
    if (staleTimerRef.current) {
      clearTimeout(staleTimerRef.current);
      staleTimerRef.current = null;
    }
    setIsStale(false);
  }, []);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearStaleWatchdog();
    };
  }, [clearStaleWatchdog]);

  const handleStart = async () => {
    setIsStarting(true);
    setEvents([]);
    setCurrentDecision(null);
    setCurrentStep(null);
    setResults(null);
    setActiveStage(1);
    lastSeenSeqRef.current = -1;
    clearStaleWatchdog();

    try {
      const res = await api.startSimulation({
        scenario: config.scenario,
        policy: config.policy,
        target: config.target,
        step_limit: config.maxSteps,
        seed: config.seed,
      });

      const runId = res.run_id;
      setActiveRunId(runId);
      activeRunIdRef.current = runId;
      setIsRunning(true);
      setIsStarting(false);
      resetStaleWatchdog();

      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const sseUrl = api.getSimulationEventStreamUrl(runId);
      const es = new EventSource(sseUrl);
      eventSourceRef.current = es;

      const handleSseMessage = (evtType: string, e: MessageEvent) => {
        try {
          const parsed = JSON.parse(e.data);
          const eventRunId = parsed.run_id || runId;

          // Reject events from older runs
          if (activeRunIdRef.current && eventRunId !== activeRunIdRef.current) {
            return;
          }

          const seq = typeof parsed.seq === "number" ? parsed.seq : Date.now();
          // Reject duplicates or out of order
          if (seq <= lastSeenSeqRef.current) {
            return;
          }
          lastSeenSeqRef.current = seq;
          resetStaleWatchdog();

          const simEvt: SimulationEvent = {
            run_id: eventRunId,
            seq: seq,
            timestamp: parsed.timestamp || new Date().toISOString(),
            event_type: evtType as any,
            data: parsed.data || parsed,
          };

          setEvents((prev) => [...prev, simEvt]);

          const d = simEvt.data;
          if (evtType === "simulation_started") {
            setActiveStage(1);
          } else if (evtType === "attack_stage_updated") {
            setActiveStage(2);
            if (d.step !== undefined) setCurrentStep(d.step);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              target: d.target,
              stage: d.stage,
              tactics: d.tactics,
            }));
          } else if (evtType === "incident_observed") {
            setActiveStage(3);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              incident_text: d.text,
              prepared_text: d.prepared_text,
              cisa_phase: d.cisa_phase,
              ground_truth_action: d.ground_truth_action,
              ground_truth_name: d.ground_truth_name,
            }));
          } else if (evtType === "embedding_generated") {
            setActiveStage(4);
          } else if (evtType === "action_selected") {
            setActiveStage(5);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              chosen_action: d.chosen_action,
              chosen_action_name: d.chosen_action_name,
              chosen_action_phase: d.chosen_action_phase,
              confidence: d.confidence,
              critic_value: d.critic_value,
            }));
          } else if (evtType === "simulated_effect_applied") {
            setActiveStage(6);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              implementation: d.implementation,
              threat_match: d.threat_match,
            }));
          } else if (evtType === "reward_calculated") {
            setActiveStage(7);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              reward: d.step_reward,
              cumulative_reward: d.cumulative_reward,
              is_correct: d.is_correct,
            }));
          } else if (evtType === "state_updated") {
            setActiveStage(8);
            if (d.step !== undefined) setCurrentStep(d.step);
          } else if (
            evtType === "simulation_completed" ||
            evtType === "simulation_stopped"
          ) {
            setIsRunning(false);
            setActiveStage(null);
            clearStaleWatchdog();
            setResults(d as SimulationResults);
            es.close();
          }
        } catch (err) {
          console.error("SSE parse error", err);
        }
      };

      const eventTypes = [
        "simulation_started",
        "attack_stage_updated",
        "incident_observed",
        "embedding_generated",
        "action_selected",
        "simulated_effect_applied",
        "reward_calculated",
        "state_updated",
        "simulation_completed",
        "simulation_stopped",
      ];

      eventTypes.forEach((t) => {
        es.addEventListener(t, (e: MessageEvent) => handleSseMessage(t, e));
      });

      es.onerror = () => {
        api
          .getSimulationDetail(runId)
          .then((detail) => {
            const s = (detail.status || "").toLowerCase();
            if (s === "completed" || s === "stopped" || s === "failed") {
              setIsRunning(false);
              setActiveStage(null);
              clearStaleWatchdog();
              if (detail.results) setResults(detail.results as SimulationResults);
              es.close();
            }
          })
          .catch(() => {});
      };
    } catch (err) {
      setIsStarting(false);
      setIsRunning(false);
      setActiveStage(null);
      clearStaleWatchdog();
      console.error("Failed to start simulation", err);
    }
  };

  const handleStop = async () => {
    if (!activeRunId) return;
    try {
      await api.stopSimulation(activeRunId);
      clearStaleWatchdog();
    } catch (err) {
      console.error("Failed to stop simulation", err);
    }
  };

  const handleReset = async () => {
    if (isRunning && activeRunId) {
      await handleStop();
    }
    try {
      await api.resetSimulations();
    } catch {
      // Ignore reset endpoint error if offline
    }
    setEvents([]);
    setCurrentDecision(null);
    setCurrentStep(null);
    setActiveStage(null);
    setResults(null);
    setActiveRunId(null);
    setIsRunning(false);
    clearStaleWatchdog();
  };

  const currentScenarioName =
    ATTACK_SCENARIOS.find((s) => s.id === config.scenario)?.name || config.scenario;

  const currentStatusString = isRunning ? "Running" : results ? "Completed" : "Ready";

  return (
    <div className="space-y-6 pb-12">
      {/* 1. Research Title Box */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <div className="space-y-1.5">
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent
          </h1>
          <p className="text-xs sm:text-sm text-slate-600 font-medium">
            Witharana A.D.S. &amp; Perera W.P.C.K.
          </p>
        </div>
      </div>

      {/* 2. System Component Status */}
      <section>
        <SystemStatusCards />
      </section>

      {/* 3. SOC Environment & Simulation Controls (Clean 2-Box Layout) */}
      <section>
        <SimulationConfigForm
          config={config}
          onConfigChange={setConfig}
          onStart={handleStart}
          onStop={handleStop}
          onReset={handleReset}
          isStarting={isStarting}
          isRunning={isRunning}
          activeRunId={activeRunId}
        />
      </section>

      {/* 4. Live System Architecture (React Flow Implementation) */}
      <section>
        <ArchitectureFlow
          scenario={currentScenarioName}
          simulationStatus={currentStatusString as any}
          currentStep={currentStep}
          maxSteps={config.maxSteps}
          activeStage={activeStage}
          selectedPolicy={config.policy}
          isStale={isStale}
        />
      </section>

      {/* 5. Live Decisions & Simulation Event Log (Side by Side) */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CurrentDecisionPanel
          selectedPolicy={config.policy}
          currentDecision={currentDecision}
          currentStep={currentStep}
          isRunning={isRunning}
        />
        <EventTimeline
          events={events}
          isRunning={isRunning}
          activeRunId={activeRunId}
        />
      </section>

      {/* 6. Current Simulation Results */}
      <section>
        <ResultsSummaryCards results={results} isRunning={isRunning} />
      </section>
    </div>
  );
}
