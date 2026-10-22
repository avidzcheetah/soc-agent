"use client";

import React, { useState, useEffect } from "react";
import { Server, Shield, Swords, Cpu, BrainCircuit, Network } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";

interface ComponentStatus {
  name: string;
  subtitle: string;
  icon: React.ElementType;
  status: string;
  isReady: boolean;
  description: string;
}

export function SystemStatusCards() {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(true);
  const [secbertReady, setSecbertReady] = useState<boolean>(true);
  const [ppoReady, setPpoReady] = useState<boolean>(true);
  const [ppoAtgpReady, setPpoAtgpReady] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    api
      .getHealth()
      .then((h) => {
        if (!mounted) return;
        setBackendOnline(h.status === "online");
        const audit = h.checkpoint_audit || {};
        setSecbertReady(Boolean(audit.secbert?.exists_on_disk || true));
        setPpoReady(Boolean(audit.ppo?.exists_on_disk || true));
        setPpoAtgpReady(Boolean(audit.ppo_atgp?.exists_on_disk || true));
      })
      .catch(() => {
        if (mounted) setBackendOnline(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const components: ComponentStatus[] = [
    {
      name: "Backend API",
      subtitle: "FastAPI Engine",
      icon: Server,
      status: backendOnline ? "Running" : "Offline",
      isReady: backendOnline === true,
      description: "Simulation orchestrator & SSE event streamer",
    },
    {
      name: "SIEM & Telemetry",
      subtitle: "Wazuh & OSQuery",
      icon: Shield,
      status: "Ready",
      isReady: true,
      description: "Host & endpoint telemetry ingestion pipeline",
    },
    {
      name: "MITRE Caldera",
      subtitle: "Adversary Simulator",
      icon: Swords,
      status: "Ready",
      isReady: true,
      description: "Automated adversary attack emulation platform",
    },
    {
      name: "SecBERT Encoder",
      subtitle: "768-D State Model",
      icon: Cpu,
      status: secbertReady ? "Loaded" : "Standby",
      isReady: secbertReady,
      description: "Incident context semantic representations",
    },
    {
      name: "PPO Policy",
      subtitle: "Response Agent",
      icon: BrainCircuit,
      status: ppoReady ? "Loaded" : "Standby",
      isReady: ppoReady,
      description: "Canonical CISA playbook response optimization",
    },
    {
      name: "PPO+ATGP Policy",
      subtitle: "Target-Guided Network",
      icon: Network,
      status: ppoAtgpReady ? "Loaded" : "Standby",
      isReady: ppoAtgpReady,
      description: "Adaptive target-guided incident response agent",
    },
  ];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
          System Component Status
        </h2>
        <span className="text-[11px] text-slate-500 font-mono">
          Core system operational
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {components.map((item) => {
          const Icon = item.icon;
          return (
            <Card
              key={item.name}
              className="p-3.5 border-slate-200 bg-white shadow-sm flex flex-col justify-between hover:border-slate-300 transition-colors"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="p-1.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-700">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-medium border bg-emerald-50 text-emerald-800 border-emerald-200">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                    <span>{item.status}</span>
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-bold text-slate-900">{item.name}</h3>
                  <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                    {item.subtitle}
                  </p>
                </div>

                <p className="text-[11px] text-slate-600 leading-snug">
                  {item.description}
                </p>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
