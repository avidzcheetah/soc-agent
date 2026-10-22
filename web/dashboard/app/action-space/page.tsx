"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  Search,
  Filter,
  CheckCircle2,
  ExternalLink,
  Info,
  Layers,
  ArrowRight,
  Terminal,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { CisaPhaseBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { api } from "@/lib/api";

const ATTACK_TACTICS = [
  "Init. Access",
  "Execution",
  "Persistence",
  "Priv. Esc.",
  "Def. Evasion",
  "Cred. Access",
  "Discovery",
  "Lat. Mvmt.",
  "Exfiltration",
  "C2",
  "Impact",
];

const TACTIC_MAPPING: Record<number, string[]> = {
  0: ["Discovery"],
  1: ["Def. Evasion", "Discovery"],
  2: [
    "Init. Access",
    "Execution",
    "Persistence",
    "Def. Evasion",
    "Cred. Access",
    "Lat. Mvmt.",
    "Exfiltration",
    "C2",
  ],
  3: ["Init. Access", "Cred. Access", "C2"],
  4: ["Exfiltration", "C2"],
  5: ["Init. Access", "C2"],
  6: ["Lat. Mvmt.", "C2"],
  7: ["Lat. Mvmt.", "Exfiltration", "C2", "Impact"],
  8: ["Execution", "Persistence"],
  9: ["Init. Access", "Execution", "Persistence", "Def. Evasion"],
  10: ["Init. Access"],
  11: ["Cred. Access", "Lat. Mvmt."],
  12: ["Persistence", "Priv. Esc.", "Cred. Access"],
  13: ["Persistence"],
  14: ["Persistence", "Def. Evasion"],
  15: ["Def. Evasion"],
  16: ["Init. Access", "Priv. Esc."],
  17: [
    "Init. Access",
    "Execution",
    "Persistence",
    "Priv. Esc.",
    "Def. Evasion",
    "Cred. Access",
    "Discovery",
    "Lat. Mvmt.",
    "Exfiltration",
    "C2",
    "Impact",
  ],
  18: ["C2"],
  19: [
    "Init. Access",
    "Execution",
    "Persistence",
    "Priv. Esc.",
    "Def. Evasion",
    "Cred. Access",
    "Discovery",
    "Lat. Mvmt.",
    "Exfiltration",
    "C2",
    "Impact",
  ],
};

export default function ActionSpacePage() {
  const [loading, setLoading] = useState(true);
  const [actions, setActions] = useState<any[]>([]);
  const [selectedPhase, setSelectedPhase] = useState<string>("All");
  const [searchQuery, setSearchQuery] = useState<string>("");

  useEffect(() => {
    async function loadData() {
      try {
        const res = await api.getActions();
        setActions(res.actions || []);
      } catch (e) {
        console.error("Failed to load action space", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return <LoadingState message="Loading CISA 20-Action Space..." />;
  }

  const filteredActions = actions.filter((a) => {
    const matchesPhase = selectedPhase === "All" || a.phase.toLowerCase() === selectedPhase.toLowerCase();
    const matchesQuery =
      searchQuery === "" ||
      a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.threat_match.toLowerCase().includes(searchQuery.toLowerCase()) ||
      a.implementation.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesPhase && matchesQuery;
  });

  const phasesCount = {
    All: actions.length,
    Detection: actions.filter((a) => a.phase === "Detection").length,
    Containment: actions.filter((a) => a.phase === "Containment").length,
    Eradication: actions.filter((a) => a.phase === "Eradication").length,
    "All Phases": actions.filter((a) => a.phase === "All Phases").length,
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-900/90 via-[#0a1122]/80 to-slate-900/90 p-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-teal-500/10 text-teal-400 border border-teal-500/20">
              Authoritative 20-Action Discrete Space
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono text-slate-400 bg-slate-800/60 border border-slate-700/50">
              CISA Playbooks & MITRE CAR
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
            Canonical Action Space & ATT&CK Matrix
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 max-w-3xl leading-relaxed">
            The reinforcement learning agent outputs one of 20 discrete mitigation actions per alert. Actions are derived from the
            <strong className="text-slate-200"> CISA Federal Incident & Vulnerability Response Playbooks</strong> and ordered by increasing intervention severity to enable proportional response reward shaping.
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Phase Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-900 border border-slate-800 overflow-x-auto">
          {["All", "Detection", "Containment", "Eradication", "All Phases"].map((phase) => (
            <button
              key={phase}
              onClick={() => setSelectedPhase(phase)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors shrink-0 flex items-center gap-1.5 ${
                selectedPhase === phase
                  ? "bg-teal-500/15 text-teal-300 border border-teal-500/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <span>{phase}</span>
              <span className="text-[10px] font-mono px-1 rounded bg-slate-800 text-slate-400">
                {phasesCount[phase as keyof typeof phasesCount] || 0}
              </span>
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search action, threat, or implementation..."
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder:text-slate-400 focus:outline-none focus:border-teal-500"
          />
        </div>
      </div>

      {/* Canonical Action Space Table */}
      <Card className="overflow-hidden border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/90 text-slate-400 font-medium uppercase text-[10px] tracking-wider border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 w-12 text-center">Index</th>
                <th className="py-3 px-4">Action Name</th>
                <th className="py-3 px-4">CISA Phase</th>
                <th className="py-3 px-4">Threat Match Category</th>
                <th className="py-3 px-4">SOC Technical Implementation</th>
                <th className="py-3 px-4 text-center">Class Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredActions.map((act) => (
                <tr key={act.index} className="hover:bg-slate-900/50 transition-colors">
                  <td className="py-3 px-4 text-center font-mono font-bold text-slate-400">
                    {act.index}
                  </td>
                  <td className="py-3 px-4 font-mono font-semibold text-slate-100">
                    {act.name}
                  </td>
                  <td className="py-3 px-4">
                    <CisaPhaseBadge phase={act.phase} />
                  </td>
                  <td className="py-3 px-4 text-slate-300">
                    {act.threat_match}
                  </td>
                  <td className="py-3 px-4 text-slate-400 max-w-md">
                    {act.implementation}
                  </td>
                  <td className="py-3 px-4 text-center">
                    {act.zero_shot ? (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20">
                        Zero-Shot (0.0w)
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/20">
                        Standard
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* MITRE ATT&CK Tactic Coverage Matrix */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <div>
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <Layers className="h-4 w-4 text-teal-400" />
            <span>Action-to-ATT&CK Tactic Coverage Matrix</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Verified mapping showing which defensive response actions address each MITRE ATT&CK enterprise tactic.
          </p>
        </div>

        <Card className="overflow-hidden border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 font-medium uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4 w-48">Action Name</th>
                  {ATTACK_TACTICS.map((tac) => (
                    <th key={tac} className="py-3 px-2 text-center font-mono">
                      {tac}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                {actions.map((act) => {
                  const covered = TACTIC_MAPPING[act.index] || [];
                  return (
                    <tr key={act.index} className="hover:bg-slate-900/40">
                      <td className="py-2.5 px-4 font-semibold text-slate-200">
                        <span className="text-slate-400 text-[10px] mr-2">#{act.index}</span>
                        {act.name}
                      </td>
                      {ATTACK_TACTICS.map((tac) => {
                        const isPrimary = covered.includes(tac);
                        return (
                          <td key={tac} className="py-2.5 px-2 text-center">
                            {isPrimary ? (
                              <span
                                title={`${act.name} effectively addresses ${tac}`}
                                className="inline-block h-3 w-3 rounded-full bg-teal-400 shadow-sm shadow-teal-400/50"
                              />
                            ) : (
                              <span className="text-slate-700">·</span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="p-3 bg-slate-950 border-t border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-teal-400" />
                <span className="text-slate-300">Primary Effectiveness</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="text-slate-600 font-bold">·</span>
                <span>Not Primary Target</span>
              </span>
            </div>
            <span>Source: data/action_space.csv & CISA Playbooks</span>
          </div>
        </Card>
      </div>
    </div>
  );
}
