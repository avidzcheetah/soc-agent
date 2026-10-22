"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  BrainCircuit,
  Search,
  Filter,
  CheckCircle2,
  XCircle,
  Eye,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Info,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge, CisaPhaseBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import { Decision } from "@/types";

export default function DecisionsListPage() {
  const [loading, setLoading] = useState(true);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(20);
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    async function loadDecisions() {
      setLoading(true);
      try {
        const res = await api.getDecisions(page, limit, statusFilter);
        setDecisions(res.decisions);
        setTotal(res.total);
      } catch (e) {
        console.error("Failed to load decisions", e);
      } finally {
        setLoading(false);
      }
    }
    loadDecisions();
  }, [page, limit, statusFilter]);

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-100">AI Decision Inspector</h1>
        <p className="text-xs text-slate-400 mt-1">
          Investigate how the frozen SecBERT 768-dimensional state encoder and PPO policy select mitigation actions in the SOC environment.
        </p>
      </div>

      {/* Filter Card */}
      <Card>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Filter Outcome:</span>
            <div className="flex rounded-lg bg-slate-900 p-0.5 border border-slate-800 text-xs">
              <button
                onClick={() => {
                  setStatusFilter("all");
                  setPage(1);
                }}
                className={`px-3 py-1 rounded-md transition-colors ${
                  statusFilter === "all" ? "bg-teal-500/20 text-teal-300 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                All Decisions
              </button>
              <button
                onClick={() => {
                  setStatusFilter("correct");
                  setPage(1);
                }}
                className={`px-3 py-1 rounded-md transition-colors flex items-center gap-1 ${
                  statusFilter === "correct" ? "bg-teal-500/20 text-teal-300 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <CheckCircle2 className="h-3 w-3 text-teal-400" /> Matches (+1.0)
              </button>
              <button
                onClick={() => {
                  setStatusFilter("mismatch");
                  setPage(1);
                }}
                className={`px-3 py-1 rounded-md transition-colors flex items-center gap-1 ${
                  statusFilter === "mismatch" ? "bg-rose-500/20 text-rose-300 font-semibold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <XCircle className="h-3 w-3 text-rose-400" /> Mismatches (-1.0)
              </button>
            </div>
          </div>

          <div className="text-xs text-slate-400 font-mono">
            Evaluated Policy: <strong className="text-slate-200">v1.0-ppo-core</strong> | Encoder: <strong className="text-teal-400">EXP_20260727_001</strong>
          </div>
        </div>
      </Card>

      {/* Decisions Table */}
      <Card>
        <CardHeader
          title="Policy Decision Records"
          subtitle={`Showing ${decisions.length} of ${total} recorded agent decisions`}
        />

        {loading ? (
          <LoadingState message="Loading agent decisions..." />
        ) : decisions.length === 0 ? (
          <EmptyState title="No decisions found" description="Try selecting a different filter." />
        ) : (
          <div className="overflow-x-auto">
            <table className="research-table">
              <thead>
                <tr>
                  <th className="w-32">Decision ID</th>
                  <th className="w-32">Incident Link</th>
                  <th className="w-28">CISA Phase</th>
                  <th>Chosen Action</th>
                  <th>Ground Truth Action</th>
                  <th className="w-24 text-center">Reward</th>
                  <th className="w-24 text-center">Critic V(s)</th>
                  <th className="w-20 text-right">Inspect</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((d) => (
                  <tr key={d.id}>
                    <td className="font-mono text-xs text-teal-400 font-medium">
                      <Link href={`/decisions/${d.id}`} className="hover:underline">
                        {d.id}
                      </Link>
                    </td>
                    <td className="font-mono text-xs text-slate-300">
                      <Link href={`/incidents/${d.incident_id}`} className="hover:underline text-sky-400">
                        {d.incident_id}
                      </Link>
                    </td>
                    <td>
                      <CisaPhaseBadge phase={d.cisa_phase} />
                    </td>
                    <td>
                      <div className="flex items-center gap-1.5 font-mono text-xs">
                        {d.is_correct ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-teal-400 shrink-0" />
                        ) : (
                          <XCircle className="h-3.5 w-3.5 text-rose-400 shrink-0" />
                        )}
                        <span className="text-slate-200 font-medium">{d.chosen_action_name}</span>
                        <span className="text-slate-400">({d.chosen_action})</span>
                      </div>
                    </td>
                    <td className="text-xs font-mono text-slate-400">
                      {d.ground_truth_name} ({d.ground_truth_action})
                    </td>
                    <td className="text-center">
                      <span
                        className={`inline-block font-mono text-xs font-bold px-2 py-0.5 rounded ${
                          d.reward > 0
                            ? "bg-teal-500/10 text-teal-300 border border-teal-500/30"
                            : "bg-rose-500/10 text-rose-300 border border-rose-500/30"
                        }`}
                      >
                        {d.reward > 0 ? "+1.0" : "-1.0"}
                      </span>
                    </td>
                    <td className="text-center font-mono text-xs text-slate-300">
                      {d.critic_value.toFixed(2)}
                    </td>
                    <td className="text-right">
                      <Link
                        href={`/decisions/${d.id}`}
                        className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-teal-400 inline-flex transition-colors"
                        title="Inspect Decision"
                      >
                        <Eye className="h-4 w-4" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between pt-4 mt-2 border-t border-slate-800/80 text-xs text-slate-400">
          <div>
            Page <strong className="text-slate-200">{page}</strong> of{" "}
            <strong className="text-slate-200">{totalPages}</strong>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page <= 1 || loading}
              className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-800 transition-colors flex items-center gap-1"
            >
              <ChevronLeft className="h-3.5 w-3.5" /> Previous
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page >= totalPages || loading}
              className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-800 transition-colors flex items-center gap-1"
            >
              Next <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
}
