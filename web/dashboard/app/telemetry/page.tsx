"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Radio,
  Search,
  Filter,
  ArrowUpDown,
  ExternalLink,
  ChevronLeft,
  ChevronRight,
  Database,
  AlertCircle,
  Eye,
  Server,
  Info,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge, CisaPhaseBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState, UnavailableBanner } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import { IncidentEvent, TelemetrySource } from "@/types";
import { cn } from "@/lib/utils";

export default function TelemetryPage() {
  const [loading, setLoading] = useState(true);
  const [sources, setSources] = useState<TelemetrySource[]>([]);
  const [events, setEvents] = useState<IncidentEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(25);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [phaseFilter, setPhaseFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<"id" | "phase" | "action">("id");
  const [sortAsc, setSortAsc] = useState(true);

  // Fetch telemetry sources
  useEffect(() => {
    async function fetchSources() {
      try {
        const res = await api.getTelemetrySources();
        setSources(res.sources);
      } catch (e) {
        console.error("Failed to load telemetry sources", e);
      }
    }
    fetchSources();
  }, []);

  // Fetch events with pagination & filters
  useEffect(() => {
    async function fetchEvents() {
      setLoading(true);
      try {
        const res = await api.getTelemetryEvents(page, limit, sourceFilter, phaseFilter, searchQuery);
        setEvents(res.events);
        setTotal(res.total);
      } catch (e) {
        console.error("Failed to load telemetry events", e);
      } finally {
        setLoading(false);
      }
    }
    fetchEvents();
  }, [page, limit, sourceFilter, phaseFilter, searchQuery]);

  const handleSort = (field: "id" | "phase" | "action") => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  const sortedEvents = [...events].sort((a, b) => {
    let comparison = 0;
    if (sortField === "id") comparison = a.id.localeCompare(b.id);
    else if (sortField === "phase") comparison = a.cisa_phase.localeCompare(b.cisa_phase);
    else if (sortField === "action") comparison = a.action_name.localeCompare(b.action_name);
    return sortAsc ? comparison : -comparison;
  });

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-100">Telemetry Sources & Incident Repository</h1>
        <p className="text-xs text-slate-400 mt-1">
          Monitor source ingestion status, review curated cyber event streams, and inspect normalized incident records.
        </p>
      </div>

      {/* Telemetry Sources Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-slate-200 uppercase tracking-wider font-mono">
            Configured Telemetry Ingestion Sources
          </h2>
          <span className="text-[11px] text-slate-400 font-mono">
            Live SIEM/EDR: Disconnected (Phase 3) | Historical Benchmarks: Connected
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sources.map((src, i) => (
            <Card key={i} className="flex flex-col justify-between">
              <div>
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                      <Radio className="h-4 w-4 text-teal-400" />
                      {src.name}
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">{src.category}</p>
                  </div>
                  <StatusBadge status={src.status} />
                </div>

                <p className="text-xs text-slate-300 mt-3 leading-relaxed">{src.note}</p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-1.5 text-[11px]">
                <div className="flex justify-between text-slate-400">
                  <span>Endpoint:</span>
                  <span className="font-mono text-slate-300 truncate max-w-[170px]" title={src.endpoint || "N/A"}>
                    {src.endpoint || "Not configured"}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Processed Records:</span>
                  <span className="font-mono font-semibold text-teal-400">
                    {src.processed_count.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Collection Interval:</span>
                  <span className="text-slate-300">
                    {src.collection_interval_sec ? `${src.collection_interval_sec}s` : "Historical Static"}
                  </span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Parsing Errors:</span>
                  <span className="text-slate-300">{src.parsing_errors}</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <Card>
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search by incident ID, target host, keyword, or action..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              className="w-full pl-9 pr-4 py-2 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-2.5">
            {/* Source Dataset */}
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className="hidden sm:inline">Source:</span>
              <select
                value={sourceFilter}
                onChange={(e) => {
                  setSourceFilter(e.target.value);
                  setPage(1);
                }}
                className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-teal-500"
              >
                <option value="all">All Sources (15.3k)</option>
                <option value="CISSM">CISSM (14.3k)</option>
                <option value="rcATT">rcATT (1.0k)</option>
              </select>
            </div>

            {/* CISA Phase */}
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className="hidden sm:inline">Phase:</span>
              <select
                value={phaseFilter}
                onChange={(e) => {
                  setPhaseFilter(e.target.value);
                  setPage(1);
                }}
                className="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-teal-500"
              >
                <option value="all">All CISA Phases</option>
                <option value="Detection">Detection</option>
                <option value="Containment">Containment</option>
                <option value="Eradication">Eradication</option>
                <option value="All Phases">All Phases</option>
              </select>
            </div>

            {/* Page Size */}
            <select
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value));
                setPage(1);
              }}
              className="bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-slate-300 focus:outline-none"
            >
              <option value={10}>10 / page</option>
              <option value={25}>25 / page</option>
              <option value={50}>50 / page</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Incident Event Table */}
      <Card>
        <CardHeader
          title="Cybersecurity Incident Stream"
          subtitle={`Showing ${events.length} of ${total.toLocaleString()} filtered records`}
          action={
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Sort:</span>
              <button
                onClick={() => handleSort("id")}
                className="px-2 py-1 rounded bg-slate-850 hover:bg-slate-800 text-slate-300 border border-slate-750 flex items-center gap-1"
              >
                ID <ArrowUpDown className="h-3 w-3" />
              </button>
              <button
                onClick={() => handleSort("phase")}
                className="px-2 py-1 rounded bg-slate-850 hover:bg-slate-800 text-slate-300 border border-slate-750 flex items-center gap-1"
              >
                Phase <ArrowUpDown className="h-3 w-3" />
              </button>
            </div>
          }
        />

        {loading ? (
          <LoadingState message="Querying incident repository..." />
        ) : sortedEvents.length === 0 ? (
          <EmptyState
            title="No matching events found"
            description="Try modifying search keywords or resetting source/phase filters."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="research-table">
              <thead>
                <tr>
                  <th className="w-32">Event ID</th>
                  <th className="w-24">Source</th>
                  <th className="w-32">CISA Phase</th>
                  <th className="w-44">Target / Entity</th>
                  <th>Incident Description</th>
                  <th className="w-44">Authoritative Action</th>
                  <th className="w-24 text-right">Inspect</th>
                </tr>
              </thead>
              <tbody>
                {sortedEvents.map((evt) => (
                  <tr key={evt.id} className="group transition-colors">
                    <td className="font-mono text-xs text-teal-400 font-medium">
                      <Link
                        href={`/incidents/${evt.id}`}
                        className="hover:underline flex items-center gap-1"
                      >
                        {evt.id}
                      </Link>
                    </td>
                    <td>
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-[10px] font-mono border font-medium",
                          evt.source_dataset === "CISSM"
                            ? "bg-sky-500/10 text-sky-300 border-sky-500/20"
                            : "bg-purple-500/10 text-purple-300 border-purple-500/20"
                        )}
                      >
                        {evt.source_dataset}
                      </span>
                    </td>
                    <td>
                      <CisaPhaseBadge phase={evt.cisa_phase} />
                    </td>
                    <td className="text-xs text-slate-300 truncate max-w-[180px]" title={evt.target}>
                      {evt.target}
                    </td>
                    <td className="text-xs text-slate-300 max-w-md">
                      <p className="line-clamp-2" title={evt.text}>
                        {evt.summary}
                      </p>
                    </td>
                    <td>
                      <span className="font-mono text-xs text-slate-200 bg-slate-900 px-2 py-1 rounded border border-slate-800 inline-block truncate max-w-[170px]" title={evt.action_name}>
                        {evt.action_name} ({evt.action_label})
                      </span>
                    </td>
                    <td className="text-right">
                      <Link
                        href={`/incidents/${evt.id}`}
                        className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-teal-400 inline-flex transition-colors"
                        title="View Incident Detail"
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

        {/* Pagination Bar */}
        <div className="flex items-center justify-between pt-4 mt-2 border-t border-slate-800/80 text-xs text-slate-400">
          <div>
            Showing Page <strong className="text-slate-200">{page}</strong> of{" "}
            <strong className="text-slate-200">{totalPages}</strong> ({total.toLocaleString()} total incidents)
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page <= 1 || loading}
              className="px-2.5 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-800 transition-colors flex items-center gap-1"
            >
              <ChevronLeft className="h-3.5 w-3.5" /> Previous
            </button>
            <span className="font-mono text-slate-300 px-2">
              {page} / {totalPages}
            </span>
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
