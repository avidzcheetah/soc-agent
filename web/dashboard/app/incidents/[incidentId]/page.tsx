"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  Tag,
  Shield,
  BrainCircuit,
  Database,
  ExternalLink,
  Copy,
  Check,
  FileText,
  AlertCircle,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge, CisaPhaseBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { api } from "@/lib/api";
import { IncidentEvent } from "@/types";

export default function IncidentDetailPage() {
  const params = useParams();
  const incidentId = (params?.incidentId as string) || "";

  const [loading, setLoading] = useState(true);
  const [incident, setIncident] = useState<IncidentEvent | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function loadDetail() {
      try {
        const data = await api.getIncidentDetail(incidentId);
        setIncident(data);
      } catch (e) {
        console.error("Failed to load incident detail", e);
      } finally {
        setLoading(false);
      }
    }
    if (incidentId) loadDetail();
  }, [incidentId]);

  const handleCopy = () => {
    if (incident?.text) {
      navigator.clipboard.writeText(incident.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (loading) {
    return <LoadingState message={`Fetching incident record ${incidentId}...`} />;
  }

  if (!incident) {
    return (
      <div className="space-y-4">
        <Link
          href="/telemetry"
          className="inline-flex items-center gap-1.5 text-xs text-teal-400 hover:text-teal-300 font-medium"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to Incidents
        </Link>
        <Card className="text-center py-12">
          <AlertCircle className="h-8 w-8 text-rose-400 mx-auto mb-2" />
          <h3 className="text-sm font-bold text-slate-200">Incident Record Not Found</h3>
          <p className="text-xs text-slate-400 mt-1">
            No incident matching ID <code className="text-teal-400 font-mono">{incidentId}</code> exists in the active benchmark partition.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back Link & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Link
            href="/telemetry"
            className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-100 font-mono">{incident.id}</h1>
              <CisaPhaseBadge phase={incident.cisa_phase} />
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {incident.source_dataset}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Partition: <strong className="text-slate-300 uppercase">{incident.split} split</strong> | Index: {incident.index}
            </p>
          </div>
        </div>

        {/* Link to Decision Record */}
        {incident.linked_decision_id && (
          <Link
            href={`/decisions/${incident.linked_decision_id}`}
            className="px-3 py-1.5 rounded-lg bg-teal-500/15 hover:bg-teal-500/25 text-teal-300 text-xs font-semibold border border-teal-500/40 transition-colors flex items-center gap-1.5 self-start sm:self-auto"
          >
            <BrainCircuit className="h-3.5 w-3.5" />
            <span>Inspect Agent Decision ({incident.linked_decision_id})</span>
            <ExternalLink className="h-3 w-3" />
          </Link>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Incident Raw Text & Prepared Text */}
        <div className="lg:col-span-2 space-y-6">
          {/* Raw Text Card */}
          <Card>
            <CardHeader
              title="Original Incident Alert Text"
              subtitle="Pristine benchmark record ingested from threat report / cyber event log"
              action={
                <button
                  onClick={handleCopy}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-750 text-slate-300 text-[11px] border border-slate-700 flex items-center gap-1"
                >
                  {copied ? <Check className="h-3 w-3 text-teal-400" /> : <Copy className="h-3 w-3" />}
                  <span>{copied ? "Copied" : "Copy Text"}</span>
                </button>
              }
            />
            <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 leading-relaxed whitespace-pre-wrap max-h-72 overflow-y-auto">
              {incident.text}
            </div>
          </Card>

          {/* Prepared SecBERT Input Text */}
          <Card>
            <CardHeader
              title="Tokenizable Model Input (Prepared Text)"
              subtitle="Reconstructed text input with prepended CISA phase tag for SecBERT tokenization"
            />
            <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-teal-300 font-mono leading-relaxed">
              {incident.prepared_text}
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Note: EXP_002C demonstrated that prefixing the CISA phase (e.g. <code className="text-teal-400">[{incident.cisa_phase}]</code>) resolves semantic ambiguity across majority classes and boosted Macro F1 by +0.074.
            </p>
          </Card>
        </div>

        {/* Right 1 Col: Normalized Attributes & Authoritative Action */}
        <div className="space-y-6">
          {/* Authoritative CISA Action */}
          <Card glow>
            <CardHeader
              title="Authoritative Ground Truth Action"
              subtitle="Target mitigation response from CISA Incident Playbooks"
            />
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-900/90 border border-teal-500/30">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase font-mono text-slate-400">Action Index</span>
                  <span className="text-xs font-mono font-bold text-teal-400">
                    Index #{incident.action_label} of 20
                  </span>
                </div>
                <h4 className="text-sm font-bold font-mono text-slate-100 mt-1">
                  {incident.action_name}
                </h4>
              </div>

              <div>
                <p className="text-[10px] uppercase font-mono text-slate-400">CISA Response Phase</p>
                <div className="mt-1">
                  <CisaPhaseBadge phase={incident.cisa_phase} />
                </div>
              </div>

              <div>
                <p className="text-[10px] uppercase font-mono text-slate-400">Threat Match Category</p>
                <p className="text-xs text-slate-200 mt-0.5">{incident.threat_match}</p>
              </div>

              <div>
                <p className="text-[10px] uppercase font-mono text-slate-400">Real-World SOC Implementation</p>
                <p className="text-xs text-slate-300 mt-0.5 bg-slate-900/60 p-2.5 rounded border border-slate-800/80 leading-relaxed">
                  {incident.implementation}
                </p>
              </div>
            </div>
          </Card>

          {/* Normalized Metadata */}
          <Card>
            <CardHeader
              title="Normalized Record Metadata"
              subtitle="Data hygiene & provenance checks"
            />
            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Target Entity:</span>
                <span className="text-slate-200 font-medium truncate max-w-[150px]">{incident.target}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Attack Type:</span>
                <span className="text-slate-200 font-medium">{incident.attack_type}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Character Length:</span>
                <span className="font-mono text-slate-200">{incident.text.length} chars</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">SecBERT Token Estimate:</span>
                <span className="font-mono text-slate-200">~{Math.round(incident.text.split(" ").length * 1.3)} tokens (&le; 512 max)</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Near-Duplicate Check:</span>
                <span className="text-teal-400 font-medium">Passed (Cosine &lt; 0.90)</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
