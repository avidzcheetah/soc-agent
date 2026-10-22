import React from "react";
import { cn, getStatusBadge, getCisaPhaseBadge } from "@/lib/utils";

interface StatusBadgeProps {
  status: string;
  className?: string;
  showDot?: boolean;
}

export function StatusBadge({ status, className, showDot = true }: StatusBadgeProps) {
  const badge = getStatusBadge(status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border",
        badge.bg,
        badge.text,
        badge.border,
        className
      )}
    >
      {showDot && (
        <span
          className={cn("h-1.5 w-1.5 rounded-full", {
            "bg-teal-400 animate-pulse": status.toLowerCase().includes("online") || status.toLowerCase().includes("connected") || status.toLowerCase().includes("active"),
            "bg-slate-400": status.toLowerCase().includes("planned") || status.toLowerCase().includes("offline") || status.toLowerCase().includes("disconnected"),
            "bg-amber-400": status.toLowerCase().includes("warning") || status.toLowerCase().includes("degraded"),
            "bg-rose-400": status.toLowerCase().includes("error") || status.toLowerCase().includes("negative"),
          })}
        />
      )}
      {status}
    </span>
  );
}

export function CisaPhaseBadge({ phase, className }: { phase: string; className?: string }) {
  const badge = getCisaPhaseBadge(phase);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium border",
        badge.bg,
        badge.text,
        badge.border,
        className
      )}
    >
      {phase}
    </span>
  );
}
