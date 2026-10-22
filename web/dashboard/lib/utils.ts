import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(dateString: string | null | undefined): string {
  if (!dateString) return "Not available";
  try {
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return dateString;
    return d.toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return dateString;
  }
}

export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || isNaN(value)) {
    return "Not measured";
  }
  return `${(value * 100).toFixed(decimals)}%`;
}

export function getCisaPhaseBadge(phase: string | undefined): {
  bg: string;
  text: string;
  border: string;
} {
  switch (phase?.toLowerCase()) {
    case "detection":
      return {
        bg: "bg-sky-500/10",
        text: "text-sky-400",
        border: "border-sky-500/30",
      };
    case "containment":
      return {
        bg: "bg-amber-500/10",
        text: "text-amber-400",
        border: "border-amber-500/30",
      };
    case "eradication":
      return {
        bg: "bg-emerald-500/10",
        text: "text-emerald-400",
        border: "border-emerald-500/30",
      };
    case "all phases":
      return {
        bg: "bg-purple-500/10",
        text: "text-purple-400",
        border: "border-purple-500/30",
      };
    default:
      return {
        bg: "bg-slate-500/10",
        text: "text-slate-400",
        border: "border-slate-500/30",
      };
  }
}

export function getStatusBadge(status: string | undefined): {
  bg: string;
  text: string;
  border: string;
} {
  const s = status?.toLowerCase() || "";
  if (s.includes("connected") || s.includes("online") || s.includes("healthy") || s.includes("completed") || s.includes("active") || s.includes("locked")) {
    return {
      bg: "bg-teal-500/10",
      text: "text-teal-400",
      border: "border-teal-500/30",
    };
  }
  if (s.includes("planned") || s.includes("unmeasured") || s.includes("offline") || s.includes("disconnected") || s.includes("inactive")) {
    return {
      bg: "bg-slate-700/30",
      text: "text-slate-400",
      border: "border-slate-600/30",
    };
  }
  if (s.includes("warning") || s.includes("degraded") || s.includes("mismatch") || s.includes("regression")) {
    return {
      bg: "bg-amber-500/10",
      text: "text-amber-400",
      border: "border-amber-500/30",
    };
  }
  if (s.includes("error") || s.includes("fail") || s.includes("negative")) {
    return {
      bg: "bg-rose-500/10",
      text: "text-rose-400",
      border: "border-rose-500/30",
    };
  }
  return {
    bg: "bg-slate-800/40",
    text: "text-slate-400",
    border: "border-slate-700/40",
  };
}
