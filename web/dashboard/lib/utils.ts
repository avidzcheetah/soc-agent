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
        bg: "bg-sky-50",
        text: "text-sky-700 font-medium",
        border: "border-sky-200",
      };
    case "containment":
      return {
        bg: "bg-amber-50",
        text: "text-amber-800 font-medium",
        border: "border-amber-200",
      };
    case "eradication":
      return {
        bg: "bg-emerald-50",
        text: "text-emerald-700 font-medium",
        border: "border-emerald-200",
      };
    case "all phases":
      return {
        bg: "bg-purple-50",
        text: "text-purple-700 font-medium",
        border: "border-purple-200",
      };
    default:
      return {
        bg: "bg-slate-100",
        text: "text-slate-700 font-medium",
        border: "border-slate-200",
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
      bg: "bg-teal-50",
      text: "text-teal-800 font-medium",
      border: "border-teal-200",
    };
  }
  if (s.includes("planned") || s.includes("unmeasured") || s.includes("offline") || s.includes("disconnected") || s.includes("inactive")) {
    return {
      bg: "bg-slate-100",
      text: "text-slate-600 font-medium",
      border: "border-slate-200",
    };
  }
  if (s.includes("warning") || s.includes("degraded") || s.includes("mismatch") || s.includes("regression")) {
    return {
      bg: "bg-amber-50",
      text: "text-amber-800 font-medium",
      border: "border-amber-200",
    };
  }
  if (s.includes("error") || s.includes("fail") || s.includes("negative")) {
    return {
      bg: "bg-rose-50",
      text: "text-rose-700 font-medium",
      border: "border-rose-200",
    };
  }
  return {
    bg: "bg-slate-100",
    text: "text-slate-700 font-medium",
    border: "border-slate-200",
  };
}
