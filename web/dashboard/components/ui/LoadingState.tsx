import React from "react";
import { Loader2 } from "lucide-react";

export function LoadingState({ message = "Loading research data..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <Loader2 className="h-8 w-8 animate-spin text-teal-400 mb-3" />
      <p className="text-sm text-slate-400">{message}</p>
    </div>
  );
}

export function SkeletonRow({ cols = 5 }: { cols?: number }) {
  return (
    <tr className="animate-pulse border-b border-slate-800/40">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="py-4 px-4">
          <div className="h-4 bg-slate-800/60 rounded w-3/4"></div>
        </td>
      ))}
    </tr>
  );
}
