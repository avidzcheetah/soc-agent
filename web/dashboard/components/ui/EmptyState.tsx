import React from "react";
import { AlertCircle, HelpCircle } from "lucide-react";

export function EmptyState({
  title = "No records found",
  description = "No matching research records or evaluation data available.",
  action,
}: {
  title?: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center rounded-xl border border-dashed border-slate-200 bg-slate-50/60">
      <div className="p-3 bg-slate-100 rounded-full mb-3 text-slate-500">
        <HelpCircle className="h-6 w-6" />
      </div>
      <h4 className="text-sm font-semibold text-slate-800">{title}</h4>
      <p className="text-xs text-slate-500 mt-1 max-w-sm">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function UnavailableBanner({
  title = "Integration Unavailable in Current Phase",
  description,
}: {
  title?: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3 p-4 rounded-lg border border-slate-200 bg-slate-50 text-slate-700">
      <AlertCircle className="h-5 w-5 text-slate-500 shrink-0 mt-0.5" />
      <div className="text-xs">
        <p className="font-semibold text-slate-800">{title}</p>
        <p className="text-slate-500 mt-0.5">{description}</p>
      </div>
    </div>
  );
}
