import React from "react";
import { cn } from "@/lib/utils";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
}

export function Card({ children, className, glow = false, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-slate-800/80 bg-[#0d1527] p-5 text-slate-100 shadow-sm",
        glow && "border-teal-500/30 shadow-[0_0_15px_-3px_rgba(20,184,166,0.1)]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
  className,
}: {
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-4 flex items-start justify-between gap-4 border-b border-slate-800/60 pb-3", className)}>
      <div>
        <h3 className="text-base font-semibold text-slate-100 tracking-tight">{title}</h3>
        {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}
