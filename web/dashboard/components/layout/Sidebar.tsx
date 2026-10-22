"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  PlaySquare,
  Network,
  Award,
  Activity,
  ShieldCheck,
  ChevronRight,
  Server,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  {
    name: "Overview",
    href: "/overview",
    icon: LayoutDashboard,
    badge: null,
  },
  {
    name: "Simulation Control",
    href: "/",
    icon: PlaySquare,
    badge: "Main",
  },
  {
    name: "Architecture",
    href: "/architecture",
    icon: Network,
    badge: "React Flow",
  },
  {
    name: "Results & Evaluation",
    href: "/results",
    icon: Award,
    badge: "EXP_002G",
  },
  {
    name: "System Health",
    href: "/system-health",
    icon: Activity,
    badge: null,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 shrink-0 border-r border-slate-800 bg-[#060a14] flex flex-col justify-between min-h-screen">
      {/* Brand Header */}
      <div>
        <div className="p-5 border-b border-slate-800/80">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-teal-500/10 border border-teal-500/20 rounded-lg text-teal-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-100 tracking-tight leading-tight">
                SOC Agent
              </h1>
              <p className="text-[10px] text-teal-400 font-medium tracking-wide">
                Transformer-Based Incident Response
              </p>
            </div>
          </div>
          <div className="mt-3 inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800/60 border border-slate-700/50 text-[11px] text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-teal-400" />
            <span>Research Control Panel</span>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors",
                  isActive
                    ? "bg-teal-500/15 text-teal-300 border border-teal-500/30"
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                )}
              >
                <div className="flex items-center gap-3">
                  <Icon className={cn("h-4 w-4", isActive ? "text-teal-400" : "text-slate-400")} />
                  <span>{item.name}</span>
                </div>
                {item.badge ? (
                  <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300 border border-slate-700 font-mono">
                    {item.badge}
                  </span>
                ) : isActive ? (
                  <ChevronRight className="h-3 w-3 text-teal-400" />
                ) : null}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-900/30 space-y-2 text-xs">
        <div>
          <p className="text-[11px] font-semibold text-slate-300">Research Platform</p>
          <p className="text-[10px] text-slate-400">
            SecBERT (768d) + PPO / PPO+ATGP
          </p>
        </div>
        <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-400 font-mono">
          <span>Action Space: 20</span>
          <span className="text-teal-400">Gymnasium</span>
        </div>
      </div>
    </aside>
  );
}
