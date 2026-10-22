"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { EvaluationArchitecture } from "@/types";

interface Props {
  architectures: EvaluationArchitecture[];
}

export function PolicyComparisonChart({ architectures }: Props) {
  // Structure data for Recharts side-by-side comparison
  const secbert = architectures.find((a) => a.id === "secbert_standalone");
  const ppo = architectures.find((a) => a.id === "ppo_core");

  const data = [
    {
      metric: "Top-1 Accuracy",
      SecBERT: secbert?.accuracy ? Number((secbert.accuracy * 100).toFixed(1)) : 0,
      PPO: ppo?.accuracy ? Number((ppo.accuracy * 100).toFixed(1)) : 0,
    },
    {
      metric: "Macro F1",
      SecBERT: secbert?.macro_f1 ? Number((secbert.macro_f1 * 100).toFixed(1)) : 0,
      PPO: ppo?.macro_f1 ? Number((ppo.macro_f1 * 100).toFixed(1)) : 0,
    },
    {
      metric: "Weighted F1",
      SecBERT: secbert?.weighted_f1 ? Number((secbert.weighted_f1 * 100).toFixed(1)) : 0,
      PPO: ppo?.weighted_f1 ? Number((ppo.weighted_f1 * 100).toFixed(1)) : 0,
    },
    {
      metric: "MCC (Matthews)",
      SecBERT: secbert?.mcc ? Number((secbert.mcc * 100).toFixed(1)) : 0,
      PPO: ppo?.mcc ? Number((ppo.mcc * 100).toFixed(1)) : 0,
    },
    {
      metric: "Top-2 Accuracy",
      SecBERT: secbert?.top2_accuracy ? Number((secbert.top2_accuracy * 100).toFixed(1)) : 0,
      PPO: ppo?.top2_accuracy ? Number((ppo.top2_accuracy * 100).toFixed(1)) : 0,
    },
  ];

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="metric"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0d1527",
              borderColor: "#1e293b",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#f1f5f9",
            }}
            formatter={(val: number) => [`${val}%`, ""]}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{ paddingBottom: "10px", fontSize: "11px" }}
          />
          <Bar
            name="SecBERT Standalone (EXP_002G)"
            dataKey="SecBERT"
            fill="#14b8a6"
            radius={[4, 4, 0, 0]}
          />
          <Bar
            name="Frozen SecBERT + PPO (v1.0 Core)"
            dataKey="PPO"
            fill="#38bdf8"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
