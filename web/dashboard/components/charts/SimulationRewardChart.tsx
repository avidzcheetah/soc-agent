"use client";

import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { SimulationStep } from "@/types";

interface Props {
  steps: SimulationStep[];
}

export function SimulationRewardChart({ steps }: Props) {
  const data = steps.map((s) => ({
    step: `Step ${s.step}`,
    reward: s.reward,
    cumulative: s.cumulative_reward,
    isCorrect: s.is_correct,
  }));

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <defs>
            <linearGradient id="rewardGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#14b8a6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="step"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            interval={Math.floor(steps.length / 8)}
          />
          <YAxis
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
          />
          <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 2" />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0d1527",
              borderColor: "#1e293b",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#f1f5f9",
            }}
            formatter={(value: any, name: string) => [
              value,
              name === "cumulative" ? "Cumulative Reward" : name,
            ]}
          />
          <Area
            type="monotone"
            dataKey="cumulative"
            name="Cumulative Reward"
            stroke="#14b8a6"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#rewardGrad)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
