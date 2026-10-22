"use client";

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface TrajectoryPoint {
  step: number;
  reward_a: number | null;
  reward_b: number | null;
  cum_reward_a: number | null;
  cum_reward_b: number | null;
  action_a?: string;
  action_b?: string;
  is_correct_a?: boolean;
  is_correct_b?: boolean;
}

interface Props {
  trajectory: TrajectoryPoint[];
  policyAName?: string;
  policyBName?: string;
}

export function PolicyTrajectoryComparisonChart({
  trajectory,
  policyAName = "Policy A (PPO)",
  policyBName = "Policy B (PPO+ATGP)",
}: Props) {
  const data = trajectory.map((t) => ({
    step: `Step ${t.step}`,
    stepNum: t.step,
    [policyAName]: t.cum_reward_a,
    [policyBName]: t.cum_reward_b,
  }));

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="step"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            axisLine={{ stroke: "#334155" }}
            interval={Math.max(0, Math.floor(trajectory.length / 8))}
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
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{ paddingBottom: "10px", fontSize: "11px" }}
          />
          <Line
            type="monotone"
            dataKey={policyAName}
            name={policyAName}
            stroke="#38bdf8"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#38bdf8" }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey={policyBName}
            name={policyBName}
            stroke="#14b8a6"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "#14b8a6" }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
