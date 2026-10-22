"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { DecisionProbability } from "@/types";

interface Props {
  probabilities: DecisionProbability[];
}

export function ActionDistributionChart({ probabilities }: Props) {
  const data = probabilities.map((p) => ({
    name: p.name,
    probability: Number((p.probability * 100).toFixed(1)),
    isChosen: p.is_chosen,
    isGroundTruth: p.is_ground_truth,
    phase: p.phase,
  }));

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 110, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: "#94a3b8", fontSize: 10 }}
            axisLine={{ stroke: "#334155" }}
            unit="%"
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#cbd5e1", fontSize: 10 }}
            axisLine={{ stroke: "#334155" }}
            interval={0}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0d1527",
              borderColor: "#1e293b",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#f1f5f9",
            }}
            formatter={(value: any, name: string, props: any) => {
              const { isChosen, isGroundTruth, phase } = props.payload;
              const tags = [];
              if (isChosen) tags.push("SELECTED ACTION");
              if (isGroundTruth) tags.push("GROUND TRUTH");
              return [
                `${value}% (${phase}) ${tags.length > 0 ? "— [" + tags.join(", ") + "]" : ""}`,
                "Policy Probability",
              ];
            }}
          />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => {
              let fill = "#334155"; // Default low prob
              if (entry.isChosen && entry.isGroundTruth) {
                fill = "#14b8a6"; // Correct selection: teal
              } else if (entry.isChosen && !entry.isGroundTruth) {
                fill = "#f43f5e"; // Mistake selection: red
              } else if (!entry.isChosen && entry.isGroundTruth) {
                fill = "#eab308"; // Missed ground truth: amber
              } else if (entry.probability > 3) {
                fill = "#38bdf8"; // Candidate: sky blue
              }
              return <Cell key={`cell-${index}`} fill={fill} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
