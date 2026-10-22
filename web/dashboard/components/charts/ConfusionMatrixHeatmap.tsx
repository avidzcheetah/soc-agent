"use client";

import React, { useState } from "react";
import { ConfusionMatrixData } from "@/types";
import { cn } from "@/lib/utils";

interface Props {
  data: ConfusionMatrixData;
}

export function ConfusionMatrixHeatmap({ data }: Props) {
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    trueLabel: string;
    predLabel: string;
    val: number;
  } | null>(null);

  const { labels, matrix } = data;

  // Find max value in diagonal and off-diagonal for proportional coloring
  let maxDiag = 1;
  let maxOffDiag = 1;

  matrix.forEach((row, rIdx) => {
    row.forEach((val, cIdx) => {
      if (rIdx === cIdx) {
        if (val > maxDiag) maxDiag = val;
      } else {
        if (val > maxOffDiag) maxOffDiag = val;
      }
    });
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>Vertical: Ground Truth Label | Horizontal: Predicted Label</span>
        {hoveredCell && (
          <span className="font-mono text-slate-200 bg-slate-800 px-2.5 py-1 rounded border border-slate-700">
            True: <strong className="text-teal-400">{hoveredCell.trueLabel}</strong> → Predicted:{" "}
            <strong className="text-sky-400">{hoveredCell.predLabel}</strong> ({hoveredCell.val}{" "}
            samples)
          </span>
        )}
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="inline-block min-w-full">
          {/* Header Row (Predicted Labels) */}
          <div className="flex items-center">
            <div className="w-28 shrink-0 text-[10px] uppercase font-mono text-slate-400 tracking-wider text-right pr-3 font-semibold">
              Ground Truth
            </div>
            {labels.map((lbl, idx) => (
              <div
                key={idx}
                title={lbl}
                className="w-16 shrink-0 text-center text-[10px] font-mono text-slate-400 truncate px-1 font-semibold"
              >
                {lbl}
              </div>
            ))}
          </div>

          {/* Matrix Rows */}
          <div className="mt-1 space-y-1">
            {matrix.map((row, rIdx) => {
              const trueLabel = labels[rIdx];
              const rowTotal = row.reduce((a, b) => a + b, 0);

              return (
                <div key={rIdx} className="flex items-center">
                  {/* Row Label (True Label) */}
                  <div
                    title={trueLabel}
                    className="w-28 shrink-0 text-right pr-3 text-xs font-mono text-slate-700 truncate font-medium"
                  >
                    {trueLabel}
                  </div>

                  {/* Row Cells */}
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    const predLabel = labels[cIdx];
                    const ratio = isDiagonal ? val / maxDiag : val / maxOffDiag;

                    return (
                      <div
                        key={cIdx}
                        onMouseEnter={() =>
                          setHoveredCell({
                            row: rIdx,
                            col: cIdx,
                            trueLabel,
                            predLabel,
                            val,
                          })
                        }
                        onMouseLeave={() => setHoveredCell(null)}
                        className={cn(
                          "w-16 h-10 shrink-0 m-0.5 flex items-center justify-center rounded text-xs font-mono font-medium transition-transform hover:scale-105 cursor-pointer border",
                          isDiagonal
                            ? val > 0
                              ? "bg-teal-50 border-teal-300 text-teal-900 font-bold"
                              : "bg-slate-50 border-slate-200 text-slate-400"
                            : val > 0
                            ? "bg-rose-50 border-rose-300 text-rose-800 font-bold"
                            : "bg-slate-50/60 border-slate-200 text-slate-400"
                        )}
                        style={{
                          backgroundColor: isDiagonal
                            ? `rgba(20, 184, 166, ${Math.max(0.12, ratio * 0.4)})`
                            : val > 0
                            ? `rgba(244, 63, 94, ${Math.max(0.12, ratio * 0.4)})`
                            : undefined,
                        }}
                      >
                        {val}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-100">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded bg-teal-500/20 border border-teal-500/40" />
            <span>Correct Classification (Diagonal)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded bg-rose-500/20 border border-rose-500/40" />
            <span>Misclassification Error (Off-Diagonal)</span>
          </div>
        </div>
        <span>Evaluation Split: 1,539 Holdout Records</span>
      </div>
    </div>
  );
}
