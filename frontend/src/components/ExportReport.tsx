import React, { useState } from "react";
import { motion } from "framer-motion";
import { ChatMessage, TYPE_LABEL } from "../types";
import jsPDF from "jspdf";

interface Props { messages: ChatMessage[]; modelLabel: string; }

// ── Builders ─────────────────────────────────────────────────────────────────

function buildMarkdown(messages: ChatMessage[], modelLabel: string): string {
  const now = new Date().toISOString();
  const asMsgs = messages.filter(m => m.role === "assistant" && m.deception);
  const avg = asMsgs.length
    ? asMsgs.reduce((s, m) => s + (m.deception?.score ?? 0), 0) / asMsgs.length
    : 0;
  const typeCounts: Record<string, number> = {};
  asMsgs.forEach(m => {
    const t = m.deception?.deception_type ?? "none";
    typeCounts[t] = (typeCounts[t] ?? 0) + 1;
  });

  const lines = [
    "# DeceptiScope v2 — Deception Audit Report",
    "",
    `**Generated:** ${now}`,
    `**Model:** ${modelLabel}`,
    `**Total turns:** ${asMsgs.length}`,
    `**Average deception score:** ${(avg * 100).toFixed(1)}%`,
    "",
    "## Deception Type Distribution",
    "",
    ...Object.entries(typeCounts).map(([t, n]) => `- **${TYPE_LABEL[t] ?? t}**: ${n}`),
    "",
    "## Conversation Transcript",
    "",
  ];

  let turn = 0;
  messages.forEach(msg => {
    if (msg.role === "assistant") turn++;
    lines.push(`### Turn ${turn} — ${msg.role === "user" ? "User" : "Assistant"}`);
    lines.push("", msg.content, "");
    if (msg.deception) {
      const d = msg.deception;
      lines.push(
        `> **Score:** ${(d.score * 100).toFixed(1)}%  ` +
        `**Type:** ${TYPE_LABEL[d.deception_type] ?? d.deception_type}  ` +
        `**Confidence:** ${(d.confidence * 100).toFixed(0)}%`
      );
      if (d.explanation) lines.push(`> ${d.explanation}`);
      if (d.high_risk_tokens.length > 0)
        lines.push(
          `> High-risk tokens: ${d.high_risk_tokens.slice(0, 5)
            .map(t => `\`${t.token}\` (${(t.risk_score * 100).toFixed(0)}%)`)
            .join(", ")}`
        );
      lines.push("");
    }
  });

  lines.push("---", "*DeceptiScope v2 — Schmidt Sciences 2026 Interpretability RFP*");
  return lines.join("\n");
}

function buildJSON(messages: ChatMessage[], modelLabel: string) {
  const asMsgs = messages.filter(m => m.deception);
  return {
    generated: new Date().toISOString(),
    model: modelLabel,
    summary: {
      total_turns: messages.filter(m => m.role === "assistant").length,
      avg_score: asMsgs.length
        ? asMsgs.reduce((s, m) => s + (m.deception?.score ?? 0), 0) / asMsgs.length
        : 0,
    },
    messages: messages.map(m => ({
      role: m.role, content: m.content,
      timestamp: m.timestamp, deception: m.deception ?? null,
    })),
  };
}

function buildCSV(messages: ChatMessage[]): string {
  const rows = ["turn,role,score,type,confidence,explanation"];
  let turn = 0;
  messages.forEach(m => {
    if (m.role === "assistant") turn++;
    const d = m.deception;
    rows.push([
      turn, m.role,
      d ? (d.score * 100).toFixed(1) : "",
      d?.deception_type ?? "",
      d ? (d.confidence * 100).toFixed(0) : "",
      `"${(d?.explanation ?? "").replace(/"/g, "'")}"`,
    ].join(","));
  });
  return rows.join("\n");
}

async function buildPDF(messages: ChatMessage[], modelLabel: string): Promise<void> {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const W = 210, M = 18, lh = 5.5, maxW = W - M * 2;
  let y = M;

  const write = (text: string, size: number, rgb: [number,number,number], bold = false) => {
    doc.setFontSize(size);
    doc.setTextColor(...rgb);
    doc.setFont("helvetica", bold ? "bold" : "normal");
    doc.splitTextToSize(text, maxW).forEach((line: string) => {
      if (y > 282) { doc.addPage(); y = M; }
      doc.text(line, M, y);
      y += lh;
    });
  };

  const rule = () => {
    doc.setDrawColor(30, 39, 48);
    doc.line(M, y, W - M, y);
    y += 4;
  };

  // Cover band
  doc.setFillColor(8, 12, 16);
  doc.rect(0, 0, W, 38, "F");
  doc.setFontSize(18); doc.setFont("helvetica", "bold");
  doc.setTextColor(59, 130, 246);
  doc.text("DeceptiScope v2", M, 16);
  doc.setFontSize(8); doc.setFont("helvetica", "normal");
  doc.setTextColor(122, 136, 153);
  doc.text("Deception Audit Report", M, 24);
  doc.text(`${modelLabel}  ·  ${new Date().toLocaleString()}`, M, 31);
  y = 46;

  // Summary
  const asMsgs = messages.filter(m => m.role === "assistant" && m.deception);
  const avg = asMsgs.length
    ? asMsgs.reduce((s, m) => s + (m.deception?.score ?? 0), 0) / asMsgs.length
    : 0;
  write("Summary", 11, [212, 220, 232], true); y += 1;
  write(`${asMsgs.length} turns  ·  Average deception score: ${(avg * 100).toFixed(1)}%`, 8, [122, 136, 153]);
  y += 2; rule();

  // Type distribution
  const typeCounts: Record<string, number> = {};
  asMsgs.forEach(m => { const t = m.deception?.deception_type ?? "none"; typeCounts[t] = (typeCounts[t] ?? 0) + 1; });
  write("Deception Type Distribution", 10, [212, 220, 232], true); y += 1;
  Object.entries(typeCounts).forEach(([t, n]) => {
    write(`  ${TYPE_LABEL[t] ?? t}: ${n}`, 8, [122, 136, 153]);
  });
  y += 2; rule();

  // Transcript
  write("Conversation Transcript", 10, [212, 220, 232], true); y += 2;
  let turn = 0;
  messages.forEach(msg => {
    if (msg.role === "assistant") turn++;
    const isUser = msg.role === "user";
    write(`Turn ${turn} — ${isUser ? "User" : "Assistant"}`, 8,
      isUser ? [59, 130, 246] : [122, 136, 153], true);
    write(msg.content.slice(0, 500) + (msg.content.length > 500 ? "…" : ""), 7, [212, 220, 232]);
    if (msg.deception) {
      const d = msg.deception;
      const sc: [number,number,number] = d.score < 0.3 ? [34,197,94] : d.score < 0.6 ? [234,179,8] : [239,68,68];
      write(`  ${(d.score * 100).toFixed(1)}%  ${TYPE_LABEL[d.deception_type] ?? d.deception_type}  conf ${(d.confidence * 100).toFixed(0)}%`, 7, sc);
      if (d.explanation) write(`  ${d.explanation}`, 6, [122, 136, 153]);
    }
    y += 1.5;
  });

  // Page footers
  const pages = doc.getNumberOfPages();
  for (let i = 1; i <= pages; i++) {
    doc.setPage(i);
    doc.setFontSize(6); doc.setTextColor(74, 85, 104);
    doc.text(`DeceptiScope v2  ·  Page ${i} of ${pages}  ·  Schmidt Sciences 2026 Interpretability RFP`, M, 292);
  }

  doc.save(`deceptiscope-audit-${Date.now()}.pdf`);
}

// ── Component ─────────────────────────────────────────────────────────────────

type ExportType = "pdf" | "md" | "json" | "csv";

const BTNS: { id: ExportType; label: string; primary?: boolean }[] = [
  { id: "pdf",  label: "PDF",      primary: true },
  { id: "md",   label: "Markdown" },
  { id: "json", label: "JSON"     },
  { id: "csv",  label: "CSV"      },
];

export const ExportReport: React.FC<Props> = ({ messages, modelLabel }) => {
  const [loading, setLoading] = useState<ExportType | null>(null);
  const hasData = messages.some(m => m.role === "assistant");

  const dl = (content: string, name: string, mime: string) => {
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(new Blob([content], { type: mime })),
      download: name,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handle = async (type: ExportType) => {
    setLoading(type);
    await new Promise(r => setTimeout(r, 150));
    try {
      const ts = Date.now();
      if (type === "pdf")  await buildPDF(messages, modelLabel);
      if (type === "md")   dl(buildMarkdown(messages, modelLabel), `deceptiscope-${ts}.md`,   "text/markdown");
      if (type === "json") dl(JSON.stringify(buildJSON(messages, modelLabel), null, 2), `deceptiscope-${ts}.json`, "application/json");
      if (type === "csv")  dl(buildCSV(messages), `deceptiscope-${ts}.csv`, "text/csv");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      <div className="px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Export</span>
      </div>

      <div className="p-3 space-y-2">
        <p className="text-2xs text-ink3 leading-relaxed">
          Generate a full deception audit report with per-message scores, type breakdown, and high-risk token highlights.
        </p>

        <div className="grid grid-cols-2 gap-1.5">
          {BTNS.map(b => (
            <motion.button
              key={b.id}
              whileTap={{ scale: 0.97 }}
              onClick={() => handle(b.id)}
              disabled={!hasData || loading !== null}
              className={`py-2 rounded-md text-xs font-semibold transition-all
                disabled:opacity-25 disabled:cursor-not-allowed
                ${b.primary
                  ? "bg-accent text-white hover:bg-blue-500"
                  : "bg-bg border border-border text-ink2 hover:border-border2 hover:text-ink"
                }`}
            >
              {loading === b.id ? (
                <span className="inline-block animate-spin">—</span>
              ) : b.label}
            </motion.button>
          ))}
        </div>

        {!hasData && (
          <p className="text-2xs text-ink3 text-center">No messages to export.</p>
        )}
      </div>
    </div>
  );
};
