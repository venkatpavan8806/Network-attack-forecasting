import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import type { ForecastResponse, AttackMapping } from './types';
import { STAGE_LABELS } from './api';

const WINDOW_SECONDS = 30; // must match backend app.config.WINDOW_SECONDS

export function downloadForecastPdf(forecast: ForecastResponse, attackMapping: AttackMapping[] = []) {
  const mappingByAction = new Map(attackMapping.map((m) => [m.state_label, m]));
  const doc = new jsPDF();

  doc.setFontSize(16);
  doc.setTextColor(31, 34, 51);
  doc.text('CARIBBEAN — Attack Forecast Report', 14, 18);

  doc.setFontSize(9);
  doc.setTextColor(139, 138, 158);
  doc.text(`Generated ${new Date().toLocaleString()}`, 14, 25);

  doc.setFontSize(10);
  doc.setTextColor(31, 34, 51);
  const introLines = [
    `Host: ${forecast.host_id}`,
    `Current window: #${forecast.window_idx}`,
    `Current predicted action: ${STAGE_LABELS[forecast.predicted_stage] ?? forecast.predicted_stage} (${(forecast.infiltration_probability_world_model * 100).toFixed(1)}% infiltration probability, world model)`,
    `Baseline (logistic regression) probability at this window: ${(forecast.infiltration_probability_baseline * 100).toFixed(1)}%`,
  ];
  if (forecast.true_stage) {
    introLines.push(`Ground truth at this window (demo/CSV-replay data only): ${forecast.true_stage}`);
  }
  let y = 34;
  for (const line of introLines) {
    doc.text(line, 14, y);
    y += 6;
  }

  const mapping = forecast.attack_mapping;
  if (mapping.likely_tools || mapping.likely_system_state) {
    y += 2;
    if (mapping.likely_tools) {
      doc.setFont('helvetica', 'bold');
      doc.text('Likely tools:', 14, y);
      doc.setFont('helvetica', 'normal');
      const wrapped = doc.splitTextToSize(mapping.likely_tools, 180);
      doc.text(wrapped, 40, y);
      y += 6 * wrapped.length;
    }
    if (mapping.likely_system_state) {
      doc.setFont('helvetica', 'bold');
      doc.text('If successful:', 14, y);
      doc.setFont('helvetica', 'normal');
      const wrapped = doc.splitTextToSize(mapping.likely_system_state, 180);
      doc.text(wrapped, 40, y);
      y += 6 * wrapped.length;
    }
    y += 2;
  }

  const rows = forecast.rollout.predicted_stage_per_horizon.map((action, i) => {
    const horizon = i + 1;
    const probability = forecast.rollout.infiltration_probs_world_model[i];
    const futureWindow = forecast.window_idx + horizon;
    const leadTimeMin = (horizon * WINDOW_SECONDS) / 60;
    const mapping = mappingByAction.get(action);
    const attackStage = mapping?.technique_id
      ? `${mapping.tactic} (${mapping.technique_id} — ${mapping.technique_name})`
      : mapping?.tactic ?? '—';
    return [
      `${(probability * 100).toFixed(1)}%`,
      `t+${horizon} (window #${futureWindow})`,
      STAGE_LABELS[action] ?? action,
      attackStage,
      `${leadTimeMin.toFixed(1)} min`,
    ];
  });

  autoTable(doc, {
    startY: y + 4,
    head: [['Probability', 'Future Window', 'Next Action', 'Attack Stage', 'Lead Time']],
    body: rows,
    headStyles: { fillColor: [108, 93, 211], textColor: 255, fontSize: 9 },
    bodyStyles: { fontSize: 8.5, textColor: [31, 34, 51] },
    alternateRowStyles: { fillColor: [242, 240, 251] },
    columnStyles: { 0: { cellWidth: 22 }, 1: { cellWidth: 34 }, 4: { cellWidth: 22 } },
  });

  const finalY = (doc as any).lastAutoTable?.finalY ?? y + 4;
  doc.setFontSize(8);
  doc.setTextColor(139, 138, 158);
  const note = [
    'Lead time = time from now until that future window occurs (horizon x 30s window length),',
    'i.e. how much advance warning this row represents -- not the separate benchmark-level',
    '"lead time vs. baseline" metric reported elsewhere in the app. All probabilities and',
    'predicted actions above are real output from the trained LSTM world model, computed',
    'at report-generation time -- nothing on this page is a placeholder.',
  ];
  let noteY = finalY + 8;
  for (const line of note) {
    doc.text(line, 14, noteY);
    noteY += 4;
  }

  doc.save(`caribbean-forecast-${forecast.host_id.replace(/[:.]/g, '_')}-w${forecast.window_idx}.pdf`);
}
