import { Activity, AlertTriangle, CheckCircle2, CircleHelp, Info, Leaf, ShieldCheck, Sparkles } from "lucide-react";
import { useState } from "react";

const VIEW_TABS = [
  { id: "original", label: "Original" },
  { id: "mask", label: "SAM 2.1 mask" },
  { id: "leaf", label: "Leaf only" },
  { id: "cam", label: "Grad-CAM++" },
];

function formatDisease(name = "") {
  return name.replaceAll("___", " · ").replaceAll("__", " · ").replaceAll("_", " ");
}

function formatPercent(value, digits = 2) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
}

function ConfidenceRing({ confidence }) {
  const value = Math.max(0, Math.min(1, Number(confidence) || 0));
  const radius = 38;
  const circumference = 2 * Math.PI * radius;
  // Bands are descriptive UI labels, not model probability thresholds.
  const level = value >= 0.8 ? "High" : value >= 0.6 ? "Moderate" : "Low";
  return (
    <div className="confidence-ring-wrap">
      <svg className="confidence-ring" viewBox="0 0 100 100" aria-label={`Confidence ${formatPercent(value)} (${level})`}>
        <circle className="ring-track" cx="50" cy="50" r={radius} />
        <circle
          className="ring-value"
          cx="50"
          cy="50"
          r={radius}
          style={{ "--ring-circumference": circumference, "--ring-value": circumference * value }}
        />
      </svg>
      <div className="ring-label"><strong>{formatPercent(value)}</strong><span>{level} confidence</span></div>
    </div>
  );
}

function ImageViews({ imageUrl, result }) {
  const [active, setActive] = useState("original");
  const gradcam = result.gradcam || {};
  const metrics = {
    activation: Array.isArray(gradcam.activation_resolution) ? gradcam.activation_resolution.join(" × ") : "—",
    coverage: formatPercent(gradcam.leaf_coverage),
    inside: gradcam.cam_inside_leaf == null ? "Unavailable" : formatPercent(gradcam.cam_inside_leaf),
    max: Number.isFinite(Number(gradcam.cam_max)) ? Number(gradcam.cam_max).toFixed(3) : "—",
  };
  return (
    <section className="result-panel explanation-panel" id="explanation">
      <div className="panel-heading">
        <div><span className="section-kicker">VISUAL EXPLANATION</span><h3>How the AI sees the leaf</h3></div>
        <span className="panel-index">02 / EXPLAINABILITY</span>
      </div>
      <div className="view-tabs" role="tablist" aria-label="Image visualization">
        {VIEW_TABS.map((tab) => (
          <button key={tab.id} type="button" role="tab" aria-selected={active === tab.id} className={active === tab.id ? "selected" : ""} onClick={() => setActive(tab.id)}>{tab.label}</button>
        ))}
      </div>
      <div className="visualization-stage">
        {(() => {
          const analysisImages = result.analysis_images || {};
          const sources = {
            original: analysisImages.original || imageUrl,
            mask: analysisImages.sam2_mask,
            leaf: analysisImages.leaf_only,
            cam: analysisImages.gradcam_pp,
          };
          const source = sources[active];
          return source ? (
            <img className="visualization-image" src={source} alt={`${VIEW_TABS.find((tab) => tab.id === active)?.label || "Analysis"} visualization`} />
          ) : (
            <div className="visualization-placeholder"><strong>Image not returned by the API</strong><p>The analysis completed, but this visualization is unavailable in the response.</p></div>
          );
        })()}
      </div>
      <div className="visualization-caption">
        <span><Info size={15} /> Original upload and returned numerical metrics are shown; derived image artifacts are not available.</span>
      </div>
      <div className="explanation-bottom">
        <div className="activation-note">
          <span className="note-icon"><Sparkles size={16} /></span>
          <div><strong>Why did the model predict this?</strong><p>Highlighted regions represent areas receiving stronger activation for the predicted class when a CAM image is available.</p></div>
          <span className="tooltip-wrap" tabIndex={0} aria-label="Grad-CAM++ is not proof of causality">
            <CircleHelp size={17} />
            <span className="tooltip-content">Grad-CAM++ provides a visual explanation of model activations. It should not be interpreted as proof of causal reasoning.</span>
          </span>
        </div>
        <div className="explanation-metrics">
          <Metric label="Target class" value={formatDisease(gradcam.target_class_name || result.predicted_class)} />
          <Metric label="Confidence" value={formatPercent(gradcam.confidence)} />
          <Metric label="Activation" value={metrics.activation} />
          <Metric label="Leaf coverage" value={metrics.coverage} />
          <Metric label="CAM inside leaf" value={metrics.inside} />
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return <div className="explanation-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function SegmentationMetrics({ result }) {
  const metadata = result.segmentation_metadata || {};
  return (
    <section className="result-panel segmentation-metrics">
      <div className="panel-heading"><div><span className="section-kicker">SEGMENTATION</span><h3>Leaf segmentation</h3></div><span className="metric-leaf-icon"><Leaf size={18} /></span></div>
      <div className="segmentation-stat-grid">
        <Metric label="Mask area" value={formatPercent(result.mask_area_ratio)} />
        <Metric label="Fallback used" value={result.fallback_used ? "Yes" : "No"} />
        <Metric label="Candidates" value={metadata.candidate_count ?? "—"} />
        <Metric label="Selected score" value={metadata.selected_candidate_score == null ? "—" : formatPercent(metadata.selected_candidate_score)} />
        <Metric label="Mask resolution" value={metadata.mask_shape?.join(" × ") || "—"} />
      </div>
    </section>
  );
}

function PredictionBars({ predictions, predictedId }) {
  return (
    <section className="result-panel prediction-panel">
      <div className="panel-heading"><div><span className="section-kicker">CLASS SCORES</span><h3>Model predictions</h3></div><span className="panel-index">{String(predictions?.length || 0).padStart(2, "0")} CLASSES</span></div>
      <div className="prediction-list">
        {(predictions || []).map((item, index) => {
          const score = Number(item.confidence) || 0;
          const winner = item.class_id === predictedId;
          return (
            <div className={`prediction-row ${winner ? "winner" : ""}`} key={`${item.class_id}-${item.class_name}`}>
              <span className="prediction-rank">{String(index + 1).padStart(2, "0")}</span>
              <div className="prediction-main">
                <div className="prediction-label"><strong>{formatDisease(item.class_name)}</strong><span>{(score * 100).toFixed(2)}%</span></div>
                <div className="probability-track"><span style={{ "--bar-value": `${Math.max(0, Math.min(score, 1)) * 100}%` }} /></div>
              </div>
              {winner && <span className="winner-mark"><CheckCircle2 size={15} /></span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function KnowledgePanel({ knowledge, groundedContext }) {
  const record = knowledge?.record;
  const status = knowledge?.status || "unavailable";
  const contextStatus = groundedContext?.status || "unavailable";
  return (
    <section className="result-panel knowledge-panel">
      <div className="panel-heading"><div><span className="section-kicker">GROUNDED INFORMATION</span><h3>Disease intelligence</h3></div><span className={`grounded-pill ${status}`}><ShieldCheck size={14} /> {status}</span></div>
      {record ? (
        <>
          <div className="knowledge-intro"><div className="knowledge-title"><span>Predicted class</span><h4>{formatDisease(knowledge.class_name)}</h4></div><span className="scientific-name">{record.scientific_name || "Scientific name unavailable"}</span></div>
          <div className="knowledge-facts">
            <div><span>Causal agent type</span><strong>{String(record.causal_agent_type || "Unspecified").replaceAll("_", " ")}</strong></div>
            <div><span>Evidence strength</span><strong>{String(record.evidence_strength || "Unspecified").replaceAll("_", " ")}</strong></div>
          </div>
          {[
            ["Symptoms", record.symptoms],
            ["Transmission", record.transmission],
            ["Favorable conditions", record.favorable_conditions],
            ["Cultural management", record.cultural_management],
            ["Biological management", record.biological_management],
            ["Chemical management", record.chemical_management],
            ["Differential diagnosis", record.differential_diagnosis],
          ].map(([title, values]) => Array.isArray(values) && values.length > 0 && (
            <div className="knowledge-block" key={title}><h5>{title}</h5><ul>{values.map((value) => <li key={value}>{value}</li>)}</ul></div>
          ))}
          {record.jurisdiction && <div className="knowledge-jurisdiction"><span>Jurisdiction</span><p>{record.jurisdiction}</p></div>}
          <div className="knowledge-safety"><ShieldCheck size={16} /><span>{record.safety_notes?.join(" ") || "Information is limited to the matched knowledge-base record. Verify diagnosis and local requirements."}</span></div>
          {record.label_check_required && <div className="label-check"><AlertTriangle size={15} />Always check the current local label and jurisdiction requirements.</div>}
          {record.source_ids?.length > 0 && (
            <div className="knowledge-sources">
              <span>Knowledge references</span>
              <div>{record.source_ids.map((sourceId) => <code key={sourceId}>{sourceId}</code>)}</div>
              <small>Reference verification is required before relying on these claims.</small>
            </div>
          )}
        </>
      ) : (
        <div className="knowledge-empty">
          {status === "uncertain" || contextStatus === "uncertain" ? <AlertTriangle size={20} /> : <CircleHelp size={20} />}
          <strong>{status === "uncertain" || contextStatus === "uncertain" ? "Prediction is uncertain" : "Grounded information unavailable"}</strong>
          <p>{groundedContext?.message || "No disease information is shown unless an exact, confident knowledge-base match is available."}</p>
        </div>
      )}
      {groundedContext?.uncertain && <p className="uncertainty-banner"><AlertTriangle size={15} />Low confidence — verify the identification before acting.</p>}
    </section>
  );
}

export default function Results({ result, imageUrl }) {
  const confidence = Number(result.confidence) || 0;
  return (
    <div className="results-section" id="results">
      <section className="result-hero">
        <div className="result-image-wrap"><img src={imageUrl} alt="Analyzed leaf" /><span className="image-overlay-tag"><CheckCircle2 size={14} /> Analysis complete</span></div>
        <div className="result-diagnosis">
          <div className="result-label"><span className="status-spark" /> DETECTED CLASS</div>
          <h2>{formatDisease(result.predicted_class)}</h2>
          <span className="result-raw-class">{result.predicted_class}</span>
          <div className="diagnosis-divider" />
          <div className="confidence-block"><ConfidenceRing confidence={confidence} /><div className="confidence-copy"><span>MODEL CONFIDENCE</span><strong>{(confidence * 100).toFixed(2)}<small>%</small></strong><p>Probability assigned to the top class</p></div></div>
          {result.grounded_context?.uncertain && <div className="uncertainty-banner"><AlertTriangle size={15} />Low confidence — review this result with an expert.</div>}
        </div>
        <div className="result-corner-label">PV / 01</div>
      </section>
      <div className="results-grid">
        <ImageViews imageUrl={imageUrl} result={result} />
        <div className="results-column">
          <SegmentationMetrics result={result} />
          <PredictionBars predictions={result.top_k_predictions} predictedId={result.class_id} />
          <KnowledgePanel knowledge={result.knowledge} groundedContext={result.grounded_context} />
          
        </div>
      </div>
    </div>
  );
}
