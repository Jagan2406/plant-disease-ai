import { ArrowRight, ImageOff, Layers3 } from "lucide-react";
import { Link } from "react-router-dom";
import {
  CompactExplanationMetrics, ExplanationDisclaimer, formatDisease,
  numberText, percent, resolutionText,
} from "../components/AnalysisViews.jsx";

function EmptyExplainability() {
  return <section className="empty-route"><span className="empty-route-icon"><Layers3 size={22} /></span><span className="technical-label">NO ACTIVATION DATA</span><h1>Explainability lab.</h1><p>Run an analysis first. The lab uses the Grad-CAM++ values returned with that prediction.</p><Link className="button-primary" to="/detect">Analyze a specimen <ArrowRight size={15} /></Link></section>;
}

export default function ExplainabilityPage({ result, previewUrl }) {
  if (!result) return <EmptyExplainability />;
  const cam = result.gradcam || {};
  const layer = resolutionText(cam.activation_resolution);
  const camImage = result.analysis_images?.gradcam_pp;
  return (
    <div className="page-container explain-page">
      <div className="page-heading-row">
        <div><div className="editorial-kicker"><span /> INTERPRETABILITY WORKBENCH</div><h1>Where did the model look?</h1><p>Grad-CAM++ metrics for the current Swin Transformer prediction.</p></div>
        <Link className="quiet-link" to="/results">Back to report <ArrowRight size={14} /></Link>
      </div>
      <section className="explain-workspace">
        <div className="explain-visual">
          <div className="visual-topbar"><span className="technical-label">GRAD-CAM++ / TARGET CLASS</span><span>{formatDisease(cam.target_class_name || result.predicted_class)}</span></div>
          {camImage ? <img src={camImage} alt="Actual Grad-CAM++ visualization" /> : (
            <div className="cam-unavailable-visual"><ImageOff size={24} /><strong>Activation image unavailable</strong><p>The backend response did not include the Grad-CAM++ visualization.</p></div>
          )}
          <div className="visual-source-note">Grad-CAM++ output image returned by the API · Activation image is generated at the backend target layer.</div>
        </div>
        <aside className="explain-model-panel">
          <span className="technical-label">MODEL TRACE</span>
          <div className="model-trace-item"><span>MODEL</span><strong>Swin Transformer-T</strong></div>
          <div className="model-trace-item"><span>TARGET LAYER RESOLUTION</span><strong>{layer || "Not returned"}</strong></div>
          <div className="model-trace-item"><span>EXPLANATION</span><strong>Grad-CAM++</strong></div>
          <div className="model-trace-item"><span>TARGET CLASS</span><strong>{cam.target_class_name || result.predicted_class}</strong></div>
          <div className="model-trace-item"><span>TARGET CONFIDENCE</span><strong>{percent(cam.confidence) || "Not returned"}</strong></div>
          <div className="model-trace-item"><span>CAM OUTPUT SHAPE</span><strong>{resolutionText(cam.cam_shape) || "Not returned"}</strong></div>
          <div className="model-trace-item"><span>CAM RANGE</span><strong>{numberText(cam.cam_min)} — {numberText(cam.cam_max)}</strong></div>
        </aside>
      </section>
      <section className="explain-metrics-block"><div className="section-title-row"><div><span className="technical-label">SPATIAL SUMMARY</span><h2>Activation metrics</h2></div><span className="section-index">BACKEND VALUES</span></div><CompactExplanationMetrics result={result} /></section>
      <ExplanationDisclaimer />
      <section className="comparison-note"><span className="technical-label">COMPARISON VIEW</span><p>The backend currently returns one normalized CAM summary, not an activation map or alternate-layer outputs. A visual comparison cannot be shown without those artifacts.</p></section>
    </div>
  );
}
