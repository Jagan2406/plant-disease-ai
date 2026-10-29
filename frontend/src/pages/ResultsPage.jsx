import { ArrowRight, Leaf, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import {
  ConfidenceMeter, PredictionDetails, ResultImageStrip, TopPredictions, formatDisease, isHealthy,
} from "../components/AnalysisViews.jsx";

function EmptyResults() {
  return <section className="empty-route"><span className="empty-route-icon"><Leaf size={22} /></span><span className="technical-label">NO ANALYSIS IN SESSION</span><h1>Results appear here.</h1><p>Run a leaf analysis to inspect the returned prediction, confidence and evidence.</p><Link className="button-primary" to="/detect">Open detection lab <ArrowRight size={15} /></Link></section>;
}

export default function ResultsPage({ result, previewUrl }) {
  if (!result) return <EmptyResults />;
  const healthy = isHealthy(result);
  const knowledgeStatus = result.grounded_context?.status || result.knowledge?.status || "unavailable";
  return (
    <div className="page-container results-page">
      <div className="result-breadcrumb"><span className="technical-label">ANALYSIS REPORT</span><span>/</span><span>SESSION RESULT</span><Link to="/detect">NEW SPECIMEN <ArrowRight size={12} /></Link></div>
      <section className={`result-title-block ${healthy ? "healthy-result" : ""}`}>
        <div className="result-title-copy">
          <div className="editorial-kicker"><span /> {healthy ? "HEALTHY-CLASS PREDICTION" : "CLASSIFICATION RESULT"}</div>
          <h1>{healthy ? "Healthy leaf predicted" : formatDisease(result.predicted_class)}</h1>
          <div className="result-identity"><span>{result.predicted_class}</span>{Number.isInteger(result.class_id) && <span>CLASS ID / {result.class_id}</span>}</div>
          {healthy && <p className="healthy-caveat"><ShieldCheck size={15} /> A healthy-class prediction does not confirm absence of all disease or stress.</p>}
        </div>
        <ConfidenceMeter result={result} />
      </section>
      <div className="result-context-bar"><span><i /> ANALYSIS COMPLETE</span><span>KNOWLEDGE · {knowledgeStatus.replaceAll("_", " ").toUpperCase()}</span><span>CONFIDENCE · {Number(result.confidence * 100).toFixed(2)}%</span></div>
      <ResultImageStrip result={result} previewUrl={previewUrl} />
      <div className="result-content-grid">
        <div className="result-main-column">
          <PredictionDetails result={result} />
          <TopPredictions result={result} />
        </div>
        <aside className="result-side-column">
          <div className="result-next-step"><span className="technical-label">FURTHER INSPECTION</span><h3>Read the explanation.</h3><p>Review activation metrics and interpretation limits for this classification.</p><Link to="/explainability">Open explainability lab <ArrowRight size={14} /></Link></div>
          <div className="result-next-step secondary"><span className="technical-label">GROUNDED CONTEXT</span><h3>Disease intelligence</h3><p>Inspect the exact knowledge record returned for the predicted class.</p><Link to="/disease">Open disease record <ArrowRight size={14} /></Link></div>
          <div className="result-note"><ShieldCheck size={15} /><p>Model outputs support review; they are not a definitive diagnosis.</p></div>
        </aside>
      </div>
    </div>
  );
}
