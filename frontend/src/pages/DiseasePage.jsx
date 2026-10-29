import { AlertTriangle, ArrowRight, BookOpen, CheckCircle2, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import {
  asItems, AvailableRecordFields, DataSection, formatDisease, HealthyBanner,
  isHealthy, isViral,
} from "../components/AnalysisViews.jsx";

function EmptyDisease() {
  return <section className="empty-route"><span className="empty-route-icon"><BookOpen size={22} /></span><span className="technical-label">NO KNOWLEDGE RECORD</span><h1>Disease intelligence.</h1><p>Run a prediction to retrieve the exact class-matched record from the knowledge base.</p><Link className="button-primary" to="/detect">Analyze a specimen <ArrowRight size={15} /></Link></section>;
}

export default function DiseasePage({ result }) {
  if (!result) return <EmptyDisease />;
  const record = result.grounded_context?.knowledge || result.knowledge?.record;
  const healthy = isHealthy(result);
  const viral = isViral(record);
  const exact = result.knowledge?.status === "grounded" && Boolean(record);
  const management = [
    ["Cultural management", "CULTURAL", record?.cultural_management],
    ["Biological management", "BIOLOGICAL", record?.biological_management],
    ["Chemical management", "CHEMICAL", record?.chemical_management],
  ];
  return (
    <div className={`page-container disease-page ${healthy ? "healthy-disease-page" : ""}`}>
      <div className="page-heading-row">
        <div><div className="editorial-kicker"><span /> KNOWLEDGE BASE / EXACT CLASS LOOKUP</div><h1>{healthy ? "Healthy-class intelligence" : "Disease intelligence"}</h1><p>Grounded information returned for this image's classifier prediction.</p></div>
        <Link className="quiet-link" to="/results">View prediction <ArrowRight size={14} /></Link>
      </div>
      {healthy && <HealthyBanner />}
      {exact ? (
        <>
          <section className="disease-identity">
            <div><span className="technical-label">PREDICTED CLASS</span><h2>{formatDisease(result.knowledge.class_name || result.predicted_class)}</h2><span className="class-raw">{result.knowledge.class_name || result.predicted_class} · ID {result.knowledge.class_id}</span></div>
            <span className="record-grounded"><CheckCircle2 size={15} /> EXACT MATCH · GROUNDED</span>
          </section>
          <AvailableRecordFields record={record} />
          {!healthy && <div className="disease-info-grid">
            <DataSection title="Symptoms" eyebrow="OBSERVED SIGNS" value={record.symptoms} icon={BookOpen} />
            <DataSection title="Transmission" eyebrow="SPREAD & RESERVOIR" value={record.transmission} icon={ArrowRight} />
            <DataSection title="Favorable conditions" eyebrow="ENVIRONMENT" value={record.favorable_conditions} icon={BookOpen} />
            <DataSection title="Differential diagnosis" eyebrow="ALTERNATIVES TO CONSIDER" value={record.differential_diagnosis} icon={AlertTriangle} />
          </div>}
          {!healthy && !viral && <div className="disease-management-grid">
            {management.map(([title, eyebrow, value]) => <DataSection key={eyebrow} title={title} eyebrow={eyebrow} value={value} icon={ShieldCheck} emphasis={eyebrow === "CHEMICAL"} />)}
          </div>}
          {viral && <div className="viral-guidance"><AlertTriangle size={19} /><div><strong>Viral disease · no curative chemical treatment claim</strong><p>Management details below are presented exactly as returned by the knowledge record. They must not be interpreted as a chemical cure.</p></div></div>}
          {viral && <div className="disease-management-grid">
            {management.filter(([, eyebrow]) => eyebrow !== "CHEMICAL").map(([title, eyebrow, value]) => <DataSection key={eyebrow} title={title} eyebrow={eyebrow} value={value} icon={ShieldCheck} />)}
          </div>}
          {viral && <DataSection title="Chemical management" eyebrow="VIRAL SAFETY" value={record.chemical_management} icon={AlertTriangle} emphasis />}
          {healthy && <div className="healthy-record-note"><CheckCircle2 size={17} /><p>This is a healthy-class label, not proof that the plant is free of all diseases, pests or abiotic stress. No disease-management advice is displayed for this class.</p></div>}
          {Array.isArray(record.source_ids) && record.source_ids.length > 0 && (
            <section className="source-status-block"><div><span className="technical-label">EVIDENCE & PROVENANCE</span><h2>Source status</h2></div><div className="source-chips">{record.source_ids.map((id) => <code key={id}>{id}</code>)}</div><p>{record.evidence_strength ? `Evidence status: ${String(record.evidence_strength).replaceAll("_", " ")}. ` : ""}Source IDs are listed in the record; per-source verification metadata is not included in this API response.</p></section>
          )}
        </>
      ) : (
        <section className="knowledge-fail-closed">
          <BookOpen size={22} /><span className="technical-label">ADVICE WITHHELD</span><h2>{result.knowledge?.status === "uncertain" || result.grounded_context?.uncertain ? "Prediction below grounding threshold" : "No exact grounded record available"}</h2><p>The prediction is preserved, but disease-specific information is not shown without an exact, eligible knowledge-base result.</p>
        </section>
      )}
      {record?.label_check_required && !healthy && <div className="disease-label-warning"><ShieldCheck size={15} /> Confirm diagnosis, current label and local jurisdiction requirements before any chemical use.</div>}
    </div>
  );
}
