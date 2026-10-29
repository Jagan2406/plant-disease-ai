import { ArrowRight, Eye, Layers3, Link2, ScanLine, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal.jsx";

const architecture = [
  ["01", "INPUT IMAGE", "Original RGB specimen", "input"],
  ["02", "SAM 2.1", "Leaf segmentation", "vision"],
  ["03", "SWIN TRANSFORMER-T", "Disease classification", "vision"],
  ["04", "GRAD-CAM++", "Post-hoc interpretability", "explain"],
  ["05", "DISEASE KNOWLEDGE BASE", "Exact class grounding", "ground"],
  ["06", "FASTAPI", "Prediction interface", "api"],
  ["07", "REACT CLIENT", "Research workflow", "client"],
];

const methods = [
  { icon: ScanLine, index: "METHOD / 01", title: "Why SAM 2.1?", text: "A segmentation model isolates the leaf foreground before classifier preprocessing. The API returns the real mask, leaf-only image, mask metadata and area ratio." },
  { icon: Layers3, index: "METHOD / 02", title: "Why Swin Transformer?", text: "The fine-tuned Swin classifier provides the disease class and ranked probabilities. The predicted class remains the classifier output throughout the response." },
  { icon: Eye, index: "METHOD / 03", title: "Why Grad-CAM++?", text: "Grad-CAM++ offers a layer-based activation summary to help inspect the classifier's response. Its actual backend-generated visualization is not causal evidence and should not stand alone as diagnosis." },
  { icon: Link2, index: "METHOD / 04", title: "Why knowledge grounding?", text: "An exact class lookup provides the disease intelligence shown after classification. Missing or uncertain records fail closed without changing the model result." },
];

export default function ResearchPage() {
  return (
    <div className="page-container research-page">
      <div className="page-heading-row">
        <div><div className="editorial-kicker"><span /> SYSTEM DESIGN / METHODS</div><h1>Research & methodology</h1><p>A modular computer-vision pipeline with explicit interpretation and grounding boundaries.</p></div>
        <span className="method-tag"><ShieldCheck size={15} /> REVIEWABLE PIPELINE</span>
      </div>
      <section className="architecture-section"><div className="section-title-row"><div><span className="technical-label">SYSTEM ARCHITECTURE</span><h2>Ordered inference path</h2></div><span className="section-index">08 MODULES</span></div>
        <div className="architecture-grid">{architecture.map(([index, title, description, kind], position) => <Reveal key={index} delay={position * 45}>
          <article className={`architecture-block ${kind}`}><span className="architecture-index">{index}</span><strong>{title}</strong><p>{description}</p><span className="architecture-hover">STAGE {index}</span></article>
        </Reveal>)}</div>
        <div className="architecture-disclaimer">The knowledge base is the final disease-intelligence stage. API response and React interface do not alter the classifier result.</div>
      </section>
      <section className="method-section"><div className="section-title-row"><div><span className="technical-label">DESIGN RATIONALE</span><h2>Methods and boundaries</h2></div><span className="section-index">01—04</span></div>
        <div className="method-grid">{methods.map(({ icon: Icon, index, title, text }) => <Reveal key={index}><article className="method-card"><div className="method-card-top"><span>{index}</span><Icon size={17} /></div><h3>{title}</h3><p>{text}</p></article></Reveal>)}</div>
      </section>
      <section className="evaluation-section"><div className="evaluation-mark"><ShieldCheck size={19} /></div><div><span className="technical-label">MODEL EVALUATION</span><h2>Use measured results, not interface estimates.</h2><p>No aggregate benchmark values are embedded in this interface. Evaluation metrics should be added only when tied to a documented evaluation run, dataset split and measurement protocol.</p></div></section>
      <div className="research-endnote"><span>PLANTVISION / METHODS</span><Link to="/detect">Open the detection lab <ArrowRight size={14} /></Link></div>
    </div>
  );
}
