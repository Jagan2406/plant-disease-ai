import { ArrowLeft, ScanLine } from "lucide-react";
import { Link } from "react-router-dom";
import AnalysisWorkspace from "../components/AnalysisWorkspace.jsx";

export default function DetectionPage(props) {
  return (
    <section className="page-container detection-page">
      <div className="page-heading-row">
        <div><div className="editorial-kicker"><span /> SPECIMEN WORKSTATION</div><h1>Detection lab</h1><p>Load one leaf image to begin a sequential segmentation, classification and explanation pass.</p></div>
        <div className="page-heading-side"><span><ScanLine size={15} /> EXPERIMENT / SINGLE IMAGE</span>{props.result && <Link to="/results"><ArrowLeft size={14} /> Previous result</Link>}</div>
      </div>
      <AnalysisWorkspace {...props} />
      <div className="lab-bottom-note"><span className="technical-label">METHOD NOTE</span><p>Progress stages are frontend request-state indicators. Backend stage timings are returned after inference for diagnostics.</p></div>
    </section>
  );
}
