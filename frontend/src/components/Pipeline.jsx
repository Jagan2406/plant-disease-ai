import { Check, Circle, LoaderCircle } from "lucide-react";

export const PIPELINE_STAGES = [
  { id: "sam2", number: "01", title: "SAM 2.1", detail: "Leaf segmentation", status: "Detecting leaf boundaries..." },
  { id: "swin", number: "02", title: "Swin Transformer", detail: "Disease classification", status: "Classifying disease..." },
  { id: "gradcam", number: "03", title: "Grad-CAM++", detail: "Visual explanation", status: "Generating visual explanation..." },
  { id: "knowledge", number: "04", title: "Knowledge Base", detail: "Disease context", status: "Retrieving disease knowledge..." },
];

export default function Pipeline({ state, activeIndex }) {
  const isRunning = state === "running";
  const isComplete = state === "complete";

  return (
    <div className={`pipeline-card ${isRunning ? "pipeline-running" : ""}`}>
      <div className="pipeline-card-heading">
        <div>
          <span className="section-kicker">INFERENCE PIPELINE</span>
          <h3>Analysis stages</h3>
        </div>
        <span className={`pipeline-state ${state}`}>
          <i /> {isRunning ? "Processing" : isComplete ? "Complete" : state === "error" ? "Interrupted" : "Ready"}
        </span>
      </div>
      {isRunning && (
        <p className="pipeline-live-status" aria-live="polite">
          {PIPELINE_STAGES[Math.min(activeIndex, PIPELINE_STAGES.length - 1)].status}
        </p>
      )}
      <div className="pipeline-list">
        <div className="pipeline-track"><span className="pipeline-track-fill" /></div>
        {PIPELINE_STAGES.map((stage, index) => {
          const done = isComplete || (isRunning && index < activeIndex);
          const active = isRunning && index === activeIndex;
          const error = state === "error" && index === activeIndex;
          return (
            <div className={`pipeline-stage ${done ? "done" : ""} ${active ? "active" : ""} ${error ? "error" : ""}`} key={stage.id}>
              <div className="pipeline-marker">
                {done ? <Check size={14} /> : active ? <LoaderCircle size={15} className="spin" /> : <Circle size={13} />}
              </div>
              <div className="pipeline-stage-copy">
                <div className="pipeline-stage-title"><span>{stage.number}</span><strong>{stage.title}</strong></div>
                <p>{stage.detail}</p>
              </div>
              {done && <span className="stage-done-label">Done</span>}
              {active && <span className="stage-active-label">Working</span>}
            </div>
          );
        })}
      </div>
      <p className="pipeline-footnote">Stages are shown in request sequence. The API reports results when inference completes.</p>
    </div>
  );
}
