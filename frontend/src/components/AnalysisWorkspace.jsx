import { ArrowUpRight, Check, FileImage, ImagePlus, LoaderCircle, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import { Link } from "react-router-dom";

const STAGE_LABELS = [
  ["01", "SEGMENTING LEAF", "SAM 2.1"],
  ["02", "CLASSIFYING", "Swin Transformer"],
  ["03", "GENERATING EXPLANATION", "Grad-CAM++"],
  ["04", "GROUNDING DISEASE DATA", "Knowledge Base"],
];

export default function AnalysisWorkspace({
  file, previewUrl, imageDimensions, error, pipelineState, activeStage, apiOnline,
  onFile, onRemove, onAnalyze,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const running = pipelineState === "running";
  const done = pipelineState === "complete";

  function dropFile(event) {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files?.[0]) onFile(event.dataTransfer.files[0]);
  }

  return (
    <div className="lab-layout">
      <aside className="lab-upload">
        <div className="lab-section-heading"><span className="technical-label">01 / SPECIMEN INPUT</span><span className="file-constraints">JPG · PNG · 15 MB MAX</span></div>
        <button
          type="button"
          className={`specimen-drop ${dragging ? "drag-over" : ""} ${file ? "has-file" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={dropFile}
        >
          {file && previewUrl ? (
            <>
              <img src={previewUrl} alt={`Selected specimen ${file.name}`} />
              <span className="specimen-drop-overlay"><ImagePlus size={14} /> Replace specimen</span>
            </>
          ) : (
            <span className="drop-prompt">
              <span className="upload-glyph"><UploadCloud size={22} /></span>
              <strong>Drop a leaf image here</strong>
              <small>or browse files</small>
              <span className="drop-file-types"><FileImage size={13} /> JPG, JPEG or PNG</span>
            </span>
          )}
        </button>
        <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" onChange={(event) => { if (event.target.files?.[0]) onFile(event.target.files[0]); event.target.value = ""; }} />
        {file ? (
          <div className="specimen-meta">
            <div className="specimen-filename"><span className="meta-key">FILE</span><strong title={file.name}>{file.name}</strong></div>
            <div className="specimen-meta-row"><span>{imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height}px` : "Reading dimensions…"}</span><span>{(file.size / 1024).toFixed(0)} KB</span></div>
            <div className="specimen-actions">
              <button type="button" className="text-action remove-action" onClick={onRemove} disabled={running}><Trash2 size={14} /> Remove</button>
              <button type="button" className="button-primary" onClick={onAnalyze} disabled={running}>
                {running ? <><LoaderCircle size={15} className="spin" /> Processing</> : <>Analyze specimen <ArrowUpRight size={15} /></>}
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-assurance"><span className="assurance-mark">i</span><p>Image processing runs in memory. Your upload is not stored by this interface.</p></div>
        )}
        {error && <div className="lab-error" role="alert">{error}</div>}
      </aside>

      <section className="lab-specimen-view" aria-label="Specimen image preview">
        <div className="lab-section-heading"><span className="technical-label">02 / IMAGE VIEWPORT</span><span className="viewport-status"><i /> {file ? "SPECIMEN LOADED" : "AWAITING INPUT"}</span></div>
        <div className={`image-viewport ${file ? "has-image" : ""}`}>
          {file && previewUrl ? (
            <>
              <img src={previewUrl} alt={`Preview of ${file.name}`} />
              <span className="viewport-corner top-left">RGB / INPUT</span>
              <span className="viewport-corner top-right">{imageDimensions ? `${imageDimensions.width} × ${imageDimensions.height}` : "IMAGE"}</span>
              <span className="viewport-corner bottom-left">ORIGINAL SPECIMEN</span>
            </>
          ) : (
            <div className="viewport-empty">
              <div className="empty-crosshair"><span /><i /></div>
              <strong>Image viewport</strong>
              <p>Select or drop a leaf image to begin analysis.</p>
              <span>NO SPECIMEN LOADED</span>
            </div>
          )}
          {running && <div className="viewport-scanner" aria-hidden="true" />}
        </div>
        {file && <div className="viewport-caption"><span>{file.name}</span><span>Original RGB image · not yet transformed</span></div>}
      </section>

      <aside className="lab-status">
        <div className="lab-section-heading"><span className="technical-label">03 / ANALYSIS SEQUENCE</span><span className={`sequence-indicator ${running ? "busy" : done ? "complete" : ""}`}><i />{running ? "RUNNING" : done ? "COMPLETE" : "READY"}</span></div>
        <ol className="sequence-list">
          {STAGE_LABELS.map(([number, title, model], index) => {
            const stageDone = done || (running && index < activeStage);
            const active = running && index === activeStage;
            return (
              <li className={`sequence-item ${stageDone ? "stage-complete" : ""} ${active ? "stage-active" : ""}`} key={number}>
                <span className="sequence-marker">{stageDone ? <Check size={13} /> : active ? <LoaderCircle size={13} className="spin" /> : number}</span>
                <div className="sequence-text"><strong>{title}</strong><span>{model}</span></div>
                <span className="sequence-state">{stageDone ? "DONE" : active ? "ACTIVE" : "—"}</span>
              </li>
            );
          })}
        </ol>
        <div className="lab-config">
          <span className="technical-label">RUNTIME</span>
          <div><span>API service</span><b className={apiOnline ? "value-live" : "value-offline"}><i />{apiOnline ? "Connected" : "Offline"}</b></div>
          <div><span>Inference device</span><b>CPU</b></div>
          <div><span>Analysis mode</span><b>Single image</b></div>
        </div>
        <div className="lab-footer-note">{done ? <Link to="/results">View analysis results <ArrowUpRight size={14} /></Link> : null}</div>
      </aside>
    </div>
  );
}
