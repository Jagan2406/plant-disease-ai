import { FileImage, ImagePlus, LoaderCircle, RotateCcw, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";

const MAX_FILE_BYTES = 15 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

function formatBytes(bytes) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(0)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadPanel({
  file,
  previewUrl,
  imageDimensions,
  busy,
  error,
  onFile,
  onAnalyze,
  onRemove,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function acceptFile(candidate) {
    if (candidate) onFile(candidate);
  }

  return (
    <div className="upload-card">
      <div className="upload-card-heading">
        <span className="section-kicker">IMAGE INPUT</span>
        <span className="upload-format">JPG · JPEG · PNG · MAX 15 MB</span>
      </div>
      {!file ? (
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
          }}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            acceptFile(event.dataTransfer.files?.[0]);
          }}
        >
          <div className="drop-icon"><UploadCloud size={25} strokeWidth={1.65} /></div>
          <strong>Drop a leaf image here</strong>
          <span>or <button type="button" className="browse-button" onClick={(event) => { event.stopPropagation(); inputRef.current?.click(); }}>browse from your device</button></span>
          <div className="drop-supported"><FileImage size={14} /> Supported formats: JPG, JPEG, PNG</div>
        </div>
      ) : (
        <div className="selected-file">
          <div className="preview-frame">
            <img src={previewUrl} alt={`Preview of ${file.name}`} />
            <span className="preview-badge"><ImagePlus size={13} /> Input image</span>
          </div>
          <div className="selected-file-meta">
            <div className="file-name-row">
              <div><strong title={file.name}>{file.name}</strong><span>{formatBytes(file.size)}{imageDimensions ? ` · ${imageDimensions.width} × ${imageDimensions.height}px` : ""}</span></div>
              <button type="button" className="icon-button" onClick={onRemove} aria-label="Remove image" disabled={busy}><X size={17} /></button>
            </div>
            <button className="button button-primary analyze-button" type="button" onClick={onAnalyze} disabled={busy}>
              {busy ? <><LoaderCircle size={16} className="spin" /> Analyzing leaf...</> : <>Start AI analysis <RotateCcw size={15} /></>}
            </button>
          </div>
        </div>
      )}
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        onChange={(event) => { acceptFile(event.target.files?.[0]); event.target.value = ""; }}
      />
      {error && <div className="form-error" role="alert">{error}</div>}
      <p className="privacy-note">Your image is analyzed in memory and is not saved by this interface.</p>
    </div>
  );
}

export { ALLOWED_TYPES, MAX_FILE_BYTES };
