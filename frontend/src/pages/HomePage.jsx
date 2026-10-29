import { ArrowDownRight, ArrowRight, ArrowUpRight, Leaf, Microscope, ScanLine, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import Reveal from "../components/Reveal.jsx";

const stages = [
  ["01", "IMAGE", "Original leaf specimen"],
  ["02", "SAM 2.1", "Leaf segmentation"],
  ["03", "SWIN TRANSFORMER", "Disease classification"],
  ["04", "GRAD-CAM++", "Visual explanation"],
  ["05", "KNOWLEDGE BASE", "Grounded disease context"],
  ["06", "RECOMMENDATION", "Optional language layer"],
];

function BotanicalFigure() {
  return (
    <div className="home-figure">
      <div className="figure-topline"><span>PLATE 01 · BOTANICAL SPECIMEN</span><span>SCHEMATIC VIEW</span></div>
      <div className="figure-grid" />
      <div className="figure-axis axis-x" /><div className="figure-axis axis-y" />
      <svg viewBox="0 0 440 390" className="botanical-svg" role="img" aria-label="Schematic botanical leaf with segmentation boundary">
        <defs>
          <linearGradient id="botanical-leaf" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#dce8c8" /><stop offset=".58" stopColor="#91ad7b" /><stop offset="1" stopColor="#628364" />
          </linearGradient>
          <filter id="botanical-shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="10" stdDeviation="8" floodColor="#304c36" floodOpacity=".15" /></filter>
          <clipPath id="botanical-clip"><path d="M107 286 C72 214 102 114 309 67 C344 176 293 290 174 322 C143 327 119 312 107 286 Z" /></clipPath>
        </defs>
        <g filter="url(#botanical-shadow)">
          <path d="M107 286 C72 214 102 114 309 67 C344 176 293 290 174 322 C143 327 119 312 107 286 Z" fill="url(#botanical-leaf)" />
          <g clipPath="url(#botanical-clip)" className="botanical-veins">
            <path d="M111 298 C185 226 251 143 310 68" className="vein-main" />
            <path d="M149 262 C129 238 113 219 92 205 M173 237 C151 213 136 192 118 177 M198 211 C180 186 166 167 151 149 M223 184 C208 158 198 141 188 123" />
            <path d="M167 247 C195 244 222 236 252 222 M190 221 C219 218 248 207 277 190 M215 193 C242 188 272 176 297 159 M239 166 C266 159 289 148 315 130" />
            <path d="M145 270 C163 285 175 298 183 317 M177 240 C199 254 212 271 223 292 M211 207 C235 220 249 238 262 258" />
          </g>
          <path className="leaf-contour" d="M107 286 C72 214 102 114 309 67 C344 176 293 290 174 322 C143 327 119 312 107 286 Z" />
        </g>
        <g className="measure-marks"><path d="M92 120h15M99.5 112.5v15M315 270h15M322.5 262.5v15" /><circle cx="186" cy="222" r="3" /><circle cx="252" cy="163" r="3" /></g>
        <path className="botanical-scan" d="M69 195 H355" />
      </svg>
      <div className="figure-callout callout-one"><span>01 / BOUNDARY</span><i /></div>
      <div className="figure-callout callout-two"><span>02 / VEIN NETWORK</span><i /></div>
      <div className="figure-labels">
        <span><i /> LEAF SEGMENTATION</span>
        <span><i /> DISEASE CLASSIFICATION</span>
        <span><i /> VISUAL EXPLANATION</span>
      </div>
      <div className="figure-bottomline"><span>ILLUSTRATIVE SCHEMATIC · NOT MODEL OUTPUT</span><span>PV / FIG. 01</span></div>
    </div>
  );
}

function HomePipeline() {
  return (
    <section className="home-pipeline content-width">
      <div className="section-heading-editorial">
        <div><span className="technical-label">A TRACEABLE INFERENCE PATH</span><h2>From leaf to evidence.</h2></div>
        <p>Each stage contributes a distinct signal. The output stays tied to the original classifier prediction.</p>
      </div>
      <div className="pipeline-rail">
        {stages.map(([number, title, description], index) => <Reveal key={number} delay={index * 75}>
          <div className="pipeline-rail-stage">
            <span className="rail-number">{number}</span>
            <div className="rail-node"><i /></div>
            <div className="rail-copy"><strong>{title}</strong><span>{description}</span></div>
            {index < stages.length - 1 && <ArrowRight size={14} className="rail-arrow" />}
          </div>
        </Reveal>)}
      </div>
    </section>
  );
}

export default function HomePage({ apiOnline }) {
  return (
    <>
      <section className="home-hero content-width">
        <div className="home-copy">
          <div className="editorial-kicker"><span /> PLANT DISEASE AI · RESEARCH INTERFACE</div>
          <h1>See the disease.<br /><em>Understand the evidence.</em></h1>
          <p>Leaf segmentation with SAM 2.1. Disease classification with a fine-tuned Swin Transformer. Grad-CAM++ explainability and knowledge-base-grounded disease context.</p>
          <div className="home-cta-row">
            <Link className="button-primary large" to="/detect">Try the detector <ArrowUpRight size={16} /></Link>
            <Link className="text-link" to="/research">Explore methodology <ArrowRight size={15} /></Link>
          </div>
          <div className="home-status-line"><span className={`status-led ${apiOnline ? "on" : ""}`} />{apiOnline ? "INFERENCE API CONNECTED" : "INFERENCE API NOT CONNECTED"}<span className="status-separator" />SINGLE-IMAGE ANALYSIS</div>
        </div>
        <BotanicalFigure />
      </section>
      <HomePipeline />
      <section className="home-principles content-width">
        <Reveal><article className="principle-card"><span className="principle-index">A / 01</span><ScanLine size={18} /><h3>Segment before classifying</h3><p>A dedicated segmentation stage isolates leaf pixels before the classification model is applied.</p></article></Reveal>
        <Reveal delay={90}><article className="principle-card"><span className="principle-index">A / 02</span><Microscope size={18} /><h3>Inspect model activation</h3><p>Grad-CAM++ metrics provide an interpretability signal alongside the model's prediction.</p></article></Reveal>
        <Reveal delay={180}><article className="principle-card"><span className="principle-index">A / 03</span><ShieldCheck size={18} /><h3>Ground disease context</h3><p>Knowledge is exposed only for an exact class lookup and follows the backend's confidence handling.</p></article></Reveal>
      </section>
      <section className="home-bottom content-width">
        <div><span className="technical-label">NO SYNTHETIC METRICS</span><h2>Evidence first.<br />Claims second.</h2></div>
        <p>Performance scores are not displayed here because no verified aggregate evaluation metrics are supplied by the prediction API. Each report uses values returned for the specific image.</p>
        <Link to="/detect" aria-label="Start a new analysis"><ArrowDownRight size={20} /></Link>
      </section>
      <div className="home-footnote"><Leaf size={13} /> Designed for research review · Not a replacement for local expert diagnosis</div>
    </>
  );
}
