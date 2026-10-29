import { ArrowDown, ArrowUpRight, ScanLine } from "lucide-react";

const leafVeins = [
  "M149 64 C132 93 115 126 84 165",
  "M137 87 C119 86 104 78 91 67",
  "M123 109 C106 109 91 101 78 91",
  "M109 131 C92 132 77 124 63 114",
  "M97 151 C80 152 65 145 52 135",
  "M119 116 C138 111 153 100 164 86",
  "M106 137 C125 132 140 122 152 109",
  "M92 158 C111 154 126 144 139 132",
];

export function LeafArtwork() {
  return (
    <div className="hero-artwork" aria-label="Illustrative leaf visualization, not live inference output">
      <div className="art-grid" />
      <div className="art-orbit orbit-one" />
      <div className="art-orbit orbit-two" />
      <div className="art-coordinate coordinate-a">VISUAL CONCEPT</div>
      <div className="art-coordinate coordinate-b">SCHEMATIC VIEW</div>
      <div className="art-leaf-wrap">
        <svg className="art-leaf" viewBox="0 0 240 240" role="img" aria-label="Stylized leaf illustration">
          <defs>
            <linearGradient id="leafFill" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#d1e9bd" />
              <stop offset="1" stopColor="#6d9e67" />
            </linearGradient>
            <filter id="leafGlow">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <clipPath id="leafClip">
              <path d="M48 177 C37 113 74 47 178 35 C201 121 153 190 79 197 C65 197 54 190 48 177 Z" />
            </clipPath>
          </defs>
          <path
            className="leaf-shape"
            d="M48 177 C37 113 74 47 178 35 C201 121 153 190 79 197 C65 197 54 190 48 177 Z"
            fill="url(#leafFill)"
          />
          <g clipPath="url(#leafClip)" className="leaf-veins">
            {leafVeins.map((path) => <path key={path} d={path} />)}
            <path d="M49 180 C91 143 129 95 177 37" className="main-vein" />
          </g>
          <path
            className="segmentation-outline"
            d="M48 177 C37 113 74 47 178 35 C201 121 153 190 79 197 C65 197 54 190 48 177 Z"
          />
          <g className="scan-points" filter="url(#leafGlow)">
            <circle cx="76" cy="123" r="3" />
            <circle cx="120" cy="109" r="3" />
            <circle cx="153" cy="72" r="3" />
            <circle cx="91" cy="164" r="2.5" />
          </g>
          <path className="scan-line" d="M38 119 H202" />
        </svg>
      </div>
      <div className="art-readout readout-left">
        <span className="readout-dot" />
        <span>ILLUSTRATION</span>
        <strong>NOT LIVE OUTPUT</strong>
      </div>
      <div className="art-readout readout-right">
        <ScanLine size={15} />
        <span>MODEL FLOW</span>
        <strong>EXPLAINABLE</strong>
      </div>
      <div className="art-crosshair crosshair-one" />
      <div className="art-crosshair crosshair-two" />
    </div>
  );
}

export default function Hero() {
  return (
    <section className="hero-section content-wrap" id="top">
      <div className="hero-copy">
        <div className="eyebrow"><span className="eyebrow-line" /> AI-POWERED CROP HEALTH ANALYSIS</div>
        <h1>Understand plant disease <em>through explainable AI.</em></h1>
        <p className="hero-subtitle">
          Segment the leaf. Identify the disease. Visualize why the model decided.
        </p>
        <div className="hero-actions">
          <a className="button button-primary" href="#analysis">
            Analyze a leaf <ArrowUpRight size={17} />
          </a>
          <a className="button button-quiet" href="#pipeline">
            Explore pipeline <ArrowDown size={15} />
          </a>
        </div>
        <div className="hero-proof">
          <div className="proof-avatars" aria-hidden="true">
            <span><ScanLine size={14} /></span>
            <span><span className="proof-swatch" /></span>
            <span><i className="proof-spark" /></span>
          </div>
          <span>Computer vision, made interpretable</span>
        </div>
      </div>
      <LeafArtwork />
    </section>
  );
}
