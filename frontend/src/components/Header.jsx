import { Activity, Leaf, Menu, X } from "lucide-react";
import { useState } from "react";

const links = [
  ["Detection", "#analysis"],
  ["How it works", "#pipeline"],
  ["Explainability", "#explanation"],
  ["About", "#about"],
];

export default function Header({ apiOnline }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="site-header">
      <a className="brand" href="#top" aria-label="PlantVision AI home">
        <span className="brand-mark">
          <Leaf size={19} strokeWidth={2.2} />
        </span>
        <span>PlantVision<span className="brand-light"> AI</span></span>
      </a>
      <button
        className="mobile-menu-toggle"
        type="button"
        aria-label={menuOpen ? "Close navigation" : "Open navigation"}
        onClick={() => setMenuOpen((open) => !open)}
      >
        {menuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <nav className={`main-nav ${menuOpen ? "is-open" : ""}`}>
        {links.map(([label, href]) => (
          <a key={label} href={href} onClick={() => setMenuOpen(false)}>
            {label}
          </a>
        ))}
      </nav>
      <div className={`online-status ${apiOnline ? "is-online" : "is-offline"}`}>
        <Activity size={15} />
        <span>{apiOnline ? "AI system online" : "API offline"}</span>
        <i aria-hidden="true" />
      </div>
    </header>
  );
}
