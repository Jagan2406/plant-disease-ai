import { Activity, ArrowUpRight, Leaf, Menu, X } from "lucide-react";
import { useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";

const navigation = [
  { to: "/detect", label: "Detection lab" },
  { to: "/results", label: "Results" },
  { to: "/explainability", label: "Explainability" },
  { to: "/disease", label: "Disease intelligence" },
  { to: "/research", label: "Research" },
];

export default function SiteShell({ apiOnline, children }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  return (
    <>
      <header className="topbar">
        <Link className="wordmark" to="/" aria-label="PlantVision home">
          <span className="wordmark-emblem"><Leaf size={18} /></span>
          <span>PLANTVISION<span className="wordmark-ai"> / AI</span></span>
        </Link>
        <button className="nav-toggle" type="button" aria-label={menuOpen ? "Close menu" : "Open menu"} onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <X size={19} /> : <Menu size={19} />}
        </button>
        <nav className={`topnav ${menuOpen ? "open" : ""}`} aria-label="Main navigation">
          <NavLink to="/" end onClick={() => setMenuOpen(false)}>Overview</NavLink>
          {navigation.map((item) => (
            <NavLink key={item.to} to={item.to} onClick={() => setMenuOpen(false)} className={({ isActive }) => `${isActive ? "active" : ""} ${item.to === "/results" && location.pathname === "/results" ? "active" : ""}`}>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className={`api-indicator ${apiOnline ? "online" : ""}`}>
          <span className="api-led" /><Activity size={14} />
          <span>{apiOnline ? "API connected" : "API offline"}</span>
        </div>
      </header>
      <main className="route-stage" key={location.pathname}>{children}</main>
      <footer className="footer">
        <Link className="footer-mark" to="/"><Leaf size={16} /> PlantVision AI</Link>
        <span>Plant disease analysis · Research interface</span>
        <Link to="/research">Methods <ArrowUpRight size={13} /></Link>
      </footer>
    </>
  );
}
