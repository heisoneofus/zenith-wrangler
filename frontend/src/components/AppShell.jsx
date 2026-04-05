import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Run" },
  { to: "/sessions", label: "Sessions" },
  { to: "/update", label: "Update" },
];

export function AppShell({ children }) {
  return (
    <div className="shell">
      <header className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Zenith Wrangler</p>
          <h1>Dataset-to-dashboard workflow, now with a web surface.</h1>
          <p className="hero__lede">
            Upload a dataset, inspect the inferred plan, render Plotly charts from backend JSON, and revise prior
            sessions without dropping the existing Python intelligence layer.
          </p>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav__link${isActive ? " nav__link--active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="page-frame">{children}</main>
    </div>
  );
}
