import { useMemo, useState } from "react";

const MODES = ["Article text", "Web URL", "Image", "Video"];
const checks = [
  { status: "Corroborated", tone: "emerald", title: "Global ocean temperatures reached a new high in 2024", detail: "Confirmed by NOAA and Copernicus climate records.", source: "NOAA · Copernicus", time: "Today, 10:42" },
  { status: "Synthetic signal", tone: "rose", title: "Viral video claims a solar storm will disable the power grid", detail: "Visual edits detected; the clip matches an older science-fiction trailer.", source: "NOAA Space Weather · AP", time: "Today, 09:18" },
  { status: "Context missing", tone: "amber", title: "Central bank plans to ban cash next month", detail: "The quoted speech concerns institutional settlement, not consumer cash.", source: "Central bank transcript", time: "Yesterday" },
];
const tone = {
  emerald: "border-emerald-800 bg-emerald-950/50 text-emerald-200",
  rose: "border-rose-900 bg-rose-950/40 text-rose-200",
  amber: "border-amber-800 bg-amber-950/40 text-amber-200",
};
const navItems = ["Overview", "Verify", "History", "Sources", "Trends"];

function Brand() {
  return <a href="#main" className="flex items-center gap-3 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-emerald-300">
    <span aria-hidden="true" className="grid h-10 w-10 place-items-center rounded-xl bg-emerald-400 text-slate-950 font-black">V</span>
    <span><span className="block font-bold text-white">Veritas Lens</span><span className="block text-xs text-slate-400">Media verification</span></span>
  </a>;
}

export default function VeritasDashboard() {
  const [mode, setMode] = useState(MODES[0]);
  const [claim, setClaim] = useState("");
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const count = useMemo(() => claim.length, [claim.length]);

  function analyze(event) {
    event.preventDefault();
    setNotice(claim.trim() ? "Analysis queued. Results are model estimates and require human review." : "Add a claim or choose a file before analyzing.");
  }

  return <div className="min-h-screen bg-[#080b0d] text-slate-100 antialiased">
    <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-lg focus:bg-white focus:px-4 focus:py-3 focus:text-slate-950">Skip to main content</a>
    <div className="md:flex">
      <aside className="hidden md:sticky md:top-0 md:flex md:h-screen md:w-64 md:shrink-0 md:flex-col md:border-r md:border-[#253534] md:bg-[#0d1216] md:p-5" aria-label="Main navigation">
        <Brand />
        <nav className="mt-10 space-y-1" aria-label="Workspace">
          {navItems.map((item, i) => <a key={item} href={i === 1 ? "#verify" : i === 2 ? "#recent" : "#main"} aria-current={i === 0 ? "page" : undefined} className={`flex min-h-11 items-center rounded-xl px-3 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300 ${i === 0 ? "bg-emerald-400/10 text-emerald-200" : "text-slate-300 hover:bg-white/5 hover:text-white"}`}>{item}</a>)}
        </nav>
        <div className="mt-auto rounded-xl border border-[#253534] bg-[#090e12] p-4 text-sm">
          <p className="font-semibold text-white">Verification with context</p>
          <p className="mt-2 text-slate-300">Signals support careful review. They are not proof on their own.</p>
        </div>
      </aside>
      <div className="min-w-0 flex-1 pb-20 md:pb-0">
        <header className="sticky top-0 z-20 flex min-h-16 items-center justify-between gap-4 border-b border-[#253534] bg-[#0d1216]/95 px-4 backdrop-blur sm:px-8">
          <div className="flex items-center gap-3">
            <button type="button" className="min-h-11 min-w-11 rounded-lg border border-[#253534] md:hidden" aria-label="Toggle navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>☰</button>
            <div className="md:hidden"><Brand /></div>
            <div className="hidden md:block"><h1 className="text-lg font-bold">Verification console</h1><p className="text-xs text-slate-400">Investigate claims across formats</p></div>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" className="min-h-11 rounded-lg border border-[#253534] px-3 text-sm text-slate-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-300" aria-label="Toggle color theme">◐ <span className="hidden sm:inline">Theme</span></button>
            <a href="#verify" className="inline-flex min-h-11 items-center rounded-lg bg-emerald-400 px-4 text-sm font-bold text-slate-950 hover:bg-emerald-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-200">New check</a>
          </div>
        </header>
        {menuOpen && <nav className="border-b border-[#253534] bg-[#0d1216] p-3 md:hidden" aria-label="Mobile navigation">{navItems.map(item => <a key={item} href={item === "Verify" ? "#verify" : "#main"} onClick={() => setMenuOpen(false)} className="block min-h-11 rounded-lg px-3 py-3 text-sm text-slate-200 hover:bg-white/5">{item}</a>)}</nav>}
        <main id="main" className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-8 lg:px-10">
          <section aria-labelledby="welcome-title" className="flex flex-wrap items-end justify-between gap-4">
            <div><p className="font-mono text-xs uppercase tracking-[0.18em] text-emerald-300">Forensic intelligence</p><h2 id="welcome-title" className="mt-2 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">Good morning. What should we verify?</h2><p className="mt-3 max-w-2xl text-base leading-7 text-slate-300">Check a claim against reliable sources and inspect the evidence behind each signal.</p></div>
            <span className="rounded-full border border-emerald-900 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-200">● Systems operational</span>
          </section>
          <section id="verify" aria-labelledby="verify-title" className="rounded-2xl border border-[#253534] bg-gradient-to-br from-[#101817] to-[#0d1216] p-4 shadow-xl shadow-black/20 sm:p-6">
            <div className="mb-5"><h2 id="verify-title" className="text-xl font-bold text-white">Start a verification</h2><p className="mt-1 text-sm text-slate-300">Choose what you want to check. Your content stays in this session.</p></div>
            <div role="tablist" aria-label="Content type" className="mb-5 flex flex-wrap gap-2">{MODES.map((item, i) => <button key={item} role="tab" id={`mode-${i}`} aria-selected={mode === item} aria-controls="verify-panel" type="button" onClick={() => { setMode(item); setNotice(""); }} className={`min-h-11 rounded-xl border px-4 text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-300 ${mode === item ? "border-emerald-500 bg-emerald-400/10 text-emerald-200" : "border-[#253534] text-slate-300 hover:bg-white/5"}`}>{item}</button>)}</div>
            <form onSubmit={analyze} id="verify-panel" role="tabpanel" aria-labelledby={`mode-${MODES.indexOf(mode)}`}>
              {mode === "Article text" ? <><label htmlFor="claim" className="mb-2 block text-sm font-semibold text-white">Claim or article excerpt</label><textarea id="claim" value={claim} onChange={e => setClaim(e.target.value.slice(0, 3000))} rows={6} maxLength={3000} placeholder="Paste the claim or a short article excerpt…" className="w-full resize-y rounded-xl border border-[#34453f] bg-[#090e12] p-4 text-base leading-7 text-white placeholder:text-slate-400 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-300/40" aria-describedby="claim-help claim-count" /><div className="mt-2 flex flex-wrap justify-between gap-2 text-sm text-slate-300"><span id="claim-help">Include a source or date if you have one.</span><span id="claim-count" aria-live="polite">{count} / 3,000 characters</span></div></> : <div className="rounded-xl border-2 border-dashed border-[#34453f] bg-[#090e12] p-8 text-center"><p className="font-semibold text-white">{mode === "Web URL" ? "Enter a web article URL" : `Choose a ${mode.toLowerCase()} to inspect`}</p>{mode === "Web URL" ? <input type="url" aria-label="Article URL" placeholder="https://example.com/article" className="mt-4 min-h-12 w-full rounded-lg border border-[#34453f] bg-[#080b0d] px-4 text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-300" /> : <label className="mt-4 inline-flex min-h-11 cursor-pointer items-center rounded-lg border border-[#34453f] px-4 text-sm text-slate-200">Select {mode.toLowerCase()}<input type="file" accept={mode === "Image" ? "image/*" : "video/*"} className="sr-only" /></label>}</div>}
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3"><p className="max-w-xl text-sm leading-6 text-slate-300">Results are model estimates for triage and can be wrong. Review source evidence before drawing conclusions.</p><button type="submit" className="min-h-12 rounded-xl bg-emerald-400 px-6 font-bold text-slate-950 hover:bg-emerald-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-200">Analyze claim</button></div>
              {notice && <p role="status" className="mt-4 rounded-lg border border-sky-900 bg-sky-950/40 p-3 text-sm text-sky-100">{notice}</p>}
            </form>
          </section>
          <div className="grid gap-6 xl:grid-cols-5">
            <section id="recent" aria-labelledby="recent-title" className="space-y-4 xl:col-span-3">
              <div className="flex items-center justify-between"><div><h2 id="recent-title" className="text-xl font-bold text-white">Recent checks</h2><p className="mt-1 text-sm text-slate-300">A short history of reviewed claims</p></div><a href="#recent" className="min-h-11 rounded-lg px-3 py-2 text-sm font-semibold text-emerald-200 hover:bg-white/5">View history</a></div>
              <div className="space-y-3">{checks.map((item, i) => <article key={item.title} className="rounded-2xl border border-[#253534] bg-[#0d1216] p-4 sm:p-5"><div className="flex flex-wrap items-center justify-between gap-3"><span className={`rounded-lg border px-3 py-1.5 text-sm font-semibold ${tone[item.tone]}`}>{item.status}</span><time className="text-sm text-slate-400">{item.time}</time></div><h3 className="mt-3 text-base font-bold text-white">{item.title}</h3><p className="mt-1 text-sm leading-6 text-slate-300">{item.detail}</p><div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-[#253534] pt-3 text-sm"><span className="text-slate-300">Sources: {item.source}</span><a href={`#report-${i + 1}`} className="min-h-11 inline-flex items-center font-semibold text-emerald-200 hover:underline">Open evidence <span aria-hidden="true" className="ml-1">→</span></a></div></article>)}</div>
            </section>
            <div className="space-y-6 xl:col-span-2">
              <section aria-labelledby="evidence-title" className="rounded-2xl border border-[#253534] bg-[#0d1216] p-5"><h2 id="evidence-title" className="text-lg font-bold text-white">Evidence snapshot</h2><p className="mt-1 text-sm text-slate-300">Current sample · corroboration overview</p><div className="mt-5 rounded-xl border border-[#253534] bg-[#090e12] p-4"><div className="flex items-center justify-between gap-3"><span className="text-sm font-semibold text-slate-200">Source alignment estimate</span><span className="font-mono text-lg font-bold text-emerald-200">89%</span></div><div className="mt-3 h-2 rounded-full bg-slate-700" role="progressbar" aria-label="Source alignment estimate" aria-valuemin="0" aria-valuemax="100" aria-valuenow="89"><div className="h-2 w-[89%] rounded-full bg-emerald-400" /></div><p className="mt-3 text-sm leading-6 text-slate-300">Model estimate from available sources, not proof of accuracy. Check the citations and context.</p></div><ul className="mt-4 space-y-3 text-sm text-slate-200"><li className="flex justify-between gap-3"><span>Primary sources</span><strong>3 found</strong></li><li className="flex justify-between gap-3"><span>Independent matches</span><strong>2 outlets</strong></li><li className="flex justify-between gap-3"><span>Open questions</span><strong>1 review</strong></li></ul><a href="#sources" className="mt-4 inline-flex min-h-11 items-center font-semibold text-emerald-200 hover:underline">Review source registry →</a></section>
              <section aria-labelledby="trends-title" className="rounded-2xl border border-[#253534] bg-[#0d1216] p-5"><div className="flex items-center justify-between gap-3"><h2 id="trends-title" className="text-lg font-bold text-white">Trend watch</h2><span className="rounded-full border border-amber-800 bg-amber-950/40 px-3 py-1 text-sm text-amber-200">Elevated activity</span></div><p className="mt-3 text-sm leading-6 text-slate-300">Claims about synthetic video are spreading faster than usual in monitored feeds.</p><a href="#trends" className="mt-3 inline-flex min-h-11 items-center font-semibold text-emerald-200 hover:underline">Explore trends →</a></section>
            </div>
          </div>
          <footer className="border-t border-[#253534] pt-5 text-sm leading-6 text-slate-400">Veritas Lens supports investigation with source-grounded signals. A model result is not a final fact-check.</footer>
        </main>
      </div>
    </div>
    <nav aria-label="Mobile navigation" className="fixed inset-x-0 bottom-0 z-30 flex justify-around border-t border-[#253534] bg-[#0d1216] px-2 py-2 md:hidden">{["Overview", "Verify", "History", "Trends"].map((item, i) => <a key={item} href={item === "Verify" ? "#verify" : item === "History" ? "#recent" : "#main"} className={`flex min-h-11 min-w-14 flex-col items-center justify-center rounded-lg px-2 text-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-emerald-300 ${i === 0 ? "text-emerald-200" : "text-slate-300"}`} aria-current={i === 0 ? "page" : undefined}><span aria-hidden="true">{["⌂", "+", "◷", "↗"][i]}</span><span>{item}</span></a>)}</nav>
  </div>;
}
