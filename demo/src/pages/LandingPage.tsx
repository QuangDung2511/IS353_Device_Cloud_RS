import { useRef } from "react";
import { useLocation } from "wouter";
import { motion, useInView } from "framer-motion";
import {
  Cpu, Server, Shield, Zap, Activity, HardDrive,
  Database, Lock, ChevronRight, ArrowRight, Users,
  Network, BarChart2, Play, Github, BookOpen, CloudLightning
} from "lucide-react";

/* ─── Fade-in wrapper ─────────────────────────────────────── */
function FadeIn({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.55, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ─── Section heading ─────────────────────────────────────── */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 mb-3">
      <div className="h-px w-8 bg-[#00ff88]" style={{ boxShadow: "0 0 6px #00ff88" }} />
      <span className="text-[#00ff88] text-xs tracking-widest" style={{ fontFamily: "var(--font-display)" }}>{children}</span>
      <div className="h-px flex-1 bg-[#2a2a3a]" />
    </div>
  );
}

/* ─── Pipeline step cards ─────────────────────────────────── */
const STEPS = [
  {
    id: "A",
    color: "#00ff88",
    icon: <HardDrive className="w-6 h-6" />,
    title: "Local Extraction",
    location: "On Device",
    desc: "User history is read directly from local storage — never sent to any server. Full privacy from the start.",
    tags: ["SQLite", "TFLite", "Private Enclave"],
  },
  {
    id: "B",
    color: "#00d4ff",
    icon: <Network className="w-6 h-6" />,
    title: "Candidate Generation",
    location: "Device → Cloud → Device",
    desc: "Only anonymised item IDs travel to the cloud. The server runs Graph Search, Vector Similarity, and Popularity ranking to return 50 candidate embeddings.",
    tags: ["KNN Graph", "ANN Search", "Popularity Blend"],
  },
  {
    id: "C",
    color: "#ff00ff",
    icon: <Cpu className="w-6 h-6" />,
    title: "On-Device Inference",
    location: "On Device",
    desc: "A quantised TFLite model scores every candidate locally using dot-product between the user vector and each embedding. Zero data leaves the device.",
    tags: ["TFLite", "Dot Product", "SAGE Encoder"],
  },
  {
    id: "D",
    color: "#00ff88",
    icon: <BarChart2 className="w-6 h-6" />,
    title: "Final Ranking",
    location: "On Device",
    desc: "Candidates are sorted by score and the top-K results are presented to the user. The complete personalisation loop is closed entirely on-device.",
    tags: ["Top-K", "Score Merge", "Zero Leakage"],
  },
];

/* ─── Team members ─────────────────────────────────────────── */
const MEMBERS = [
  { name: "Nguyễn Quang Dũng",      id: "23520335", accent: "#00ff88" },
  { name: "Trần Nguyễn Tiến Đức",   id: "23520322", accent: "#00d4ff" },
  { name: "Lê Bá Vinh",             id: "22521670", accent: "#ff00ff" },
  { name: "Huỳnh Khánh Bảo",        id: "23520101", accent: "#00ff88" },
];

/* ─── Main Component ──────────────────────────────────────── */
export default function LandingPage() {
  const [, navigate] = useLocation();

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-[#e0e0e0] font-mono overflow-x-hidden">

      {/* ── NAVBAR ─────────────────────────────────────────── */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[#2a2a3a] bg-[#0a0a0f]/90 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CloudLightning className="w-5 h-5 text-[#ff00ff]" strokeWidth={3} style={{ filter: "drop-shadow(0 0 4px #ff00ff)" }} />
            <span className="text-[#00ff88] font-bold text-sm" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.12em" }}>
              DCCL
            </span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-xs text-[#8888aa]">
            {["about", "pipeline", "team"].map((s) => (
              <a key={s} href={`#${s}`} className="hover:text-[#00ff88] transition-colors uppercase tracking-widest">
                {s}
              </a>
            ))}
          </div>
          <button
            onClick={() => navigate("/demo")}
            className="flex items-center gap-2 text-xs px-4 py-2 border border-[#00ff88] text-[#00ff88] chamfer bg-[#00ff88]/5 hover:bg-[#00ff88]/15 transition-colors cursor-pointer"
            style={{ boxShadow: "0 0 8px rgba(0,255,136,0.2)" }}
          >
            <Play className="w-3 h-3 fill-current" /> LAUNCH DEMO
          </button>
        </div>
      </nav>

      {/* ── HERO ───────────────────────────────────────────── */}
      <section id="hero" className="relative min-h-screen flex flex-col items-center justify-center pt-16 px-6 text-center overflow-hidden">
        {/* Grid background */}
        <div
          className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: "linear-gradient(#00ff88 1px, transparent 1px), linear-gradient(90deg, #00ff88 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />
        {/* Radial glow */}
        <div className="absolute inset-0 pointer-events-none" style={{ background: "radial-gradient(ellipse 80% 60% at 50% 40%, rgba(0,255,136,0.04) 0%, transparent 70%)" }} />

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="relative z-10 flex flex-col items-center gap-6 max-w-4xl"
        >
          {/* Badge */}
          <div className="flex items-center gap-2 text-[10px] text-[#00d4ff] bg-[#00d4ff]/10 border border-[#00d4ff]/30 px-4 py-1.5 chamfer tracking-widest">
            <Shield className="w-3 h-3" /> PRIVACY-PRESERVING · ON-DEVICE · REAL-TIME
          </div>

          {/* Title */}
          <h1
            className="text-5xl md:text-7xl font-black chromatic-text leading-tight"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "0.06em" }}
          >
            DCCL
          </h1>
          <h2
            className="text-lg md:text-2xl font-bold text-[#e0e0e0]/80 max-w-2xl"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "0.06em" }}
          >
            DEVICE-CLOUD COLLABORATIVE LEARNING
          </h2>
          <p className="text-sm md:text-base text-[#8888aa] max-w-xl leading-relaxed">
            A hybrid recommendation architecture that keeps your personal data locked on-device
            while still leveraging large-scale cloud intelligence for candidate generation.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 mt-2">
            <button
              onClick={() => navigate("/demo")}
              className="group flex items-center gap-3 px-8 py-4 border-2 border-[#00ff88] text-[#00ff88] bg-[#00ff88]/10 hover:bg-[#00ff88]/20 transition-colors chamfer cursor-pointer font-bold"
              style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em", boxShadow: "0 0 20px rgba(0,255,136,0.25)" }}
            >
              <Play className="w-5 h-5 fill-current" />
              RUN PIPELINE DEMO
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <a
              href="#about"
              className="flex items-center gap-2 px-6 py-4 border border-[#2a2a3a] text-[#8888aa] hover:text-[#e0e0e0] hover:border-[#00d4ff] transition-colors chamfer cursor-pointer text-sm"
            >
              <BookOpen className="w-4 h-4" /> LEARN MORE
            </a>
          </div>

          {/* Stats row */}
          <div className="flex flex-wrap justify-center gap-8 mt-8 border-t border-[#2a2a3a] pt-6 w-full max-w-lg">
            {[
              { val: "50", unit: "Vectors", desc: "per inference" },
              { val: "0", unit: "Bytes", desc: "user data uploaded" },
              { val: "4", unit: "Steps", desc: "full pipeline" },
            ].map((s) => (
              <div key={s.desc} className="flex flex-col items-center gap-1">
                <div className="text-3xl font-black text-[#00ff88]" style={{ fontFamily: "var(--font-display)", textShadow: "0 0 12px #00ff88" }}>{s.val}</div>
                <div className="text-xs text-[#e0e0e0]">{s.unit}</div>
                <div className="text-[10px] text-[#8888aa]">{s.desc}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Scroll hint */}
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="absolute bottom-8 text-[#2a2a3a] text-xs flex flex-col items-center gap-1"
        >
          <span>SCROLL</span>
          <div className="w-px h-8 bg-gradient-to-b from-[#2a2a3a] to-transparent" />
        </motion.div>
      </section>

      {/* ── ABOUT ──────────────────────────────────────────── */}
      <section id="about" className="py-24 px-6 border-t border-[#2a2a3a]">
        <div className="max-w-6xl mx-auto">
          <FadeIn>
            <SectionLabel>01 // ABOUT THE PROJECT</SectionLabel>
          </FadeIn>
          <div className="grid md:grid-cols-2 gap-12 mt-8">
            <FadeIn delay={0.1}>
              <h3 className="text-2xl font-black mb-4 text-[#e0e0e0]" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.08em" }}>
                WHY DCCL?
              </h3>
              <p className="text-sm text-[#8888aa] leading-7 mb-4">
                Traditional recommender systems send your full browsing history, purchase data, and interaction logs to remote servers for processing. This creates massive privacy risks and centralises sensitive behavioural data.
              </p>
              <p className="text-sm text-[#8888aa] leading-7">
                DCCL solves this by splitting the workload: the cloud handles only what it can — large-scale retrieval using anonymised IDs — while all personalised ranking happens locally on the user's device using a quantised neural model.
              </p>
            </FadeIn>
            <FadeIn delay={0.2}>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { icon: <Lock className="w-5 h-5" />, title: "Privacy by Design",    color: "#00ff88", desc: "User vectors never leave the device. Ranking is purely local."        },
                  { icon: <Zap  className="w-5 h-5" />, title: "Low Latency",           color: "#00d4ff", desc: "TFLite inference runs in milliseconds on commodity hardware."          },
                  { icon: <Server className="w-5 h-5"/>, title: "Cloud Efficiency",      color: "#ff00ff", desc: "Cloud only receives item IDs, reducing bandwidth and exposure."       },
                  { icon: <Activity className="w-5 h-5"/>, title: "High Accuracy",       color: "#00ff88", desc: "SAGE-encoded embeddings preserve semantic similarity at scale."       },
                ].map((f) => (
                  <div key={f.title} className="bg-[#12121a] border border-[#2a2a3a] p-4 chamfer hover:border-[#2a2a3a]/80 transition-colors">
                    <div className="mb-2" style={{ color: f.color, filter: `drop-shadow(0 0 4px ${f.color})` }}>{f.icon}</div>
                    <div className="text-xs font-bold mb-1 text-[#e0e0e0]" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.06em" }}>{f.title}</div>
                    <div className="text-[10px] text-[#8888aa] leading-4">{f.desc}</div>
                  </div>
                ))}
              </div>
            </FadeIn>
          </div>
        </div>
      </section>

      {/* ── PIPELINE ───────────────────────────────────────── */}
      <section id="pipeline" className="py-24 px-6 border-t border-[#2a2a3a] bg-[#0d0d14]">
        <div className="max-w-6xl mx-auto">
          <FadeIn>
            <SectionLabel>02 // PIPELINE ARCHITECTURE</SectionLabel>
            <h3 className="text-2xl font-black mt-2 mb-2 text-[#e0e0e0]" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.08em" }}>
              THE 4-STEP SEQUENCE
            </h3>
            <p className="text-sm text-[#8888aa] mb-10 max-w-xl">
              Each step in the pipeline is carefully designed to maximise utility while minimising data exposure.
            </p>
          </FadeIn>

          {/* Pipeline steps */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {STEPS.map((step, i) => (
              <FadeIn key={step.id} delay={i * 0.1}>
                <div
                  className="bg-[#12121a] border border-[#2a2a3a] chamfer p-5 h-full flex flex-col hover:border-opacity-100 transition-all group relative overflow-hidden"
                  style={{ "--step-color": step.color } as React.CSSProperties}
                >
                  {/* top accent line */}
                  <div className="absolute top-0 left-0 right-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${step.color}, transparent)`, opacity: 0.6 }} />

                  <div className="flex items-center justify-between mb-4">
                    <div
                      className="w-10 h-10 flex items-center justify-center border chamfer"
                      style={{ borderColor: step.color, color: step.color, boxShadow: `0 0 10px ${step.color}40` }}
                    >
                      {step.icon}
                    </div>
                    <div className="text-3xl font-black opacity-10 group-hover:opacity-20 transition-opacity" style={{ fontFamily: "var(--font-display)", color: step.color }}>
                      {step.id}
                    </div>
                  </div>

                  <div className="text-xs font-bold mb-1" style={{ fontFamily: "var(--font-display)", color: step.color, letterSpacing: "0.08em" }}>
                    STEP {step.id}
                  </div>
                  <div className="text-sm font-bold text-[#e0e0e0] mb-1" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.05em" }}>
                    {step.title}
                  </div>
                  <div className="text-[10px] text-[#8888aa] mb-1 flex items-center gap-1">
                    <ChevronRight className="w-3 h-3" style={{ color: step.color }} /> {step.location}
                  </div>
                  <p className="text-[11px] text-[#8888aa] leading-5 flex-1 mt-2">{step.desc}</p>
                  <div className="flex flex-wrap gap-1 mt-4">
                    {step.tags.map((t) => (
                      <span key={t} className="text-[9px] px-2 py-0.5 border chamfer" style={{ borderColor: `${step.color}40`, color: `${step.color}bb` }}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </FadeIn>
            ))}
          </div>

          {/* Demo CTA */}
          <FadeIn delay={0.4} className="mt-12 text-center">
            <div className="inline-flex flex-col items-center gap-4 bg-[#12121a] border border-[#2a2a3a] chamfer px-10 py-8 relative overflow-hidden">
              <div className="absolute inset-0" style={{ background: "radial-gradient(ellipse 60% 60% at 50% 50%, rgba(0,255,136,0.04) 0%, transparent 70%)" }} />
              <div className="text-xs text-[#8888aa] relative">WANT TO SEE IT IN ACTION?</div>
              <button
                onClick={() => navigate("/demo")}
                className="relative flex items-center gap-3 px-8 py-4 border-2 border-[#00ff88] text-[#00ff88] bg-[#00ff88]/10 hover:bg-[#00ff88]/20 transition-colors chamfer cursor-pointer font-bold"
                style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em", boxShadow: "0 0 20px rgba(0,255,136,0.2)" }}
              >
                <Play className="w-5 h-5 fill-current" /> LAUNCH INTERACTIVE DEMO
              </button>
              <div className="text-[10px] text-[#8888aa] relative">
                Watch all 4 steps animate live with real data flow
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* ── TEAM ───────────────────────────────────────────── */}
      <section id="team" className="py-24 px-6 border-t border-[#2a2a3a]">
        <div className="max-w-6xl mx-auto">
          <FadeIn>
            <SectionLabel>03 // TEAM MEMBERS</SectionLabel>
            <h3 className="text-2xl font-black mt-2 mb-2 text-[#e0e0e0]" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.08em" }}>
              THE CREW
            </h3>
            <p className="text-sm text-[#8888aa] mb-10">University of Information Technology — VNU-HCM</p>
          </FadeIn>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {MEMBERS.map((m, i) => (
              <FadeIn key={m.id} delay={i * 0.1}>
                <div
                  className="bg-[#12121a] border border-[#2a2a3a] chamfer p-5 flex flex-col gap-4 relative overflow-hidden group hover:border-opacity-80 transition-all cursor-default"
                >
                  {/* Corner accent */}
                  <div className="absolute top-0 right-0 w-8 h-8 overflow-hidden">
                    <div className="absolute top-0 right-0 border-t-2 border-r-2 w-full h-full" style={{ borderColor: m.accent }} />
                  </div>

                  {/* Avatar placeholder — initials */}
                  <div
                    className="w-12 h-12 chamfer flex items-center justify-center text-lg font-black border"
                    style={{
                      borderColor: m.accent,
                      color: m.accent,
                      fontFamily: "var(--font-display)",
                      boxShadow: `0 0 12px ${m.accent}30`,
                      backgroundColor: `${m.accent}10`,
                    }}
                  >
                    {m.name.split(" ").slice(-1)[0][0]}
                  </div>

                  <div>
                    <div className="text-sm font-bold text-[#e0e0e0] leading-snug mb-1">{m.name}</div>
                    <div className="text-[10px] text-[#8888aa] bg-black/40 border border-[#2a2a3a] px-2 py-1 chamfer inline-block">
                      ID: {m.id}
                    </div>
                  </div>

                  {/* Hover glow line */}
                  <div
                    className="absolute bottom-0 left-0 right-0 h-px opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ background: `linear-gradient(90deg, transparent, ${m.accent}, transparent)` }}
                  />
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ─────────────────────────────────────────── */}
      <footer className="border-t border-[#2a2a3a] py-8 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm">
            <CloudLightning className="w-4 h-4 text-[#ff00ff]" strokeWidth={3} />
            <span className="text-[#00ff88] font-bold" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em" }}>DCCL</span>
            <span className="text-[#8888aa]">// DEVICE-CLOUD COLLABORATIVE LEARNING</span>
          </div>
          <div className="flex items-center gap-4 text-[10px] text-[#8888aa]">
            <span>UIT — VNU-HCM</span>
            <span className="text-[#2a2a3a]">|</span>
            <span>PRIVACY-PRESERVING RECOMMENDER</span>
          </div>
          <button
            onClick={() => navigate("/demo")}
            className="flex items-center gap-2 text-[#00ff88] text-xs border border-[#00ff88]/40 px-4 py-2 chamfer hover:bg-[#00ff88]/10 transition-colors cursor-pointer"
          >
            <Play className="w-3 h-3 fill-current" /> RUN DEMO
          </button>
        </div>
      </footer>

    </div>
  );
}
