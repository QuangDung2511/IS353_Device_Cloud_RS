import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu, Server, Shield, Zap, Activity, HardDrive, Database,
  Lock, CheckCircle2, ChevronRight, Share2, Play, Network, BarChart2, ArrowLeft, CloudLightning
} from "lucide-react";
import { Link } from "wouter";
import { fetchRecommendations, type RecommendationResponse } from "@/lib/api";

/* ─── Types ───────────────────────────────────────────────────── */
// Phases 0-15 drive every visual in the demo
type Phase =
  | 0   // idle
  | 1   // typing
  | 2   // items appearing
  | 3   // all items + SECURE badge; packet departing
  | 4   // packet traveling to cloud (bridge anim)
  | 5   // cloud node 1 (GRAPH SEARCH) active
  | 6   // cloud node 2 (VECTOR SIM) active
  | 7   // cloud node 3 (POPULARITY RANK) active
  | 8   // all cloud nodes done; embeddings packet departing
  | 9   // embeddings packet traveling back (bridge anim)
  | 10  // device receiving embeddings
  | 11  // TFLite card appears; inference starting
  | 12  // vector computation animating
  | 13  // inference complete
  | 14  // ranking cards appearing (staggered)
  | 15; // complete

const HISTORY_PREVIEW_LIMIT = 6;

function vectorBars(values: number[]) {
  if (!values.length) return Array.from({ length: 8 }, () => 0.08);
  const maxAbs = Math.max(...values.map((value) => Math.abs(value)), 1e-6);
  return values.map((value) => Math.max(0.08, Math.min(1, Math.abs(value) / maxAbs)));
}

/* ─── Phase schedule (ms to next phase from each phase) ───────── */
const SCHEDULE: Partial<Record<Phase, number>> = {
  3: 600,   // brief pause after secure badge, then packet departs
  4: 1400,  // packet travels to cloud
  5: 1600,  // graph search
  6: 1400,  // vector sim
  7: 1200,  // popularity rank
  8: 600,   // brief pause, embeddings depart
  9: 1400,  // packet travels back
  10: 900,  // receiving animation
  11: 900,  // tflite card
  12: 2200, // vector computation
  13: 600,  // brief pause
  14: 1800, // rankings stagger
};

/* ─── Step label derived from phase ─────────────────────────────── */
function stepOf(phase: Phase): "A" | "B" | "C" | "D" | null {
  if (phase >= 1 && phase <= 3) return "A";
  if (phase >= 4 && phase <= 9) return "B";
  if (phase >= 10 && phase <= 13) return "C";
  if (phase >= 14) return "D";
  return null;
}

/* ─── CloudNode ─────────────────────────────────────────────────── */
interface CloudNodeProps {
  label: string;
  icon: React.ReactNode;
  state: "idle" | "active" | "done";
  desc: string;
}

function CloudNode({ label, icon, state, desc }: CloudNodeProps) {
  return (
    <motion.div
      animate={
        state === "active"
          ? { borderColor: "#ff00ff", backgroundColor: "rgba(255,0,255,0.08)", boxShadow: "0 0 18px rgba(255,0,255,0.35)" }
          : state === "done"
          ? { borderColor: "rgba(255,0,255,0.4)", backgroundColor: "rgba(255,0,255,0.04)", boxShadow: "none" }
          : { borderColor: "#2a2a3a", backgroundColor: "#1c1c2e", boxShadow: "none" }
      }
      transition={{ duration: 0.3 }}
      className="p-3 border chamfer"
    >
      <div className="flex items-center justify-between mb-2">
        <span
          className={`font-bold text-sm transition-colors duration-300 ${
            state === "idle" ? "text-[#8888aa]" : "text-[#ff00ff]"
          }`}
        >
          {label}
        </span>
        <div className={`transition-colors duration-300 ${state === "idle" ? "text-[#2a2a3a]" : "text-[#ff00ff]"}`}>
          {state === "done" ? <CheckCircle2 className="w-4 h-4" /> : icon}
        </div>
      </div>

      <div className="text-[10px] text-[#8888aa] mb-2">{desc}</div>

      {/* Processing bar */}
      <div className="h-1 bg-black/60 overflow-hidden">
        {state === "active" && (
          <motion.div
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: (SCHEDULE[5]! / 1000) * 0.8, ease: "linear" }}
            className="h-full bg-[#ff00ff] shadow-[0_0_6px_#ff00ff]"
          />
        )}
        {state === "done" && (
          <div className="h-full w-full bg-[#ff00ff]/40" />
        )}
      </div>

      {state !== "idle" && (
        <div className={`text-[9px] mt-1 ${state === "active" ? "text-[#ff00ff] animate-pulse" : "text-[#ff00ff]/60"}`}>
          {state === "active" ? "PROCESSING..." : "OK"}
        </div>
      )}
    </motion.div>
  );
}

/* ─── BridgePacket ────────────────────────────────────────────── */
interface BridgePacketProps {
  visible: boolean;
  dir: "right" | "left";
  label: string;
  color: string;
  top: string;
}

function BridgePacket({ visible, dir, label, color, top }: BridgePacketProps) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key={label}
          initial={{ x: dir === "right" ? -10 : 90, opacity: 0 }}
          animate={{ x: dir === "right" ? 90 : -10, opacity: [0, 1, 1, 0] }}
          exit={{ opacity: 0 }}
          transition={{ duration: 1.2, times: [0, 0.1, 0.85, 1], ease: "linear" }}
          className="absolute text-[9px] px-1.5 py-0.5 whitespace-nowrap font-mono z-30 chamfer"
          style={{
            top,
            border: `1px solid ${color}`,
            color,
            backgroundColor: "#0a0a0f",
            boxShadow: `0 0 8px ${color}`,
            transform: "translateX(-50%)",
          }}
        >
          {dir === "right" ? `${label} ▶` : `◀ ${label}`}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ─── VectorViz ───────────────────────────────────────────────── */
function VectorViz({
  active,
  userVec,
  candidateVec,
  dotScore,
}: {
  active: boolean;
  userVec: number[];
  candidateVec: number[];
  dotScore: number;
}) {
  const userBars = vectorBars(userVec);
  const candidateBars = vectorBars(candidateVec);

  return (
    <div className="bg-black/60 border border-[#2a2a3a] p-3 chamfer">
      <div className="text-[10px] text-[#00ff88]/70 mb-2">:: DOT_PRODUCT COMPUTATION</div>

      {/* USER_VEC bars */}
      <div className="mb-2">
        <div className="text-[9px] text-[#00d4ff] mb-1">USER_VEC[{userBars.length}]</div>
        <div className="flex gap-1 items-end h-8">
          {userBars.map((v, i) => (
            <motion.div
              key={i}
              initial={{ height: 0 }}
              animate={active ? { height: `${v * 100}%` } : { height: 0 }}
              transition={{ duration: 0.4, delay: i * 0.07 }}
              className="flex-1 bg-[#00d4ff]"
              style={{ minHeight: 2, boxShadow: active ? "0 0 4px #00d4ff" : "none" }}
            />
          ))}
        </div>
      </div>

      {/* dot operator */}
      <div className="text-center text-[#e0e0e0] text-sm my-1">•</div>

      {/* CAND_VEC bars */}
      <div className="mb-3">
        <div className="text-[9px] text-[#ff00ff] mb-1">CAND_VEC[{candidateBars.length}]</div>
        <div className="flex gap-1 items-end h-8">
          {candidateBars.map((v, i) => (
            <motion.div
              key={i}
              initial={{ height: 0 }}
              animate={active ? { height: `${v * 100}%` } : { height: 0 }}
              transition={{ duration: 0.4, delay: i * 0.07 + 0.3 }}
              className="flex-1 bg-[#ff00ff]"
              style={{ minHeight: 2, boxShadow: active ? "0 0 4px #ff00ff" : "none" }}
            />
          ))}
        </div>
      </div>

      {/* Result */}
      <div className="flex items-center justify-between border-t border-[#2a2a3a] pt-2">
        <span className="text-[10px] text-[#8888aa]">= SCORE</span>
        <motion.span
          initial={{ opacity: 0 }}
          animate={active ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 1.4 }}
          className="text-[#00ff88] font-bold text-sm"
          style={{ textShadow: "0 0 10px #00ff88" }}
        >
          {dotScore.toFixed(3)}
        </motion.span>
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────── */
export default function DCCLDemo() {
  const [phase, setPhase] = useState<Phase>(0);
  const [demoData, setDemoData] = useState<RecommendationResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [typedText, setTypedText] = useState("");
  const [itemCount, setItemCount] = useState(0);
  const [embeddingProgress, setEmbeddingProgress] = useState(0);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const deviceScrollRef = useRef<HTMLDivElement | null>(null);
  const extractionRef = useRef<HTMLDivElement | null>(null);
  const receivingRef = useRef<HTMLDivElement | null>(null);
  const inferenceRef = useRef<HTMLDivElement | null>(null);
  const rankingRef = useRef<HTMLDivElement | null>(null);

  const historyItems = demoData?.history.slice(0, HISTORY_PREVIEW_LIMIT) ?? [];
  const products = demoData?.recommendations ?? [];
  const candidateCount = demoData?.cloud.candidate_count ?? 0;
  const embeddingDim = demoData?.cloud.embedding_dim ?? 0;
  const sourceSummary = demoData?.cloud.source_summary;
  const inference = demoData?.inference;

  const clearAllTimers = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  const schedule = useCallback((fn: () => void, ms: number) => {
    const id = setTimeout(fn, ms);
    timers.current.push(id);
    return id;
  }, []);

  /* Phase advancement */
  useEffect(() => {
    if (phase === 0) return;
    const delay = SCHEDULE[phase];
    if (delay !== undefined) {
      const id = schedule(() => setPhase((p) => (p + 1) as Phase), delay);
      return () => clearTimeout(id);
    }
  }, [phase, schedule]);

  /* Phase 1: typewriter */
  useEffect(() => {
    if (phase !== 1) return;
    const lines = [
      `> selecting random local user: ${demoData?.user_id ?? "..."}`,
      `> accessing local_storage/user_${demoData?.user_id ?? "..."}.json...`,
      `> scanning ${demoData?.history.length ?? 0} private interactions...`,
    ];
    let li = 0, ci = 0;
    const iv = setInterval(() => {
      if (li >= lines.length) {
        clearInterval(iv);
        // items appear
        Array.from({ length: historyItems.length }).forEach((_, n) =>
          schedule(() => setItemCount(n + 1), n * 350 + 100)
        );
        schedule(() => setPhase(3), historyItems.length * 350 + 700);
        return;
      }
      if (ci <= lines[li].length) {
        setTypedText(lines.slice(0, li).join("\n") + (li > 0 ? "\n" : "") + lines[li].slice(0, ci));
        ci++;
      } else {
        li++;
        ci = 0;
      }
    }, 28);
    return () => clearInterval(iv);
  }, [demoData, historyItems.length, phase, schedule]);

  /* Phase 10: embedding progress bar */
  useEffect(() => {
    if (phase !== 10) return;
    let p = 0;
    const iv = setInterval(() => {
      p += 4 + Math.random() * 6;
      if (p >= 100) { p = 100; clearInterval(iv); }
      setEmbeddingProgress(Math.round(p));
    }, 40);
    return () => clearInterval(iv);
  }, [phase]);

  /* Keep the active device-side stage in view as the pipeline advances. */
  useEffect(() => {
    const section =
      phase >= 14 ? rankingRef :
      phase >= 11 ? inferenceRef :
      phase >= 9 ? receivingRef :
      phase >= 1 ? extractionRef :
      null;

    if (!section) return;

    const id = window.setTimeout(() => {
      const scroller = deviceScrollRef.current;
      const element = section.current;
      if (!scroller || !element) return;

      const targetTop = Math.max(0, element.offsetTop - scroller.clientHeight * 0.12);
      scroller.scrollTo({ top: targetTop, behavior: "smooth" });
    }, 80);

    return () => window.clearTimeout(id);
  }, [phase]);

  const handleRun = async () => {
    clearAllTimers();
    setTypedText("");
    setItemCount(0);
    setEmbeddingProgress(0);
    setPhase(0);
    setApiError(null);
    setIsLoading(true);

    try {
      const response = await fetchRecommendations(undefined, 5, 50);
      setDemoData(response);
      setPhase(1);
    } catch (error) {
      setDemoData(null);
      setApiError(error instanceof Error ? error.message : "Unable to reach DCCL API");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    clearAllTimers();
    setTypedText("");
    setItemCount(0);
    setEmbeddingProgress(0);
    setApiError(null);
    setPhase(0);
  };

  const step = stepOf(phase);
  const nodeState = (n: 1 | 2 | 3): "idle" | "active" | "done" => {
    const activePhase = [5, 6, 7][n - 1];
    if (phase < activePhase) return "idle";
    if (phase === activePhase) return "active";
    return "done";
  };

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0f] text-[#e0e0e0] font-mono overflow-hidden select-none">

      {/* ── HEADER ─────────────────────────────────────────────── */}
      <header className="border-b border-[#2a2a3a] px-6 py-4 flex flex-col items-center gap-3 relative z-10 bg-[#0a0a0f]">
        <div className="absolute left-6 top-4">
          <Link href="/">
            <button className="flex items-center gap-2 text-[#8888aa] hover:text-[#e0e0e0] transition-colors border border-[#2a2a3a] hover:border-[#8888aa] px-3 py-1.5 chamfer text-xs">
              <ArrowLeft className="w-4 h-4" />
              BACK
            </button>
          </Link>
        </div>
        <h1 className="text-2xl md:text-3xl font-black text-center chromatic-text flex items-center gap-3" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.12em" }}>
          <CloudLightning className="text-[#ff00ff] w-7 h-7 flex-shrink-0" strokeWidth={3} />
          DCCL // DEVICE-CLOUD COLLABORATIVE LEARNING
          <Activity className="text-[#00ff88] w-7 h-7 flex-shrink-0" />
        </h1>

        {apiError && (
          <div className="text-[10px] text-[#ff0055] border border-[#ff0055]/40 bg-[#ff0055]/10 px-3 py-1 chamfer max-w-3xl text-center">
            CLOUD OFFLINE :: {apiError}
          </div>
        )}

        {/* Step indicator */}
        <div className="flex items-center gap-2 text-xs bg-[#1c1c2e] px-5 py-2 chamfer border border-[#2a2a3a]">
          {(["A", "B", "C", "D"] as const).map((s, idx) => {
            const labels = ["EXTRACT", "CANDIDATES", "INFERENCE", "RANKING"];
            const active = step === s;
            const done = step !== null && ["A","B","C","D"].indexOf(step) > idx;
            return (
              <React.Fragment key={s}>
                {idx > 0 && <ChevronRight className="w-3 h-3 opacity-40" />}
                <span
                  className="transition-all duration-300"
                  style={{
                    color: active ? "#00ff88" : done ? "#00ff88aa" : "#8888aa",
                    textShadow: active ? "0 0 8px #00ff88" : "none",
                  }}
                >
                  [{s}] {labels[idx]}
                </span>
              </React.Fragment>
            );
          })}
        </div>
      </header>

      {/* ── MAIN DASHBOARD ───────────────────────────────────────── */}
      <main className="flex-1 min-h-0 grid grid-cols-[1fr_90px_1fr] p-4 gap-0 relative z-10 overflow-x-auto" style={{ minWidth: 1100 }}>

        {/* ── LEFT PANEL: DEVICE ─────────────────────────────────── */}
        <div className="flex min-h-0 flex-col bg-[#12121a] border border-[#2a2a3a] chamfer-lg ambient-glow-device relative overflow-hidden">
          {/* header bar */}
          <div className="bg-[#00ff88]/10 text-[#00ff88] border-b border-[#00ff88]/30 p-3 font-bold flex items-center justify-between" style={{ fontFamily: "var(--font-display)", fontSize: 12, letterSpacing: "0.1em" }}>
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4" />
              [ DEVICE :: USER_TERMINAL ]
            </div>
            <div className="flex items-center gap-2 text-[10px]">
              <span className="text-[#8888aa]">PRIV_ENCLAVE</span>
              <div className="w-2 h-2 rounded-full bg-[#00ff88]" style={{ boxShadow: "0 0 6px #00ff88" }} />
            </div>
          </div>

          <div ref={deviceScrollRef} className="p-4 flex-1 min-h-0 flex flex-col gap-4 overflow-y-auto scroll-smooth">

            {/* ── STEP A: Local extraction ── */}
            <div
              ref={extractionRef}
              className={`transition-all duration-500 ${phase >= 1 && phase <= 3 ? "drop-shadow-[0_0_10px_rgba(0,255,136,0.22)]" : ""}`}
            >
              <div className="text-[10px] text-[#00ff88]/60 mb-1.5 flex items-center gap-1">
                <HardDrive className="w-3 h-3" /> :: LOCAL_HISTORY_EXTRACTION
              </div>
              <div className="bg-black/60 border border-[#2a2a3a] p-3 chamfer font-mono text-xs min-h-[110px] relative">
                {phase === 0 ? (
                  <div className="text-[#8888aa] h-full flex items-center justify-center">AWAITING SEQUENCE...</div>
                ) : (
                  <>
                    <pre className="text-[#00ff88] whitespace-pre-wrap leading-5">
                      {typedText}<span className={phase === 1 ? "animate-blink" : "opacity-0"}>_</span>
                    </pre>
                    <ul className="mt-2 space-y-1">
                      {historyItems.slice(0, itemCount).map((item, i) => (
                        <motion.li
                          key={item.item_id}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.25 }}
                          className="flex items-center gap-2 pl-2 border-l-2 border-[#00ff88]/60 text-[#e0e0e0]"
                        >
                          <HardDrive className="w-2.5 h-2.5 text-[#00d4ff] flex-shrink-0" />
                          ITEM_{item.item_id}: {item.title}
                        </motion.li>
                      ))}
                    </ul>
                    {/* SECURE badge */}
                    <AnimatePresence>
                      {phase >= 3 && (
                        <motion.div
                          initial={{ opacity: 0, y: -4 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-2 flex items-center gap-2 text-[#00ff88] bg-[#00ff88]/10 border border-[#00ff88]/40 px-2 py-1 text-[10px] chamfer"
                        >
                          <Shield className="w-3 h-3" /> SECURE. NO CLOUD UPLOAD.
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </>
                )}
              </div>
            </div>

            {/* ── STEP B: Receiving embeddings ── */}
            <AnimatePresence>
              {phase >= 9 && phase <= 12 && (
                <motion.div
                  ref={receivingRef}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`bg-black/60 border p-3 chamfer transition-shadow duration-500 ${
                    phase >= 9 && phase <= 10
                      ? "border-[#00d4ff] shadow-[0_0_16px_rgba(0,212,255,0.22)]"
                      : "border-[#00d4ff]/40"
                  }`}
                >
                  <div className="text-[10px] text-[#00d4ff] mb-2 flex items-center gap-1">
                    <Network className="w-3 h-3" /> :: RECEIVING EMBEDDINGS FROM CLOUD
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-black/60 overflow-hidden chamfer">
                      <motion.div
                        className="h-full bg-[#00d4ff]"
                        style={{ width: `${embeddingProgress}%`, boxShadow: "0 0 6px #00d4ff" }}
                      />
                    </div>
                    <span className="text-[#00d4ff] text-[10px] w-10 text-right">{embeddingProgress}%</span>
                  </div>
                  <div className="text-[9px] text-[#8888aa] mt-1">
                    {embeddingProgress < 100
                      ? `LOADING ${Math.round((embeddingProgress / 100) * candidateCount)}/${candidateCount} VECTORS...`
                      : `${candidateCount}/${candidateCount} VECTORS RECEIVED · DIM=${embeddingDim}`}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── STEP C: TFLite inference ── */}
            <AnimatePresence>
              {phase >= 11 && (
                <motion.div
                  ref={inferenceRef}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="border border-[#00ff88] bg-[#00ff88]/5 chamfer relative overflow-hidden"
                  style={{ boxShadow: phase >= 11 && phase <= 13 ? "0 0 24px rgba(0,255,136,0.28)" : "0 0 16px rgba(0,255,136,0.2)" }}
                >
                  <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-[#00ff88] to-transparent" />
                  <div className="p-3">
                    <div className="text-[10px] text-[#00ff88] mb-2 flex items-center gap-2">
                      <Database className="w-3 h-3" />
                      [ {inference?.model ?? "user_sage_decoder.tflite"} ] &nbsp;
                      <span className="text-[#00ff88]/50">LOCAL INFERENCE ONLY</span>
                    </div>
                    {inference?.runtime && (
                      <div className="text-[9px] text-[#8888aa] mb-2">
                        runtime: {inference.runtime}
                      </div>
                    )}
                    <VectorViz
                      active={phase >= 12}
                      userVec={inference?.user_vector_preview ?? []}
                      candidateVec={inference?.candidate_vector_preview ?? []}
                      dotScore={inference?.dot_score_preview ?? 0}
                    />
                    <div className="mt-2 flex items-center gap-1 text-[9px] text-[#00ff88]/70">
                      <Lock className="w-2.5 h-2.5" /> ZERO DATA LEAVES DEVICE.
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* ── STEP D: Rankings ── */}
            <AnimatePresence>
              {phase >= 14 && (
                <motion.div
                  ref={rankingRef}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="drop-shadow-[0_0_12px_rgba(0,255,136,0.18)]"
                >
                  <div className="text-[10px] text-[#00ff88] mb-2 border-b border-[#00ff88]/20 pb-1 flex items-center gap-1">
                    <BarChart2 className="w-3 h-3" /> :: TOP RECOMMENDATIONS
                  </div>
                  <div className="space-y-1.5">
                    {products.map((item, i) => (
                      <motion.div
                        key={item.rank}
                        initial={{ opacity: 0, x: -16 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.12 }}
                        className="flex items-center gap-2 bg-[#1c1c2e] border border-[#2a2a3a] p-2 chamfer group hover:border-[#00ff88]/50 transition-colors cursor-default"
                      >
                        <div className="text-[#00ff88] font-bold w-4 text-center text-xs" style={{ textShadow: "0 0 8px #00ff88" }}>
                          {item.rank}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs truncate">{item.title}</div>
                          <div className="flex items-center gap-2 mt-1">
                            <div className="h-1 bg-black/60 flex-1 overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${item.score_percent}%` }}
                                transition={{ duration: 0.8, delay: i * 0.12 + 0.3, ease: "easeOut" }}
                                className="h-full bg-[#00ff88]"
                                style={{ boxShadow: "0 0 5px #00ff88" }}
                              />
                            </div>
                            <span className="text-[9px] text-[#00ff88] w-12 text-right">{item.score_percent}%</span>
                          </div>
                          <div className="text-[9px] text-[#8888aa] mt-0.5 truncate">ASIN {item.asin} · LOGIT {item.score.toFixed(3)}</div>
                        </div>
                        <div className="text-[9px] text-[#8888aa] px-1.5 py-0.5 bg-black/40 border border-[#2a2a3a] flex-shrink-0">
                          {item.tag}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>

        {/* ── CENTER BRIDGE ─────────────────────────────────────── */}
        <div className="relative flex items-stretch justify-center overflow-visible">
          <svg
            className="w-full h-full absolute inset-0"
            viewBox="0 0 90 600"
            preserveAspectRatio="none"
            style={{ overflow: "visible" }}
          >
            {/* Static circuit traces */}
            <line x1="0" y1="160" x2="90" y2="160" stroke="#1c1c2e" strokeWidth="2" />
            <line x1="0" y1="320" x2="90" y2="320" stroke="#1c1c2e" strokeWidth="2" />
            <line x1="45" y1="0" x2="45" y2="600" stroke="#1c1c2e" strokeWidth="1" strokeDasharray="4 6" />

            {/* Outgoing trace glow (phase 3-4) */}
            {(phase === 3 || phase === 4) && (
              <motion.line
                x1="0" y1="160" x2="90" y2="160"
                stroke="#00d4ff"
                strokeWidth="2"
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 1, 0] }}
                transition={{ duration: 1.4, times: [0, 0.1, 0.8, 1] }}
                style={{ filter: "drop-shadow(0 0 4px #00d4ff)" }}
              />
            )}

            {/* Incoming trace glow (phase 8-9) */}
            {(phase === 8 || phase === 9) && (
              <motion.line
                x1="90" y1="320" x2="0" y2="320"
                stroke="#ff00ff"
                strokeWidth="2"
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 1, 0] }}
                transition={{ duration: 1.4, times: [0, 0.1, 0.8, 1] }}
                style={{ filter: "drop-shadow(0 0 4px #ff00ff)" }}
              />
            )}

            {/* Node dots on traces */}
            <circle cx="0" cy="160" r="3" fill="#2a2a3a" />
            <circle cx="90" cy="160" r="3" fill="#2a2a3a" />
            <circle cx="0" cy="320" r="3" fill="#2a2a3a" />
            <circle cx="90" cy="320" r="3" fill="#2a2a3a" />
          </svg>

          {/* Moving packet: ID_LIST → Cloud */}
          <BridgePacket
            visible={phase === 3 || phase === 4}
            dir="right"
            label={`ID_LIST: ${demoData?.history.length ?? 0}`}
            color="#00d4ff"
            top="calc(27% - 10px)"
          />

          {/* Moving packet: EMBEDDINGS ← Cloud */}
          <BridgePacket
            visible={phase === 8 || phase === 9}
            dir="left"
            label={`EMBED: ${candidateCount}v`}
            color="#ff00ff"
            top="calc(53% - 10px)"
          />
        </div>

        {/* ── RIGHT PANEL: CLOUD ─────────────────────────────────── */}
        <div className="flex min-h-0 flex-col bg-[#12121a] border border-[#2a2a3a] chamfer-lg ambient-glow-cloud relative overflow-hidden">
          {/* header bar */}
          <div className="bg-[#ff00ff]/10 text-[#ff00ff] border-b border-[#ff00ff]/30 p-3 font-bold flex items-center justify-between" style={{ fontFamily: "var(--font-display)", fontSize: 12, letterSpacing: "0.1em" }}>
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4" />
              [ CLOUD :: MEGA_SERVER_7 ]
            </div>
            <div className="flex items-center gap-1">
              {[0, 1, 2].map((i) => (
                <div key={i} className="w-2 h-2 rounded-full bg-[#ff00ff]" style={{ opacity: phase >= 5 + i ? 1 : 0.2, boxShadow: phase >= 5 + i ? "0 0 6px #ff00ff" : "none" }} />
              ))}
            </div>
          </div>

          <div className="p-4 flex-1 flex flex-col gap-4">
            {phase < 4 ? (
              <div className="flex-1 flex items-center justify-center text-[#8888aa] text-sm animate-pulse">
                STANDBY...
              </div>
            ) : (
              <>
                <div className="text-[10px] text-[#ff00ff]/60 flex items-center gap-1">
                  <Database className="w-3 h-3" /> :: CANDIDATE_GENERATION_PIPELINE
                </div>

                {/* Incoming packet indicator */}
                <AnimatePresence>
                  {(phase === 4 || phase === 5) && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: [0, 1, 1, 0] }}
                      transition={{ duration: 1.2, times: [0, 0.1, 0.8, 1] }}
                      className="text-[10px] text-[#00d4ff] bg-[#00d4ff]/10 border border-[#00d4ff]/30 px-2 py-1 chamfer flex items-center gap-1"
                    >
                      <Network className="w-3 h-3" /> RECEIVED ID_LIST: {demoData?.history.length ?? 0} ITEMS — INITIATING PIPELINE
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Three cloud nodes */}
                <div className="flex flex-col gap-3">
                  <CloudNode
                    label="[1] GRAPH SEARCH"
                    icon={<Share2 className="w-4 h-4" />}
                    state={nodeState(1)}
                    desc={`KNN graph traversal — ${sourceSummary?.graph_neighbors ?? 0} neighbors`}
                  />
                  <CloudNode
                    label="[2] VECTOR SIM"
                    icon={<Activity className="w-4 h-4" />}
                    state={nodeState(2)}
                    desc={`Cosine similarity — ${sourceSummary?.vector_neighbors ?? 0} vectors`}
                  />
                  <CloudNode
                    label="[3] POPULARITY RANK"
                    icon={<BarChart2 className="w-4 h-4" />}
                    state={nodeState(3)}
                    desc={`Fallback blend — ${sourceSummary?.fallback_items ?? 0} items`}
                  />
                </div>

                {/* Embeddings departing */}
                <AnimatePresence>
                  {phase >= 8 && phase <= 10 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="text-[10px] text-[#ff00ff] bg-[#ff00ff]/10 border border-[#ff00ff]/30 px-2 py-1.5 chamfer flex items-center gap-2"
                    >
                      <Zap className="w-3 h-3" />
                      DISPATCHING {candidateCount} EMBEDDING VECTORS TO DEVICE...
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Inference waiting */}
                <AnimatePresence>
                  {phase >= 11 && phase <= 13 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="p-3 text-[#ff00ff] bg-[#ff00ff]/5 border border-[#ff00ff]/30 text-xs text-center chamfer animate-pulse"
                    >
                      VECTORS DELIVERED — DEVICE COMPUTING LOCALLY...
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Complete */}
                <AnimatePresence>
                  {phase >= 14 && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.97 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="p-4 bg-[#00ff88]/8 border border-[#00ff88]/50 chamfer flex flex-col items-center gap-2 text-center"
                      style={{ boxShadow: "0 0 20px rgba(0,255,136,0.12)" }}
                    >
                      <CheckCircle2 className="w-7 h-7 text-[#00ff88]" style={{ filter: "drop-shadow(0 0 6px #00ff88)" }} />
                      <div className="text-[#00ff88] font-bold text-sm" style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em" }}>
                        PROTOCOL COMPLETE
                      </div>
                      <div className="text-[10px] text-[#8888aa]">USER DATA: NEVER TRANSMITTED.</div>
                      <div className="text-[9px] text-[#2a2a3a] mt-1">
                        ── PRIVACY PRESERVED BY ARCHITECTURE ──
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}
          </div>
        </div>

      </main>

      {/* ── FOOTER ────────────────────────────────────────────────── */}
      <footer className="p-4 flex flex-col items-center gap-2 border-t border-[#2a2a3a] relative z-10 bg-[#0a0a0f]">
        {/* Progress bar during run */}
        {phase > 0 && phase < 15 && (
          <div className="w-80 h-1 bg-[#1c1c2e] chamfer overflow-hidden">
            <motion.div
              className="h-full bg-[#00ff88]"
              animate={{ width: `${(phase / 15) * 100}%` }}
              transition={{ duration: 0.4 }}
              style={{ boxShadow: "0 0 6px #00ff88" }}
            />
          </div>
        )}

        {phase === 0 ? (
          <button
            onClick={handleRun}
            disabled={isLoading}
            className="group relative px-10 py-4 bg-[#00ff88]/10 border-2 border-[#00ff88] text-[#00ff88] font-bold text-lg chamfer glitch-hover flex items-center gap-3 overflow-hidden neon-shadow-green cursor-pointer disabled:opacity-50 disabled:cursor-wait"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "0.12em" }}
          >
            <div className="absolute inset-0 bg-[#00ff88] opacity-0 group-hover:opacity-15 transition-opacity" />
            <Play className="w-5 h-5 fill-current" />
            {isLoading ? "CONNECTING API..." : "RUN SEQUENCE"}
          </button>
        ) : phase === 15 ? (
          <button
            onClick={handleReset}
            className="px-8 py-3 bg-[#ff00ff]/10 border border-[#ff00ff] text-[#ff00ff] font-bold text-sm chamfer hover:bg-[#ff00ff]/20 transition-colors cursor-pointer"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em" }}
          >
            SYSTEM_RESET()
          </button>
        ) : (
          <div
            className="px-8 py-3 border border-[#2a2a3a] text-[#8888aa] font-bold text-sm chamfer bg-[#12121a] flex items-center gap-3"
            style={{ fontFamily: "var(--font-display)", letterSpacing: "0.1em" }}
          >
            <span className="animate-pulse text-[#00ff88]">▶</span>
            PROCESSING [{Math.round((phase / 15) * 100)}%]
            <span className="animate-blink text-[#00ff88]">_</span>
          </div>
        )}
      </footer>

    </div>
  );
}
