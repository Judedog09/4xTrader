import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import axios from 'axios';
import {
  LayoutDashboard, Terminal, Zap, Power, Globe,
  RefreshCw, Search, LineChart as ChartIcon, ChevronRight,
  CheckCircle, AlertCircle, Target, Shield, Flame, BarChart3, PieChart,
  Save, X, Plus, Minus, Eye, EyeOff,
  Calendar, BookOpen, TestTube, Award, Percent,
  TrendingUp, TrendingDown, Play, Edit3, Tag, Settings, Sliders
} from 'lucide-react';

const API = 'http://localhost:8000/api';

// ─────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────
const GlobalStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body, html, #root {
      font-family: 'Inter', sans-serif;
      background: #060810;
      color: #e2e8f0;
      height: 100vh;
      overflow: hidden;
      font-size: 13px;
      line-height: 1.5;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

    /* ── Layout ── */
    .app-shell {
      display: grid;
      grid-template-columns: 220px 1fr 320px;
      grid-template-rows: 100vh;
      width: 100vw;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    .sidebar {
      background: #0d1117;
      border-right: 1px solid #1e2533;
      display: flex;
      flex-direction: column;
      padding: 20px 14px;
      overflow: hidden;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 0 6px;
      margin-bottom: 28px;
    }
    .brand-text { font-size: 15px; font-weight: 900; color: #00ffbb; letter-spacing: -0.3px; }
    .brand-sub { font-size: 9px; color: #3d4f6e; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; }

    .nav-section-label {
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: #3d4f6e;
      padding: 0 10px;
      margin: 16px 0 6px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 9px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12.5px;
      color: #4a5568;
      border: 1px solid transparent;
      margin-bottom: 2px;
      white-space: nowrap;
      transition: color 0.15s, background 0.15s, border-color 0.15s;
    }
    .nav-item:hover { color: #94a3b8; background: #131920; }
    .nav-item.active { color: #00ffbb; background: rgba(0,255,187,0.07); border-color: rgba(0,255,187,0.15); }

    .sidebar-footer {
      margin-top: auto;
      padding-top: 16px;
      border-top: 1px solid #1e2533;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .market-badge {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #131920;
      border: 1px solid #1e2533;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .market-badge-label { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1px; }
    .market-badge-status { display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 700; }
    .pulse-dot { width: 7px; height: 7px; border-radius: 50%; }
    .pulse-dot.live { background: #00ffbb; box-shadow: 0 0 6px #00ffbb; animation: pulse 2s infinite; }
    .pulse-dot.closed { background: #ff4466; }

    .btn-toggle {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      padding: 9px; border-radius: 8px; cursor: pointer; font-weight: 700; font-size: 11.5px;
      border: 1px solid rgba(0,255,187,0.2); background: rgba(0,255,187,0.06); color: #00ffbb;
      transition: all 0.15s;
    }
    .btn-toggle.inactive { border-color: #2d3748; background: #131920; color: #4a5568; }
    .btn-toggle:hover { filter: brightness(1.1); }

    .btn-panic {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      padding: 9px; border-radius: 8px; cursor: pointer; font-weight: 700; font-size: 11.5px;
      border: 1px solid rgba(255,68,102,0.3); background: rgba(255,68,102,0.08); color: #ff4466;
      transition: all 0.15s;
    }
    .btn-panic:hover { background: rgba(255,68,102,0.15); }

    /* ── Main ── */
    .main {
      background: #060810;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 28px;
      border-bottom: 1px solid #1e2533;
      background: #0d1117;
      position: sticky;
      top: 0;
      z-index: 20;
      flex-shrink: 0;
    }

    .topbar-left { display: flex; flex-direction: column; }
    .topbar-clock { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: #fff; line-height: 1; }
    .topbar-label { font-size: 9px; font-weight: 700; color: #3d4f6e; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }

    .topbar-right { display: flex; gap: 12px; }
    .kpi-chip {
      background: #131920;
      border: 1px solid #1e2533;
      border-radius: 10px;
      padding: 10px 16px;
      text-align: right;
    }
    .kpi-chip-label { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .kpi-chip-val { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; }

    .content { padding: 24px 28px; flex: 1; }

    /* ── Cards ── */
    .card {
      background: #0d1117;
      border: 1px solid #1e2533;
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 20px;
    }
    .card-title {
      display: flex; align-items: center; gap: 8px;
      font-size: 14px; font-weight: 700; color: #e2e8f0;
      margin-bottom: 20px;
    }
    .card-title svg { flex-shrink: 0; }

    /* ── Stat grid ── */
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    .stat-tile {
      background: #131920;
      border: 1px solid #1e2533;
      border-radius: 10px;
      padding: 14px 16px;
      transition: border-color 0.15s;
    }
    .stat-tile:hover { border-color: #2d3748; }
    .stat-tile-icon { margin-bottom: 10px; }
    .stat-tile-label { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .stat-tile-val { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 800; }
    .stat-tile-sub { font-size: 10px; color: #4a5568; margin-top: 3px; }

    /* ── Table ── */
    .data-table { width: 100%; border-collapse: collapse; }
    .data-table th { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1px; padding: 0 0 12px; text-align: left; border-bottom: 1px solid #1e2533; }
    .data-table td { padding: 14px 0; border-bottom: 1px solid #131920; vertical-align: middle; }
    .data-table tr:last-child td { border-bottom: none; }
    .data-table tr:hover td { background: #0d1117; }

    /* ── Badge ── */
    .badge { display: inline-block; padding: 3px 8px; border-radius: 5px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .badge-buy { background: rgba(0,255,187,0.12); color: #00ffbb; }
    .badge-sell { background: rgba(255,68,102,0.12); color: #ff4466; }
    .badge-bull { background: rgba(0,255,187,0.12); color: #00ffbb; }
    .badge-bear { background: rgba(255,68,102,0.12); color: #ff4466; }
    .badge-neutral { background: rgba(148,163,184,0.12); color: #94a3b8; }

    /* ── Tabs (within a page) ── */
    .tab-bar { display: flex; gap: 4px; margin-bottom: 20px; }
    .tab-btn {
      padding: 7px 14px; border-radius: 7px; font-size: 11.5px; font-weight: 600;
      cursor: pointer; border: 1px solid transparent; color: #4a5568;
      background: transparent; transition: all 0.15s;
    }
    .tab-btn:hover { color: #94a3b8; background: #131920; }
    .tab-btn.active { color: #00ffbb; background: rgba(0,255,187,0.07); border-color: rgba(0,255,187,0.15); }

    /* ── Inputs ── */
    .input {
      background: #131920;
      border: 1px solid #1e2533;
      border-radius: 7px;
      color: #e2e8f0;
      font-size: 12px;
      padding: 9px 12px;
      width: 100%;
      outline: none;
      transition: border-color 0.15s;
    }
    .input:focus { border-color: #00ffbb55; }
    .input::placeholder { color: #3d4f6e; }

    input[type='range'] {
      -webkit-appearance: none; width: 100%; height: 4px;
      border-radius: 2px; background: #1e2533; outline: none; cursor: pointer;
    }
    input[type='range']::-webkit-slider-thumb {
      -webkit-appearance: none; width: 14px; height: 14px;
      border-radius: 50%; background: #00ffbb; cursor: pointer;
    }

    textarea.input { min-height: 80px; resize: vertical; font-family: inherit; }

    /* ── Chart ── */
    .chart-wrap {
      width: 100%; height: 300px; background: #060810;
      border-radius: 10px; border: 1px solid #1e2533; position: relative; overflow: hidden;
    }
    .chart-empty {
      height: 300px; display: flex; flex-direction: column;
      align-items: center; justify-content: center; color: #3d4f6e; gap: 10px;
    }

    /* ── Heatmap ── */
    .heatmap-cell {
      width: 13px; height: 13px; border-radius: 3px;
      border: 1px solid #1e2533; cursor: default; position: relative;
      transition: transform 0.1s;
    }
    .heatmap-cell:hover { transform: scale(1.5); z-index: 5; }

    /* ── Log monitor ── */
    .log-monitor {
      background: #060810; border-radius: 10px;
      border: 1px solid #1e2533; padding: 8px;
      overflow-y: auto;
      height: calc(100vh - 360px);
      min-height: 300px;
    }
    .log-entry {
      font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
      padding: 9px 12px; border-radius: 6px; margin-bottom: 4px;
      display: flex; gap: 10px; align-items: flex-start;
      border-left: 2px solid transparent;
      color: #64748b;
    }
    .log-entry.trade { border-left-color: #00ffbb; background: rgba(0,255,187,0.04); color: #94a3b8; }
    .log-entry.error { border-left-color: #ff4466; background: rgba(255,68,102,0.04); color: #94a3b8; }
    .log-entry.heartbeat { border-left-color: #00d1ff; background: rgba(0,209,255,0.04); color: #94a3b8; }
    .log-entry.signal { border-left-color: #ffa500; background: rgba(255,165,0,0.04); color: #94a3b8; }

    /* ── Config ── */
    .indicator-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 16px;
      align-items: center;
      padding: 14px 16px;
      border-radius: 9px;
      border: 1px solid #1e2533;
      margin-bottom: 8px;
      background: #0d1117;
      transition: border-color 0.15s;
    }
    .indicator-row.enabled { border-color: rgba(0,255,187,0.2); background: rgba(0,255,187,0.03); }
    .indicator-row-info { display: flex; align-items: center; gap: 10px; }
    .indicator-toggle {
      width: 22px; height: 22px; border-radius: 6px; border: none;
      display: flex; align-items: center; justify-content: center; cursor: pointer;
      flex-shrink: 0; transition: background 0.15s;
    }
    .indicator-toggle.on { background: #00ffbb; }
    .indicator-toggle.off { background: #1e2533; }
    .weight-control { display: flex; align-items: center; gap: 8px; }
    .weight-btn {
      width: 26px; height: 26px; border-radius: 6px; border: 1px solid #1e2533;
      background: #131920; color: #94a3b8; cursor: pointer; display: flex;
      align-items: center; justify-content: center; font-weight: 700; transition: all 0.15s;
    }
    .weight-btn:hover { border-color: #00ffbb55; color: #00ffbb; }
    .weight-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; min-width: 28px; text-align: center; }

    /* ── Slider row ── */
    .param-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }
    .param-label { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }

    /* ── News panel ── */
    .news-panel {
      background: #0d1117;
      border-left: 1px solid #1e2533;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .news-header {
      padding: 18px 20px 14px;
      border-bottom: 1px solid #1e2533;
      flex-shrink: 0;
    }
    .news-scroll { flex: 1; overflow-y: auto; padding: 14px; }
    .news-card {
      background: #131920; border-radius: 10px; padding: 12px 14px;
      margin-bottom: 10px; border-left: 3px solid #1e2533;
      text-decoration: none; display: block; color: inherit;
      transition: background 0.15s, transform 0.15s;
    }
    .news-card:hover { background: #1a2235; transform: translateX(3px); }
    .news-card.bull { border-left-color: #00ffbb; }
    .news-card.bear { border-left-color: #ff4466; }

    /* ── Misc ── */
    .divider { height: 1px; background: #1e2533; margin: 20px 0; }
    .section-label { font-size: 9px; font-weight: 700; color: #3d4f6e; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px; }
    .empty-state { padding: 48px 20px; text-align: center; color: #3d4f6e; }
    .empty-state p { margin-top: 10px; font-size: 12px; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .green { color: #00ffbb; }
    .red { color: #ff4466; }
    .blue { color: #00d1ff; }
    .orange { color: #ffa500; }
    .muted { color: #4a5568; }

    @keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.4} }
    @keyframes fadeUp { from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)} }
    @keyframes spin { from{transform:rotate(0deg)}to{transform:rotate(360deg)} }
    .animate-in { animation: fadeUp 0.25s ease-out; }
    .spinning { animation: spin 1s linear infinite; }
    .live-dot { animation: pulse 1.8s infinite; }
  `}</style>
);

// ─────────────────────────────────────────────
// CHART
// ─────────────────────────────────────────────
const EquityChart = ({ history, timeframe }) => {
  const filtered = useMemo(() => {
    if (!history?.length) return [];
    const now = Date.now();
    const cuts = { '1h':3.6e6,'4h':1.44e7,'1d':8.64e7,'1w':6.048e8,'1m':2.592e9,'all':Infinity };
    return history.filter(h => now - new Date(h.time) < cuts[timeframe]);
  }, [history, timeframe]);

  const { path, minV, maxV, lastY, lastEq, up } = useMemo(() => {
    if (filtered.length < 2) return { path:'', minV:0, maxV:0, lastY:0, lastEq:0, up:true };
    const W=1000, H=280, PX=70, PY=20;
    const vals = filtered.map(h => h.equity);
    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const rng = maxV - minV || 1;
    const pts = filtered.map((h, i) => ({
      x: PX + (i/(filtered.length-1))*(W-PX-10),
      y: H - PY - ((h.equity-minV)/rng)*(H-PY*2)
    }));
    const path = 'M'+pts.map(p=>`${p.x},${p.y}`).join('L');
    const last = pts[pts.length-1];
    return { path, minV, maxV, lastY: last.y, lastEq: filtered[filtered.length-1].equity, up: filtered[filtered.length-1].equity >= filtered[0].equity };
  }, [filtered]);

  if (!path) return (
    <div className="chart-empty">
      <ChartIcon size={40} style={{ opacity:0.3 }} />
      <span style={{ fontSize:12 }}>Awaiting data…</span>
    </div>
  );

  const color = up ? '#00ffbb' : '#ff4466';
  const W=1000, H=280, PX=70;

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width:'100%', height:'100%' }}>
        <defs>
          <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.25"/>
            <stop offset="100%" stopColor={color} stopOpacity="0"/>
          </linearGradient>
        </defs>
        {[0,0.25,0.5,0.75,1].map(t => (
          <line key={t} x1={PX} y1={20+t*240} x2={990} y2={20+t*240} stroke="#1e2533" strokeWidth="1" strokeDasharray="3 5"/>
        ))}
        {[0,0.5,1].map(t => {
          const v = maxV - (maxV-minV)*t;
          return <text key={t} x="4" y={20+t*240+4} fontSize="10" fill="#3d4f6e" fontFamily="'JetBrains Mono',monospace">${v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0)}</text>;
        })}
        <path d={path+`L990,${H-20}L${PX},${H-20}Z`} fill="url(#cg)"/>
        <path d={path} fill="none" stroke={color} strokeWidth="2.5"/>
        <line x1={PX} y1={lastY} x2={985} y2={lastY} stroke="#00d1ff" strokeWidth="1" strokeDasharray="4 4" opacity="0.5"/>
        <text x={990} y={lastY+4} fontSize="10" fill="#00d1ff" fontFamily="'JetBrains Mono',monospace" textAnchor="end">${lastEq.toFixed(2)}</text>
      </svg>
    </div>
  );
};

// ─────────────────────────────────────────────
// PIE CHART (with legend, no overflow labels)
// ─────────────────────────────────────────────
const AllocationChart = ({ positions }) => {
  const data = useMemo(() => {
    if (!positions?.length) return [];
    const total = positions.reduce((s,p)=>s+p.value,0);
    const colors = ['#00ffbb','#00d1ff','#ffa500','#ff4466','#a78bfa','#f472b6','#34d399','#fb923c'];
    return positions.map((p,i) => ({ ...p, pct:(p.value/total)*100, color:colors[i%colors.length] }));
  }, [positions]);

  const slices = useMemo(() => {
    let a = -90;
    return data.map(d => {
      const start = a, end = a + d.pct/100*360;
      a = end;
      const toRad = deg => deg * Math.PI/180;
      const x1=100+80*Math.cos(toRad(start)), y1=100+80*Math.sin(toRad(start));
      const x2=100+80*Math.cos(toRad(end)),   y2=100+80*Math.sin(toRad(end));
      return { ...d, path:`M100,100L${x1},${y1}A80,80,0,${d.pct>50?1:0},1,${x2},${y2}Z` };
    });
  }, [data]);

  if (!data.length) return (
    <div className="empty-state"><PieChart size={36} style={{opacity:0.2, margin:'0 auto'}} /><p>No positions</p></div>
  );

  return (
    <div style={{ display:'flex', alignItems:'center', gap:20, flexWrap:'wrap' }}>
      <svg viewBox="0 0 200 200" style={{ width:160, height:160, flexShrink:0 }}>
        {slices.map((s,i) => <path key={i} d={s.path} fill={s.color} opacity={0.9}/>)}
        <circle cx={100} cy={100} r={42} fill="#0d1117"/>
        <text x={100} y={97} textAnchor="middle" fontSize="11" fontWeight="700" fill="#00ffbb" fontFamily="'JetBrains Mono',monospace">{data.length}</text>
        <text x={100} y={111} textAnchor="middle" fontSize="9" fill="#3d4f6e">positions</text>
      </svg>
      <div style={{ flex:1, minWidth:120 }}>
        {data.map((d,i) => (
          <div key={i} style={{ display:'flex', alignItems:'center', gap:8, marginBottom:7 }}>
            <div style={{ width:8, height:8, borderRadius:2, background:d.color, flexShrink:0 }}/>
            <span style={{ fontWeight:700, fontSize:12 }}>{d.symbol}</span>
            <span style={{ marginLeft:'auto', fontFamily:'JetBrains Mono,monospace', fontSize:11, color:'#94a3b8' }}>{d.pct.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// HEATMAP
// ─────────────────────────────────────────────
const PLHeatmap = ({ data }) => {
  const cells = useMemo(() => {
    const weeks = [];
    for (let w=0; w<13; w++) {
      const week = [];
      for (let d=0; d<7; d++) {
        const idx = w*7+d;
        week.push(data?.[idx] || { date:'', pl:0 });
      }
      weeks.push(week);
    }
    return weeks;
  }, [data]);

  const color = pl => {
    if (!pl) return '#1e2533';
    const t = Math.min(Math.abs(pl)/800,1);
    return pl>0 ? `rgba(0,255,187,${0.15+t*0.85})` : `rgba(255,68,102,${0.15+t*0.85})`;
  };

  if (!data?.length) return (
    <div className="empty-state"><Calendar size={36} style={{opacity:0.2, margin:'0 auto'}}/><p>P/L history will appear here as you trade</p></div>
  );

  return (
    <div>
      <div style={{ display:'flex', gap:3, flexWrap:'nowrap', overflowX:'auto', paddingBottom:8 }}>
        {cells.map((week,w) => (
          <div key={w} style={{ display:'flex', flexDirection:'column', gap:3 }}>
            {week.map((day,d) => (
              <div key={d} className="heatmap-cell" style={{ background:color(day.pl) }}
                title={day.date ? `${day.date}: ${day.pl>=0?'+':''}$${day.pl.toFixed(2)}` : ''} />
            ))}
          </div>
        ))}
      </div>
      <div style={{ display:'flex', gap:16, marginTop:10, fontSize:10, color:'#3d4f6e' }}>
        <span style={{ display:'flex', alignItems:'center', gap:5 }}><span style={{ width:10, height:10, borderRadius:2, background:'#1e2533', display:'inline-block' }}/> No activity</span>
        <span style={{ display:'flex', alignItems:'center', gap:5 }}><span style={{ width:10, height:10, borderRadius:2, background:'rgba(0,255,187,0.5)', display:'inline-block' }}/> Profit</span>
        <span style={{ display:'flex', alignItems:'center', gap:5 }}><span style={{ width:10, height:10, borderRadius:2, background:'rgba(255,68,102,0.5)', display:'inline-block' }}/> Loss</span>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// METRICS GRID
// ─────────────────────────────────────────────
const MetricTile = ({ label, value, sub, color='#e2e8f0', Icon }) => (
  <div className="stat-tile">
    {Icon && <div className="stat-tile-icon"><Icon size={16} color={color}/></div>}
    <div className="stat-tile-label">{label}</div>
    <div className="stat-tile-val" style={{ color }}>{value}</div>
    {sub && <div className="stat-tile-sub">{sub}</div>}
  </div>
);

// ─────────────────────────────────────────────
// ACTIVE POSITIONS TAB
// ─────────────────────────────────────────────
const PositionsTab = ({ stats }) => (
  <div className="animate-in">
    <div className="card">
      <div className="card-title"><Shield size={16} color="#00ffbb"/> Active Positions</div>
      {stats.positions?.length ? (
        <table className="data-table">
          <thead>
            <tr><th>Asset</th><th>Qty</th><th>Entry</th><th>Current</th><th>P/L</th><th style={{textAlign:'right'}}>Value</th></tr>
          </thead>
          <tbody>
            {stats.positions.map(p => {
              const pct = ((p.price-(p.avg_entry_price||p.price))/(p.avg_entry_price||p.price))*100;
              return (
                <tr key={p.symbol}>
                  <td><span style={{ fontWeight:800, fontSize:14 }}>{p.symbol}</span></td>
                  <td className="mono muted">{parseFloat(p.qty).toFixed(2)}</td>
                  <td className="mono muted">${(p.avg_entry_price||p.price).toFixed(2)}</td>
                  <td className="mono">${p.price.toFixed(2)}</td>
                  <td className={`mono ${p.pl>=0?'green':'red'}`} style={{fontWeight:700}}>
                    {p.pl>=0?'+':''}${p.pl.toFixed(2)}<br/>
                    <span style={{fontSize:10}}>{pct>=0?'+':''}{pct.toFixed(2)}%</span>
                  </td>
                  <td className="mono" style={{textAlign:'right', fontWeight:600}}>${p.value.toLocaleString('en-US',{minimumFractionDigits:2})}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <div className="empty-state"><Shield size={48} style={{opacity:0.2, margin:'0 auto'}}/><p>No open positions</p><p style={{marginTop:6}}>Positions will appear here once the bot executes trades</p></div>
      )}
    </div>

    {stats.positions?.length > 0 && (
      <div className="card">
        <div className="card-title"><PieChart size={16} color="#00ffbb"/> Portfolio Allocation</div>
        <div style={{ display:'grid', gridTemplateColumns:'280px 1fr', gap:40, alignItems:'center' }}>
          <AllocationChartLarge positions={stats.positions}/>
          <div>
            {(() => {
              const total = stats.positions.reduce((s,p)=>s+p.value,0);
              const colors = ['#00ffbb','#00d1ff','#ffa500','#ff4466','#a78bfa','#f472b6','#34d399','#fb923c'];
              return stats.positions.map((p,i) => (
                <div key={p.symbol} style={{ display:'flex', alignItems:'center', gap:12, marginBottom:14 }}>
                  <div style={{ width:10, height:10, borderRadius:3, background:colors[i%colors.length], flexShrink:0 }}/>
                  <span style={{ fontWeight:800, fontSize:13, minWidth:60 }}>{p.symbol}</span>
                  <div style={{ flex:1, height:6, background:'#1e2533', borderRadius:3, overflow:'hidden' }}>
                    <div style={{ width:`${(p.value/total*100).toFixed(1)}%`, height:'100%', background:colors[i%colors.length], borderRadius:3 }}/>
                  </div>
                  <span className="mono muted" style={{ minWidth:42, textAlign:'right', fontSize:11 }}>{(p.value/total*100).toFixed(1)}%</span>
                  <span className="mono" style={{ minWidth:80, textAlign:'right', fontSize:11 }}>${p.value.toLocaleString('en-US',{minimumFractionDigits:2})}</span>
                </div>
              ));
            })()}
          </div>
        </div>
      </div>
    )}
  </div>
);

// ─────────────────────────────────────────────
// PERFORMANCE TAB
// ─────────────────────────────────────────────
const PerformanceTab = ({ stats, analytics, bot }) => {
  const { daily_pnl:day=0, weekly_pnl:week=0, monthly_pnl:month=0, all_time_pnl:allTime=0 } = bot;
  const eq = stats.equity || 1;
  return (
    <div className="animate-in">
      <div className="card">
        <div className="card-title"><BarChart3 size={16} color="#00ffbb"/> P/L Overview</div>
        <div className="stat-grid">
          <MetricTile Icon={Calendar} label="Today" value={`${day>=0?'+':''}$${day.toFixed(2)}`} sub={`${(day/eq*100).toFixed(2)}%`} color={day>=0?'#00ffbb':'#ff4466'}/>
          <MetricTile Icon={BarChart3} label="This Week" value={`${week>=0?'+':''}$${week.toFixed(2)}`} sub={`${(week/eq*100).toFixed(2)}%`} color={week>=0?'#00ffbb':'#ff4466'}/>
          <MetricTile Icon={TrendingUp} label="This Month" value={`${month>=0?'+':''}$${month.toFixed(2)}`} sub={`${(month/eq*100).toFixed(2)}%`} color={month>=0?'#00ffbb':'#ff4466'}/>
          <MetricTile Icon={Award} label="All-Time P/L" value={`${allTime>=0?'+':''}$${allTime.toFixed(2)}`} sub={`${analytics?.total_trades||0} total trades`} color={allTime>=0?'#00ffbb':'#ff4466'}/>
        </div>
      </div>
      <div className="card">
        <div className="card-title"><Target size={16} color="#00ffbb"/> Performance Stats</div>
        <div className="stat-grid">
          <MetricTile Icon={Target} label="Win Rate" value={analytics?.win_rate?`${analytics.win_rate.toFixed(1)}%`:'—'} sub={`${analytics?.winning_trades||0}W / ${analytics?.losing_trades||0}L`} color="#00d1ff"/>
          <MetricTile Icon={TrendingUp} label="Avg Win" value={analytics?.avg_win?`$${analytics.avg_win.toFixed(2)}`:'—'} sub="per winning trade" color="#00ffbb"/>
          <MetricTile Icon={TrendingDown} label="Avg Loss" value={analytics?.avg_loss?`$${Math.abs(analytics.avg_loss).toFixed(2)}`:'—'} sub="per losing trade" color="#ff4466"/>
          <MetricTile Icon={Percent} label="Profit Factor" value={analytics?.profit_factor?analytics.profit_factor.toFixed(2):'—'} sub="gross profit / loss" color="#ffa500"/>
          <MetricTile Icon={AlertCircle} label="Max Drawdown" value={analytics?.max_drawdown?`${analytics.max_drawdown.toFixed(2)}%`:'—'} sub="peak to trough" color="#ff4466"/>
          <MetricTile Icon={Shield} label="Sharpe Ratio" value={analytics?.sharpe_ratio?analytics.sharpe_ratio.toFixed(2):'—'} sub="risk-adjusted return" color="#00d1ff"/>
          <MetricTile Icon={Award} label="Best Trade" value={analytics?.best_trade?`$${analytics.best_trade.toFixed(2)}`:'—'} sub={analytics?.best_trade_symbol||'—'} color="#00ffbb"/>
          <MetricTile Icon={AlertCircle} label="Worst Trade" value={analytics?.worst_trade?`$${analytics.worst_trade.toFixed(2)}`:'—'} sub={analytics?.worst_trade_symbol||'—'} color="#ff4466"/>
          <MetricTile Icon={Flame} label="Win Streak" value={analytics?.current_win_streak||0} sub={`Best: ${analytics?.longest_win_streak||0}`} color="#00ffbb"/>
          <MetricTile Icon={Flame} label="Loss Streak" value={analytics?.current_loss_streak||0} sub={`Worst: ${analytics?.longest_loss_streak||0}`} color="#ff4466"/>
          <MetricTile Icon={TrendingUp} label="Recovery Factor" value={analytics?.recovery_factor?analytics.recovery_factor.toFixed(2):'—'} sub="total P/L / max DD" color="#ffa500"/>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// ALLOCATION TAB
// ─────────────────────────────────────────────
const AllocationChartLarge = ({ positions }) => {
  const data = useMemo(() => {
    if (!positions?.length) return [];
    const total = positions.reduce((s,p)=>s+p.value,0);
    const colors = ['#00ffbb','#00d1ff','#ffa500','#ff4466','#a78bfa','#f472b6','#34d399','#fb923c'];
    return positions.map((p,i) => ({ ...p, pct:(p.value/total)*100, color:colors[i%colors.length] }));
  }, [positions]);

  const slices = useMemo(() => {
    let a = -90;
    return data.map(d => {
      const start=a, end=a+d.pct/100*360; a=end;
      const r=deg=>deg*Math.PI/180;
      const x1=160+130*Math.cos(r(start)), y1=160+130*Math.sin(r(start));
      const x2=160+130*Math.cos(r(end)),   y2=160+130*Math.sin(r(end));
      return { ...d, path:`M160,160L${x1},${y1}A130,130,0,${d.pct>50?1:0},1,${x2},${y2}Z` };
    });
  }, [data]);

  if (!data.length) return null;
  return (
    <svg viewBox="0 0 320 320" style={{ width:'100%', maxWidth:320 }}>
      {slices.map((s,i) => <path key={i} d={s.path} fill={s.color} opacity={0.9}/>)}
      <circle cx={160} cy={160} r={65} fill="#0d1117"/>
      <text x={160} y={154} textAnchor="middle" fontSize="13" fontWeight="700" fill="#00ffbb" fontFamily="'JetBrains Mono',monospace">{data.length}</text>
      <text x={160} y={171} textAnchor="middle" fontSize="10" fill="#3d4f6e">positions</text>
    </svg>
  );
};

const AllocationTab = ({ stats }) => (
  <div className="animate-in">
    <div className="card">
      <div className="card-title"><PieChart size={16} color="#00ffbb"/> Portfolio Allocation</div>
      {stats.positions?.length ? (
        <div style={{ display:'grid', gridTemplateColumns:'320px 1fr', gap:48, alignItems:'start' }}>
          <AllocationChartLarge positions={stats.positions}/>
          <table className="data-table">
            <thead>
              <tr><th>Asset</th><th>Value</th><th>Allocation</th><th style={{textAlign:'right'}}>P/L</th></tr>
            </thead>
            <tbody>
              {(() => {
                const total = stats.positions.reduce((s,p)=>s+p.value,0);
                const colors = ['#00ffbb','#00d1ff','#ffa500','#ff4466','#a78bfa','#f472b6','#34d399','#fb923c'];
                return stats.positions.map((p,i) => (
                  <tr key={p.symbol}>
                    <td>
                      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                        <div style={{ width:10, height:10, borderRadius:3, background:colors[i%colors.length], flexShrink:0 }}/>
                        <span style={{ fontWeight:800, fontSize:14 }}>{p.symbol}</span>
                      </div>
                    </td>
                    <td className="mono">${p.value.toLocaleString('en-US',{minimumFractionDigits:2})}</td>
                    <td>
                      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
                        <div style={{ flex:1, height:6, background:'#1e2533', borderRadius:3, overflow:'hidden' }}>
                          <div style={{ width:`${(p.value/total*100).toFixed(1)}%`, height:'100%', background:colors[i%colors.length], borderRadius:3 }}/>
                        </div>
                        <span className="mono muted" style={{ minWidth:38, textAlign:'right' }}>{(p.value/total*100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className={`mono ${p.pl>=0?'green':'red'}`} style={{textAlign:'right', fontWeight:700}}>{p.pl>=0?'+':''}${p.pl.toFixed(2)}</td>
                  </tr>
                ));
              })()}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state"><PieChart size={48} style={{opacity:0.2, margin:'0 auto'}}/><p>No positions to display</p></div>
      )}
    </div>
  </div>
);

// ─────────────────────────────────────────────
// ANALYTICS TAB
// ─────────────────────────────────────────────
const AnalyticsTab = ({ bot, stats }) => {
  const [tf, setTf] = useState('all');
  const tfs = ['1h','4h','1d','1w','1m','all'];

  const perf = useMemo(() => {
    const h = bot.portfolio_history || [];
    if (h.length < 2) return { change:0, pct:0, up:true };
    const first = h[0].equity, last = h[h.length-1].equity;
    return { change:last-first, pct:((last-first)/first)*100, up:last>=first };
  }, [bot.portfolio_history]);

  return (
    <div className="animate-in">
      {/* Chart */}
      <div className="card">
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:20, flexWrap:'wrap', gap:12 }}>
          <div>
            <div className="card-title" style={{ marginBottom:4 }}><ChartIcon size={16} color="#00ffbb"/> Equity Curve</div>
            <span className="mono" style={{ fontSize:13, color:perf.up?'#00ffbb':'#ff4466', fontWeight:700 }}>
              {perf.up?'▲':'▼'} {perf.up?'+':''}${perf.change.toFixed(2)} ({perf.pct.toFixed(2)}%)
            </span>
          </div>
          <div className="tab-bar" style={{ marginBottom:0 }}>
            {tfs.map(t => <button key={t} className={`tab-btn${tf===t?' active':''}`} onClick={()=>setTf(t)}>{t.toUpperCase()}</button>)}
          </div>
        </div>
        <EquityChart history={bot.portfolio_history} timeframe={tf}/>
      </div>

      {/* Heatmap */}
      <div className="card">
        <div className="card-title"><Calendar size={16} color="#00ffbb"/> Daily P/L Heatmap</div>
        <PLHeatmap data={bot.daily_pnl_history}/>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// TRADES / JOURNAL TAB
// ─────────────────────────────────────────────
const TradesTab = ({ trades, onAddNote, onAddTag }) => {
  const [selected, setSelected] = useState(null);
  const [note, setNote] = useState('');
  const [tag, setTag] = useState('');
  const [filter, setFilter] = useState('all');

  const filtered = useMemo(() => {
    if (!trades?.length) return [];
    if (filter==='wins') return trades.filter(t=>t.pl>0);
    if (filter==='losses') return trades.filter(t=>t.pl<0);
    if (filter==='buy') return trades.filter(t=>t.side==='buy');
    if (filter==='sell') return trades.filter(t=>t.side==='sell');
    return trades;
  }, [trades, filter]);

  const submit = (fn, val, set) => { if(val.trim()&&selected){ fn(selected.id,val); set(''); } };

  return (
    <div className="animate-in" style={{ display:'grid', gridTemplateColumns: selected ? '1fr 380px' : '1fr', gap:20 }}>
      <div className="card" style={{ marginBottom:0 }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:20, gap:12 }}>
          <div className="card-title" style={{ marginBottom:0 }}><BookOpen size={16} color="#00ffbb"/> Trade Journal</div>
          <div className="tab-bar" style={{ marginBottom:0 }}>
            {['all','wins','losses','buy','sell'].map(f =>
              <button key={f} className={`tab-btn${filter===f?' active':''}`} onClick={()=>setFilter(f)}>{f}</button>
            )}
          </div>
        </div>

        {filtered.length ? (
          <div style={{ overflowX:'auto' }}>
            <table className="data-table" style={{ minWidth:600 }}>
              <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P/L</th><th>Tags</th></tr></thead>
              <tbody>
                {filtered.map((t,i) => (
                  <tr key={i} style={{ cursor:'pointer', background:selected?.id===t.id?'rgba(0,255,187,0.04)':undefined }} onClick={()=>setSelected(t)}>
                    <td className="mono muted" style={{ fontSize:10 }}>{new Date(t.time).toLocaleDateString()}<br/>{new Date(t.time).toLocaleTimeString()}</td>
                    <td style={{ fontWeight:800 }}>{t.symbol}</td>
                    <td><span className={`badge badge-${t.side}`}>{t.side}</span></td>
                    <td className="mono">{t.qty}</td>
                    <td className="mono">${(t.entry_price||t.price).toFixed(2)}</td>
                    <td className="mono">{t.exit_price?`$${t.exit_price.toFixed(2)}`:'—'}</td>
                    <td className={`mono ${(t.pl||0)>=0?'green':'red'}`} style={{fontWeight:700}}>{(t.pl||0)>=0?'+':''}${(t.pl||0).toFixed(2)}</td>
                    <td>
                      <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
                        {(t.tags||[]).map((tg,j) => <span key={j} style={{ padding:'2px 6px', background:'rgba(0,209,255,0.1)', color:'#00d1ff', fontSize:9, borderRadius:4, fontWeight:700 }}>{tg}</span>)}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state"><BookOpen size={36} style={{opacity:0.2, margin:'0 auto'}}/><p>No trades yet</p></div>
        )}
      </div>

      {selected && (
        <div className="card animate-in" style={{ marginBottom:0 }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16 }}>
            <span style={{ fontWeight:800, fontSize:18 }}>{selected.symbol}</span>
            <button onClick={()=>setSelected(null)} style={{ background:'none', border:'none', color:'#4a5568', cursor:'pointer', padding:4 }}><X size={18}/></button>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10, marginBottom:20 }}>
            {[['Entry', `$${(selected.entry_price||selected.price).toFixed(2)}`],['Exit', selected.exit_price?`$${selected.exit_price.toFixed(2)}`:'Open'],
              ['P/L', `${(selected.pl||0)>=0?'+':''}$${(selected.pl||0).toFixed(2)}`, (selected.pl||0)>=0?'#00ffbb':'#ff4466'],
              ['Return', selected.pl_pct?`${selected.pl_pct.toFixed(2)}%`:'—', (selected.pl||0)>=0?'#00ffbb':'#ff4466']
            ].map(([l,v,c]) => (
              <div key={l} style={{ background:'#131920', borderRadius:8, padding:'10px 12px' }}>
                <div style={{ fontSize:9, color:'#3d4f6e', fontWeight:700, textTransform:'uppercase', letterSpacing:'1px', marginBottom:4 }}>{l}</div>
                <div className="mono" style={{ fontSize:15, fontWeight:800, color:c||'#e2e8f0' }}>{v}</div>
              </div>
            ))}
          </div>

          <div style={{ marginBottom:16 }}>
            <div className="section-label">Tags</div>
            <div style={{ display:'flex', gap:6, flexWrap:'wrap', marginBottom:10 }}>
              {(selected.tags||[]).map((tg,i) => <span key={i} style={{ padding:'4px 10px', background:'rgba(0,209,255,0.1)', color:'#00d1ff', fontSize:11, borderRadius:6, fontWeight:600 }}><Tag size={9} style={{marginRight:4, display:'inline'}}/>{tg}</span>)}
            </div>
            <div style={{ display:'flex', gap:8 }}>
              <input className="input" placeholder="Add tag…" value={tag} onChange={e=>setTag(e.target.value)} onKeyPress={e=>e.key==='Enter'&&submit(onAddTag,tag,setTag)} style={{ flex:1 }}/>
              <button onClick={()=>submit(onAddTag,tag,setTag)} style={{ padding:'9px 14px', background:'rgba(0,209,255,0.1)', border:'1px solid rgba(0,209,255,0.3)', borderRadius:7, color:'#00d1ff', cursor:'pointer', fontWeight:700, fontSize:11, whiteSpace:'nowrap' }}>Add</button>
            </div>
          </div>

          <div>
            <div className="section-label">Notes</div>
            <div style={{ maxHeight:140, overflowY:'auto', marginBottom:10 }}>
              {(selected.notes||[]).map((n,i) => (
                <div key={i} style={{ background:'#131920', borderRadius:8, padding:'10px 12px', marginBottom:8 }}>
                  <div style={{ fontSize:10, color:'#3d4f6e', marginBottom:4 }}>{n.timestamp}</div>
                  <div style={{ fontSize:12, color:'#94a3b8', lineHeight:1.5 }}>{n.text}</div>
                </div>
              ))}
            </div>
            <textarea className="input" placeholder="Add note…" value={note} onChange={e=>setNote(e.target.value)} style={{ marginBottom:8 }}/>
            <button onClick={()=>submit(onAddNote,note,setNote)} style={{ width:'100%', padding:'10px', background:'rgba(0,255,187,0.08)', border:'1px solid rgba(0,255,187,0.2)', borderRadius:7, color:'#00ffbb', cursor:'pointer', fontWeight:700, fontSize:12 }}>
              <Edit3 size={13} style={{marginRight:6, display:'inline'}}/>Save Note
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────
// BACKTEST TAB
// ─────────────────────────────────────────────
const BacktestTab = ({ onRun }) => {
  const [cfg, setCfg] = useState({ start_date:'2024-01-01', end_date:'2024-12-31', initial_capital:100000, auto_threshold:82, sell_threshold:45, risk_per_trade:0.15, indicators:{ RSI:{enabled:true,weight:5}, MACD:{enabled:true,weight:4}, VWAP:{enabled:true,weight:3}, EMA_CROSS:{enabled:false,weight:2}, ATR:{enabled:false,weight:1} } });
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);

  const run = async () => {
    setRunning(true);
    try { setResults(await onRun(cfg)); } catch(e){}
    setRunning(false);
  };

  const retPos = results?.total_return >= 0;

  return (
    <div className="animate-in" style={{ display:'grid', gridTemplateColumns:'340px 1fr', gap:20, alignItems:'start' }}>
      {/* Config panel */}
      <div className="card" style={{ marginBottom:0 }}>
        <div className="card-title"><TestTube size={16} color="#00ffbb"/> Parameters</div>

        <div style={{ marginBottom:16 }}>
          <div className="section-label">Date Range</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
            <input type="date" className="input" value={cfg.start_date} onChange={e=>setCfg({...cfg,start_date:e.target.value})}/>
            <input type="date" className="input" value={cfg.end_date} onChange={e=>setCfg({...cfg,end_date:e.target.value})}/>
          </div>
        </div>

        <div style={{ marginBottom:16 }}>
          <div className="section-label">Initial Capital</div>
          <input type="number" className="input" value={cfg.initial_capital} onChange={e=>setCfg({...cfg,initial_capital:parseFloat(e.target.value)})}/>
        </div>

        <div style={{ marginBottom:16 }}>
          <div className="section-label">Indicators</div>
          {Object.entries(cfg.indicators).map(([k,v]) => (
            <div key={k} className={`indicator-row${v.enabled?' enabled':''}`}>
              <div className="indicator-row-info">
                <button className={`indicator-toggle${v.enabled?' on':' off'}`} onClick={()=>setCfg({...cfg,indicators:{...cfg.indicators,[k]:{...v,enabled:!v.enabled}}})}>
                  {v.enabled&&<CheckCircle size={13} color="#000"/>}
                </button>
                <span style={{ fontWeight:700, fontSize:12 }}>{k}</span>
              </div>
              <div className="weight-control">
                <button className="weight-btn" onClick={()=>setCfg({...cfg,indicators:{...cfg.indicators,[k]:{...v,weight:Math.max(1,v.weight-1)}}})}><Minus size={11}/></button>
                <span className="weight-val" style={{ color:v.enabled?'#00ffbb':'#4a5568' }}>{v.weight}</span>
                <button className="weight-btn" onClick={()=>setCfg({...cfg,indicators:{...cfg.indicators,[k]:{...v,weight:Math.min(10,v.weight+1)}}})}><Plus size={11}/></button>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginBottom:16 }}>
          <div className="section-label">Entry Threshold</div>
          <div className="param-row">
            <input type="range" min={50} max={100} value={cfg.auto_threshold} onChange={e=>setCfg({...cfg,auto_threshold:+e.target.value})}/>
            <span className="mono green" style={{ fontWeight:700, fontSize:13, minWidth:30, textAlign:'right' }}>{cfg.auto_threshold}</span>
          </div>
        </div>

        <div style={{ marginBottom:20 }}>
          <div className="section-label">Exit Threshold</div>
          <div className="param-row">
            <input type="range" min={0} max={70} value={cfg.sell_threshold} onChange={e=>setCfg({...cfg,sell_threshold:+e.target.value})}/>
            <span className="mono red" style={{ fontWeight:700, fontSize:13, minWidth:30, textAlign:'right' }}>{cfg.sell_threshold}</span>
          </div>
        </div>

        <button onClick={run} disabled={running} style={{ width:'100%', padding:12, background:running?'#1e2533':'#00ffbb', border:'none', borderRadius:8, color:running?'#4a5568':'#000', fontWeight:800, fontSize:13, cursor:running?'not-allowed':'pointer', display:'flex', alignItems:'center', justifyContent:'center', gap:8 }}>
          {running ? <><RefreshCw size={15} className="spinning"/>Running…</> : <><Play size={15}/>Run Backtest</>}
        </button>
      </div>

      {/* Results */}
      <div>
        {results ? (
          <div className="animate-in">
            <div className="card" style={{ marginBottom:0 }}>
              <div className="card-title"><BarChart3 size={16} color="#00ffbb"/> Backtest Results</div>
              <div className="stat-grid" style={{ marginBottom:20 }}>
                <MetricTile label="Final Equity" value={`$${results.final_equity?.toLocaleString()}`} color="#00ffbb"/>
                <MetricTile label="Total Return" value={`${retPos?'+':''}${results.total_return?.toFixed(2)}%`} color={retPos?'#00ffbb':'#ff4466'}/>
                <MetricTile label="Win Rate" value={`${results.win_rate?.toFixed(1)}%`} color="#00d1ff"/>
                <MetricTile label="Total Trades" value={results.total_trades||0}/>
                <MetricTile label="Profit Factor" value={results.profit_factor?.toFixed(2)||'—'} color="#ffa500"/>
                <MetricTile label="Max Drawdown" value={`${results.max_drawdown?.toFixed(2)}%`} color="#ff4466"/>
                <MetricTile label="Sharpe Ratio" value={results.sharpe_ratio?.toFixed(2)||'—'} color="#00d1ff"/>
                <MetricTile label="Avg Trade P/L" value={`$${results.avg_trade?.toFixed(2)||'0.00'}`}/>
              </div>

              {results.equity_curve?.length > 1 && (
                <>
                  <div className="section-label">Equity Curve</div>
                  <EquityChart history={results.equity_curve} timeframe="all"/>
                </>
              )}

              <div style={{ marginTop:16, padding:14, background:retPos?'rgba(0,255,187,0.06)':'rgba(255,68,102,0.06)', border:`1px solid ${retPos?'rgba(0,255,187,0.2)':'rgba(255,68,102,0.2)'}`, borderRadius:8, fontSize:12, color:'#94a3b8', lineHeight:1.6 }}>
                <strong style={{ color:retPos?'#00ffbb':'#ff4466' }}>Analysis: </strong>{results.analysis}
              </div>
            </div>
          </div>
        ) : (
          <div className="card" style={{ marginBottom:0 }}>
            <div className="empty-state" style={{ paddingTop:80, paddingBottom:80 }}>
              <TestTube size={48} style={{ opacity:0.2, margin:'0 auto' }}/>
              <p style={{ fontSize:14, marginTop:12 }}>Configure parameters and run the backtest</p>
              <p style={{ marginTop:6 }}>Results and equity curve will appear here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// CONFIG TAB
// ─────────────────────────────────────────────
const ConfigTab = ({ strategy, onUpdate }) => {
  const [cfg, setCfg] = useState(strategy);
  const [dirty, setDirty] = useState(false);

  useEffect(() => { setCfg(strategy); setDirty(false); }, [strategy]);

  const mark = (fn) => { fn(); setDirty(true); };

  const toggleInd = k => mark(() => setCfg(c => ({ ...c, indicators:{ ...c.indicators, [k]:{ ...c.indicators[k], enabled:!c.indicators[k].enabled } } })));
  const adjustWeight = (k,d) => mark(() => setCfg(c => ({ ...c, indicators:{ ...c.indicators, [k]:{ ...c.indicators[k], weight:Math.max(1,Math.min(10,c.indicators[k].weight+d)) } } })));
  const setParam = (f,v) => mark(() => setCfg(c => ({ ...c, [f]:parseFloat(v) })));

  const save = async () => { await onUpdate(cfg); setDirty(false); };

  return (
    <div className="animate-in" style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20, alignItems:'start' }}>
      {/* Indicators */}
      <div className="card" style={{ marginBottom:0 }}>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:20 }}>
          <div className="card-title" style={{ marginBottom:0 }}><Settings size={16} color="#00ffbb"/> Indicators</div>
          <button onClick={save} disabled={!dirty} style={{ padding:'8px 16px', background:dirty?'#00ffbb':'#1e2533', border:'none', borderRadius:7, color:dirty?'#000':'#4a5568', fontWeight:700, cursor:dirty?'pointer':'not-allowed', display:'flex', alignItems:'center', gap:7, fontSize:12 }}>
            <Save size={13}/>{dirty?'Save Changes':'Saved'}
          </button>
        </div>
        {Object.entries(cfg.indicators||{}).map(([k,v]) => (
          <div key={k} className={`indicator-row${v.enabled?' enabled':''}`}>
            <div className="indicator-row-info">
              <button className={`indicator-toggle${v.enabled?' on':' off'}`} onClick={()=>toggleInd(k)}>
                {v.enabled&&<CheckCircle size={13} color="#000"/>}
              </button>
              <div>
                <div style={{ fontWeight:700, fontSize:13 }}>{k}</div>
                <div style={{ fontSize:10, color:'#3d4f6e' }}>{v.desc}</div>
              </div>
            </div>
            <div className="weight-control">
              <button className="weight-btn" onClick={()=>adjustWeight(k,-1)}><Minus size={11}/></button>
              <span className="weight-val" style={{ color:v.enabled?'#00ffbb':'#4a5568' }}>{v.weight}</span>
              <button className="weight-btn" onClick={()=>adjustWeight(k,1)}><Plus size={11}/></button>
            </div>
            <div style={{ fontSize:10, color:v.enabled?'#00ffbb':'#4a5568', fontWeight:700, textAlign:'right', minWidth:50 }}>{v.enabled?'ACTIVE':'OFF'}</div>
          </div>
        ))}
      </div>

      {/* Parameters */}
      <div className="card" style={{ marginBottom:0 }}>
        <div className="card-title"><Sliders size={16} color="#00ffbb"/> Execution Parameters</div>

        {[
          { label:'Auto-Buy Threshold', field:'auto_threshold', min:50, max:100, step:1, color:'#00ffbb', fmt:v=>`${v}` },
          { label:'Sell Threshold', field:'sell_threshold', min:0, max:70, step:1, color:'#ff4466', fmt:v=>`${v}` },
          { label:'Risk Per Trade', field:'risk_per_trade', min:0.05, max:0.5, step:0.05, color:'#ffa500', fmt:v=>`${(v*100).toFixed(0)}%` },
          { label:'Manual Limit', field:'manual_limit', min:0.1, max:0.5, step:0.05, color:'#00d1ff', fmt:v=>`${(v*100).toFixed(0)}%` }
        ].map(p => (
          <div key={p.field} style={{ background:'#131920', borderRadius:9, padding:14, marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:10 }}>
              <div className="param-label" style={{ marginBottom:0 }}>{p.label}</div>
              <span className="mono" style={{ fontWeight:800, fontSize:15, color:p.color }}>{p.fmt(cfg[p.field]||0)}</span>
            </div>
            <input type="range" min={p.min} max={p.max} step={p.step} value={cfg[p.field]||0} onChange={e=>setParam(p.field,e.target.value)}/>
            <div style={{ display:'flex', justifyContent:'space-between', fontSize:9, color:'#3d4f6e', marginTop:4 }}>
              <span>{p.fmt(p.min)}</span><span>{p.fmt(p.max)}</span>
            </div>
          </div>
        ))}

        <div style={{ padding:14, background:'rgba(0,209,255,0.06)', border:'1px solid rgba(0,209,255,0.15)', borderRadius:9, fontSize:11.5, color:'#94a3b8', lineHeight:1.6 }}>
          <strong style={{ color:'#00d1ff' }}>Guide: </strong>
          Higher indicator weights increase their influence on trade signals. The auto-buy threshold sets the minimum confidence to enter a trade. The sell threshold triggers exits when confidence drops below it.
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// HEARTBEAT TAB
// ─────────────────────────────────────────────
const HeartbeatTab = ({ logs }) => {
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [auto, setAuto] = useState(true);
  const endRef = useRef(null);

  useEffect(() => { if(auto) endRef.current?.scrollIntoView({ behavior:'smooth' }); }, [logs, auto]);

  const filtered = useMemo(() => {
    if (!logs?.length) return [];
    let out = logs;
    if (filter==='trades') out = out.filter(l=>/buy|sell|trade/i.test(l));
    else if (filter==='errors') out = out.filter(l=>/error|failed|❌|⚠️/i.test(l));
    else if (filter==='system') out = out.filter(l=>/heartbeat|watchdog|engine|boot/i.test(l));
    else if (filter==='signals') out = out.filter(l=>/scanner|signal|confidence/i.test(l));
    if (query) out = out.filter(l=>l.toLowerCase().includes(query.toLowerCase()));
    return out;
  }, [logs, filter, query]);

  const entryClass = log => {
    if (/error|failed|❌/.test(log)) return 'error';
    if (/buy|sell|✅/.test(log)) return 'trade';
    if (/💓|heartbeat/.test(log)) return 'heartbeat';
    if (/scanner|signal/.test(log)) return 'signal';
    return '';
  };

  return (
    <div className="animate-in">
      <div className="card">
        {/* Controls */}
        <div style={{ display:'flex', gap:10, marginBottom:16, flexWrap:'wrap', alignItems:'center' }}>
          {/* Search */}
          <div style={{ position:'relative', flex:'1', minWidth:160 }}>
            <Search size={13} style={{ position:'absolute', left:10, top:'50%', transform:'translateY(-50%)', color:'#3d4f6e' }}/>
            <input className="input" placeholder="Search logs…" value={query} onChange={e=>setQuery(e.target.value)} style={{ paddingLeft:32 }}/>
          </div>

          {/* Filters */}
          <div className="tab-bar" style={{ marginBottom:0, flexWrap:'wrap' }}>
            {['all','trades','signals','system','errors'].map(f =>
              <button key={f} className={`tab-btn${filter===f?' active':''}`} onClick={()=>setFilter(f)}>{f}</button>
            )}
          </div>

          {/* Auto-scroll */}
          <button onClick={()=>setAuto(a=>!a)} style={{ padding:'7px 12px', borderRadius:7, border:`1px solid ${auto?'rgba(0,255,187,0.2)':'#1e2533'}`, background:auto?'rgba(0,255,187,0.06)':'transparent', color:auto?'#00ffbb':'#4a5568', cursor:'pointer', display:'flex', alignItems:'center', gap:6, fontSize:11, fontWeight:600, whiteSpace:'nowrap' }}>
            {auto?<Eye size={12}/>:<EyeOff size={12}/>} Auto
          </button>
        </div>

        {/* Log area */}
        <div className="log-monitor">
          {filtered.length ? filtered.map((l,i) => (
            <div key={i} className={`log-entry ${entryClass(l)}`}>
              <ChevronRight size={11} color="#00ffbb" style={{ marginTop:1, flexShrink:0 }}/>
              <span style={{ flex:1, wordBreak:'break-word', lineHeight:1.6 }}>{l}</span>
            </div>
          )) : (
            <div className="empty-state" style={{ paddingTop:60, paddingBottom:60 }}>
              <Terminal size={40} style={{ opacity:0.2, margin:'0 auto' }}/>
              <p>{query?'No matching entries':'No logs yet'}</p>
            </div>
          )}
          <div ref={endRef}/>
        </div>

        {/* Footer */}
        <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginTop:12, fontSize:11, color:'#3d4f6e' }}>
          <span className="mono">{filtered.length} / {logs?.length||0} entries</span>
          <div style={{ display:'flex', alignItems:'center', gap:7 }}>
            <div className="pulse-dot live" style={{ width:6, height:6 }}/>
            <span className="mono green" style={{ fontWeight:700 }}>LIVE</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('positions');
  const [stats, setStats] = useState({ equity:0, buying_power:0, cash:0, positions:[] });
  const [bot, setBot] = useState({ user_enabled:true, system_active:false, activity_logs:[], portfolio_history:[], trade_history:[], strategy:{ indicators:{RSI:{enabled:true,weight:5,desc:'Relative Strength Index'},MACD:{enabled:true,weight:4,desc:'Trend Momentum Matrix'},VWAP:{enabled:true,weight:3,desc:'Volume Weighted Price'},EMA_CROSS:{enabled:false,weight:2,desc:'EMA 20/50 Cross'},ATR:{enabled:false,weight:1,desc:'Volatility Filter'}}, auto_threshold:82, sell_threshold:45, risk_per_trade:0.15, manual_limit:0.20 }, analytics:{}, daily_pnl:0, weekly_pnl:0, monthly_pnl:0, all_time_pnl:0, daily_pnl_history:[] });
  const [news, setNews] = useState([]);
  const [time, setTime] = useState('');
  const [toast, setToast] = useState(null);

  // Clock
  useEffect(() => {
    const tick = () => setTime(new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date()));
    tick(); const id = setInterval(tick, 1000); return ()=>clearInterval(id);
  }, []);

  // Sync
  const sync = useCallback(async () => {
    try {
      const [s,b,n] = await Promise.all([
        axios.get(`${API}/stats`,{timeout:4000}),
        axios.get(`${API}/bot-status`,{timeout:4000}),
        axios.get(`${API}/news`,{timeout:4000})
      ]);
      setStats(s.data); setBot(b.data); setNews(n.data.articles||[]);
    } catch(e) {}
  }, []);

  useEffect(() => { sync(); const id=setInterval(sync,5000); return ()=>clearInterval(id); }, [sync]);

  const notify = (msg, ok=true) => { setToast({msg,ok}); setTimeout(()=>setToast(null),3000); };

  const toggleBot = async () => { try { const r=await axios.post(`${API}/toggle-bot`); setBot(b=>({...b,user_enabled:r.data.user_enabled})); notify(r.data.user_enabled?'Bot activated':'Bot paused'); } catch{ notify('Failed',false); } };
  const liquidate = async () => { if(!window.confirm('LIQUIDATE ALL? This cannot be undone.')) return; try { await axios.post(`${API}/close-all`); notify('All positions closed'); sync(); } catch{ notify('Liquidation failed',false); } };
  const updateConfig = async (cfg) => { try { await axios.post(`${API}/update-config`,cfg); setBot(b=>({...b,strategy:cfg})); notify('Config saved'); } catch{ notify('Save failed',false); } };
  const runBacktest = async (cfg) => { const r=await axios.post(`${API}/backtest`,cfg); return r.data; };
  const addNote = async (id,note) => { try { await axios.post(`${API}/add-note`,{trade_id:id,note}); notify('Note added'); sync(); } catch{ notify('Failed',false); } };
  const addTag = async (id,tag) => { try { await axios.post(`${API}/add-tag`,{trade_id:id,tag}); notify('Tag added'); sync(); } catch{ notify('Failed',false); } };

  const navItems = [
    { id:'positions',   Icon:LayoutDashboard, label:'Active Positions' },
    { id:'performance', Icon:BarChart3,        label:'Performance' },
    { id:'analytics',   Icon:ChartIcon,        label:'Analytics' },
    { id:'trades',      Icon:BookOpen,         label:'Trade Journal' },
    { id:'backtest',    Icon:TestTube,         label:'Backtesting' },
    { id:'config',      Icon:Settings,         label:'Configuration' },
    { id:'heartbeat',   Icon:Terminal,         label:'Heartbeat' },
  ];

  return (
    <div className="app-shell">
      <GlobalStyles/>

      {/* Toast */}
      {toast && (
        <div style={{ position:'fixed', top:20, right:20, zIndex:100, padding:'12px 20px', background:toast.ok?'rgba(0,255,187,0.12)':'rgba(255,68,102,0.12)', border:`1px solid ${toast.ok?'rgba(0,255,187,0.35)':'rgba(255,68,102,0.35)'}`, borderRadius:10, color:toast.ok?'#00ffbb':'#ff4466', fontWeight:700, fontSize:13, animation:'fadeUp 0.2s ease-out', boxShadow:'0 8px 24px rgba(0,0,0,0.4)' }}>
          {toast.msg}
        </div>
      )}

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <Zap size={22} color="#00ffbb" fill="#00ffbb"/>
          <div>
            <div className="brand-text">PULSE 4X</div>
            <div className="brand-sub">Titan v23</div>
          </div>
        </div>

        <div className="nav-section-label">Navigation</div>
        <nav>
          {navItems.map(({ id, Icon, label }) => (
            <div key={id} className={`nav-item${tab===id?' active':''}`} onClick={()=>setTab(id)}>
              <Icon size={15}/> {label}
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="market-badge">
            <div>
              <div className="market-badge-label">Market</div>
              <div className="market-badge-status">
                <div className={`pulse-dot${bot.system_active?' live':' closed'}`}/>
                <span style={{ color:bot.system_active?'#00ffbb':'#ff4466' }}>{bot.system_active?'LIVE':'CLOSED'}</span>
              </div>
            </div>
            <div style={{ textAlign:'right' }}>
              <div className="market-badge-label">Bot</div>
              <div style={{ fontSize:11, fontWeight:700, color:bot.user_enabled?'#00ffbb':'#4a5568' }}>{bot.user_enabled?'ACTIVE':'STANDBY'}</div>
            </div>
          </div>

          <button className={`btn-toggle${bot.user_enabled?'':' inactive'}`} onClick={toggleBot}>
            <Power size={14}/>{bot.user_enabled?'System Active':'System Standby'}
          </button>
          <button className="btn-panic" onClick={liquidate}>
            <AlertCircle size={14}/> Panic Liquidate
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        <div className="topbar">
          <div className="topbar-left">
            <div className="topbar-label">Wall Street Clock (EST)</div>
            <div className="topbar-clock">{time}</div>
          </div>
          <div className="topbar-right">
            <div className="kpi-chip">
              <div className="kpi-chip-label">Buying Power</div>
              <div className="kpi-chip-val blue">${(stats.buying_power||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
            </div>
            <div className="kpi-chip">
              <div className="kpi-chip-label">Net Equity</div>
              <div className="kpi-chip-val green">${(stats.equity||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
            </div>
            <div className="kpi-chip">
              <div className="kpi-chip-label">Cash</div>
              <div className="kpi-chip-val" style={{ color:'#e2e8f0' }}>${(stats.cash||0).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})}</div>
            </div>
          </div>
        </div>

        <div className="content">
          {tab==='positions'   && <PositionsTab stats={stats}/>}
          {tab==='performance' && <PerformanceTab stats={stats} analytics={bot.analytics} bot={bot}/>}
          {tab==='analytics'   && <AnalyticsTab bot={bot} stats={stats}/>}
          {tab==='trades'      && <TradesTab trades={bot.trade_history} onAddNote={addNote} onAddTag={addTag}/>}
          {tab==='backtest'    && <BacktestTab onRun={runBacktest}/>}
          {tab==='config'      && <ConfigTab strategy={bot.strategy} onUpdate={updateConfig}/>}
          {tab==='heartbeat'   && <HeartbeatTab logs={bot.activity_logs}/>}
        </div>
      </main>

      {/* News */}
      <aside className="news-panel">
        <div className="news-header">
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
            <div style={{ fontSize:9, fontWeight:700, color:'#3d4f6e', textTransform:'uppercase', letterSpacing:'1.2px' }}>Intelligence Feed</div>
            <Globe size={13} color="#3d4f6e"/>
          </div>
        </div>
        <div className="news-scroll">
          {news.length ? news.map((n,i) => (
            <a key={i} href={n.url} target="_blank" rel="noopener noreferrer" className={`news-card ${n.sentiment==='BULLISH'?'bull':n.sentiment==='BEARISH'?'bear':''}`}>
              <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:7 }}>
                <span style={{ fontSize:10, fontWeight:700, color:'#4a5568' }}>{n.ticker}</span>
                <span className={`badge ${n.sentiment==='BULLISH'?'badge-bull':n.sentiment==='BEARISH'?'badge-bear':'badge-neutral'}`}>{n.sentiment}</span>
              </div>
              <div style={{ fontWeight:600, fontSize:11.5, lineHeight:1.55, color:'#94a3b8' }}>{n.headline}</div>
            </a>
          )) : (
            <div className="empty-state"><Globe size={32} style={{opacity:0.2, margin:'0 auto'}}/><p>Loading feed…</p></div>
          )}
        </div>
      </aside>
    </div>
  );
}