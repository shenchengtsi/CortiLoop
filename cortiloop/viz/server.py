"""
CortiLoop Visualization Panel — memory graph browser and dashboard.

A lightweight web server using Python's built-in http.server.
Serves a single-page app with:
- Force-directed knowledge graph (D3.js)
- Memory statistics dashboard
- Decay curve visualization
- Memory timeline

Usage:
    cortiloop-viz --db cortiloop.db --port 8765
    # Then open http://localhost:8765
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any

from cortiloop.config import CortiLoopConfig
from cortiloop.storage.sqlite_store import SQLiteStore
from cortiloop.models import MemoryState


class VizAPI:
    """API layer for visualization data extraction."""

    def __init__(self, store: SQLiteStore):
        self.store = store

    def get_graph_data(self) -> dict[str, Any]:
        """Get nodes and edges for the knowledge graph."""
        units = self.store.get_active_units(limit=200)
        observations = self.store.get_active_observations(limit=200)

        nodes = []
        node_ids = set()

        for u in units:
            nodes.append({
                "id": u.id,
                "label": u.content[:60],
                "type": "unit",
                "strength": u.base_strength,
                "access_count": u.access_count,
                "entities": u.entities,
                "created": u.created_at.isoformat() if u.created_at else "",
            })
            node_ids.add(u.id)

        for o in observations:
            nodes.append({
                "id": o.id,
                "label": o.content[:60],
                "type": "observation",
                "strength": o.base_strength,
                "access_count": o.access_count,
                "entities": o.entities,
                "dimension": o.dimension,
                "created": o.created_at.isoformat() if o.created_at else "",
            })
            node_ids.add(o.id)

        # Gather edges
        edges = []
        seen_edges = set()
        for nid in node_ids:
            for edge in self.store.get_edges_from(nid):
                if edge.target_id in node_ids:
                    eid = f"{edge.source_id}-{edge.target_id}-{edge.edge_type.value}"
                    if eid not in seen_edges:
                        seen_edges.add(eid)
                        edges.append({
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "type": edge.edge_type.value,
                            "weight": edge.weight,
                            "co_activations": edge.co_activation_count,
                        })

        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        units = self.store.get_active_units(limit=10000)
        observations = self.store.get_active_observations(limit=10000)
        procedurals = self.store.get_active_procedurals()

        # State distribution
        total_units = self.store.count_units()
        active_units = len(units)

        # Access distribution
        access_counts = [u.access_count for u in units]
        avg_access = sum(access_counts) / len(access_counts) if access_counts else 0

        # Strength distribution
        strengths = [u.base_strength for u in units]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0

        return {
            "total_units": total_units,
            "active_units": active_units,
            "observations": len(observations),
            "procedurals": len(procedurals),
            "archive_units": total_units - active_units,
            "avg_access_count": round(avg_access, 2),
            "avg_strength": round(avg_strength, 3),
            "total_edges": sum(1 for u in units for _ in self.store.get_edges_from(u.id)),
        }

    def get_timeline(self) -> list[dict[str, Any]]:
        """Get memory creation timeline."""
        units = self.store.get_active_units(limit=100)
        timeline = []
        for u in sorted(units, key=lambda x: x.created_at or datetime.min):
            timeline.append({
                "id": u.id,
                "content": u.content[:100],
                "type": u.tier.value,
                "created": u.created_at.isoformat() if u.created_at else "",
                "strength": u.base_strength,
                "access_count": u.access_count,
            })
        return timeline

    def get_drilldown(self, category: str) -> list[dict[str, Any]]:
        """Get memory list for a specific stat category."""
        units = self.store.get_active_units(limit=10000)
        observations = self.store.get_active_observations(limit=10000)

        def unit_to_dict(u):
            return {
                "id": u.id, "content": u.content, "type": "unit",
                "tier": u.tier.value, "strength": round(u.base_strength, 3),
                "access_count": u.access_count, "entities": u.entities,
                "created": u.created_at.isoformat() if u.created_at else "",
            }

        def obs_to_dict(o):
            return {
                "id": o.id, "content": o.content, "type": "observation",
                "dimension": o.dimension, "strength": round(o.base_strength, 3),
                "access_count": o.access_count, "entities": o.entities,
                "created": o.created_at.isoformat() if o.created_at else "",
            }

        if category == "active_units":
            return [unit_to_dict(u) for u in units]
        elif category == "observations":
            return [obs_to_dict(o) for o in observations]
        elif category == "procedurals":
            procedurals = self.store.get_active_procedurals()
            return [{
                "id": p.id, "content": f"{p.pattern} → {p.procedure}", "type": "procedural",
                "strength": round(p.base_strength, 3), "access_count": p.access_count,
                "entities": p.entities, "confidence": round(p.confidence, 3),
                "acquisitions": p.acquisition_count,
                "created": p.created_at.isoformat() if p.created_at else "",
            } for p in procedurals]
        elif category == "archive_units":
            ns = self.store.config.namespace
            rows = self.store.conn.execute(
                f"SELECT id, content, entities, created_at, base_strength, access_count, state"
                f" FROM memory_units_{ns} WHERE state != 'active' ORDER BY created_at DESC"
            ).fetchall()
            return [{
                "id": r[0], "content": r[1], "type": "unit",
                "state": r[6], "strength": round(r[4], 3) if r[4] else 0,
                "access_count": r[5] or 0, "entities": json.loads(r[2]) if r[2] else [],
                "created": r[3].isoformat() if r[3] else "",
            } for r in rows]
        elif category == "total_edges":
            # Build ID→content lookup for resolving edge endpoints
            id_map = {}
            for u in units:
                id_map[u.id] = u.content
            for o in observations:
                id_map[o.id] = o.content

            edges = []
            seen = set()
            for u in units:
                for edge in self.store.get_edges_from(u.id):
                    eid = f"{edge.source_id}-{edge.target_id}"
                    if eid not in seen:
                        seen.add(eid)
                        edges.append({
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "source_content": id_map.get(edge.source_id, "")[:120],
                            "target_content": id_map.get(edge.target_id, "")[:120],
                            "source_entities": (next((u.entities for u in units if u.id == edge.source_id), []) or
                                                next((o.entities for o in observations if o.id == edge.source_id), [])),
                            "target_entities": (next((u.entities for u in units if u.id == edge.target_id), []) or
                                                next((o.entities for o in observations if o.id == edge.target_id), [])),
                            "type": edge.edge_type.value, "weight": round(edge.weight, 3),
                            "co_activations": edge.co_activation_count,
                        })
            return edges
        elif category == "avg_access_count":
            return sorted([unit_to_dict(u) for u in units], key=lambda x: -x["access_count"])
        elif category == "avg_strength":
            return sorted([unit_to_dict(u) for u in units], key=lambda x: -x["strength"])
        else:
            return []

    def get_decay_curves(self) -> dict[str, list[dict]]:
        """Get sample decay curves for visualization."""
        days = list(range(0, 91))
        curves = {}
        rates = {"episodic": 0.1, "semantic": 0.03, "procedural": 0.005}

        for tier, rate in rates.items():
            curves[tier] = [
                {"day": d, "strength": round(math.exp(-rate * d), 4)}
                for d in days
            ]

        return curves


# ── HTML/JS Frontend ──

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CortiLoop — Memory Visualization</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
.header { padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 12px; }
.header h1 { font-size: 18px; font-weight: 600; }
.header .badge { background: #238636; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
.tabs { display: flex; gap: 0; padding: 0 24px; background: #161b22; border-bottom: 1px solid #30363d; }
.tab { padding: 10px 16px; cursor: pointer; border-bottom: 2px solid transparent; font-size: 14px; color: #8b949e; }
.tab.active { color: #c9d1d9; border-bottom-color: #f78166; }
.tab:hover { color: #c9d1d9; }
.panel { display: none; height: calc(100vh - 100px); }
.panel.active { display: block; }
#graph-panel { position: relative; }
#graph-panel svg { width: 100%; height: 100%; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; padding: 24px; }
.stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; cursor: pointer; transition: border-color 0.2s, transform 0.1s; }
.stat-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
.stat-card:active { transform: translateY(0); }
.stat-card .value { font-size: 28px; font-weight: 700; color: #58a6ff; }
.stat-card .label { font-size: 12px; color: #8b949e; margin-top: 4px; }
.stat-card .hint { font-size: 10px; color: #484f58; margin-top: 6px; }
.drilldown-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.6); z-index: 200; }
.drilldown-overlay.active { display: flex; justify-content: center; align-items: center; }
.drilldown-panel { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; width: 80%; max-width: 900px; max-height: 80vh; display: flex; flex-direction: column; }
.drilldown-header { padding: 16px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
.drilldown-header h2 { font-size: 16px; font-weight: 600; }
.drilldown-close { background: none; border: 1px solid #30363d; color: #8b949e; border-radius: 6px; padding: 4px 12px; cursor: pointer; font-size: 13px; }
.drilldown-close:hover { border-color: #f78166; color: #f78166; }
.drilldown-body { overflow-y: auto; padding: 16px 20px; flex: 1; }
.drilldown-item { padding: 12px 16px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 8px; }
.drilldown-item .content { font-size: 14px; line-height: 1.5; margin-bottom: 6px; }
.drilldown-item .meta { font-size: 11px; color: #8b949e; display: flex; gap: 12px; flex-wrap: wrap; }
.drilldown-item .meta span { display: inline-flex; align-items: center; gap: 3px; }
.drilldown-item .entities { margin-top: 6px; }
.drilldown-item .entity-tag { display: inline-block; background: #1c2128; border: 1px solid #30363d; border-radius: 4px; padding: 1px 6px; font-size: 11px; color: #d2a8ff; margin-right: 4px; }
.drilldown-count { font-size: 13px; color: #8b949e; }
.timeline-list { padding: 24px; max-width: 800px; }
.timeline-item { padding: 12px 16px; border-left: 3px solid #30363d; margin-left: 16px; margin-bottom: 8px; background: #161b22; border-radius: 0 8px 8px 0; }
.timeline-item.episodic { border-left-color: #f78166; }
.timeline-item.semantic { border-left-color: #58a6ff; }
.timeline-item.procedural { border-left-color: #3fb950; }
.timeline-item .meta { font-size: 11px; color: #8b949e; }
.decay-panel { padding: 24px; }
.decay-panel svg { background: #161b22; border-radius: 8px; border: 1px solid #30363d; }
.tooltip { position: absolute; background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; font-size: 12px; pointer-events: none; z-index: 100; max-width: 300px; }
.node-unit { fill: #f78166; }
.node-observation { fill: #58a6ff; }
.link { stroke: #30363d; stroke-opacity: 0.6; }
.link-co_occurrence { stroke: #8b949e; }
.link-temporal { stroke: #d2a8ff; }
.link-causal { stroke: #f78166; }
.link-semantic { stroke: #58a6ff; }
</style>
</head>
<body>
<div class="header">
  <h1>🧠 CortiLoop</h1>
  <span class="badge">Memory Visualization</span>
</div>
<div class="tabs">
  <div class="tab active" data-panel="graph-panel">Knowledge Graph</div>
  <div class="tab" data-panel="stats-panel">Statistics</div>
  <div class="tab" data-panel="timeline-panel">Timeline</div>
  <div class="tab" data-panel="decay-panel">Decay Curves</div>
</div>
<div id="graph-panel" class="panel active"></div>
<div id="stats-panel" class="panel"></div>
<div id="timeline-panel" class="panel"></div>
<div id="decay-panel" class="panel decay-panel"></div>
<div class="tooltip" id="tooltip" style="display:none"></div>
<div class="drilldown-overlay" id="drilldown-overlay">
  <div class="drilldown-panel">
    <div class="drilldown-header">
      <h2 id="drilldown-title">Details</h2>
      <div style="display:flex;align-items:center;gap:12px">
        <span class="drilldown-count" id="drilldown-count"></span>
        <button class="drilldown-close" onclick="closeDrilldown()">Close</button>
      </div>
    </div>
    <div class="drilldown-body" id="drilldown-body"></div>
  </div>
</div>

<script>
// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');
  });
});

const tooltip = document.getElementById('tooltip');

// ── Knowledge Graph ──
async function loadGraph() {
  const data = await fetch('/api/graph').then(r => r.json());
  const panel = document.getElementById('graph-panel');
  const w = panel.clientWidth, h = panel.clientHeight;

  const svg = d3.select('#graph-panel').append('svg').attr('viewBox', [0, 0, w, h]);
  const g = svg.append('g');
  svg.call(d3.zoom().on('zoom', e => g.attr('transform', e.transform)));

  const sim = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.edges).id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(w/2, h/2))
    .force('collision', d3.forceCollide(20));

  const link = g.selectAll('.link').data(data.edges).enter().append('line')
    .attr('class', d => 'link link-' + d.type)
    .attr('stroke-width', d => Math.max(1, d.weight * 2));

  const node = g.selectAll('circle').data(data.nodes).enter().append('circle')
    .attr('r', d => 5 + (d.access_count || 0) * 0.5)
    .attr('class', d => 'node-' + d.type)
    .attr('opacity', d => 0.3 + d.strength * 0.7)
    .call(d3.drag().on('start', dragStart).on('drag', dragging).on('end', dragEnd));

  const label = g.selectAll('text').data(data.nodes).enter().append('text')
    .text(d => d.label.substring(0, 25))
    .attr('font-size', 9).attr('fill', '#8b949e').attr('dx', 10).attr('dy', 3);

  node.on('mouseover', (e, d) => {
    tooltip.style.display = 'block';
    tooltip.innerHTML = '<b>' + d.label + '</b><br>Type: ' + d.type +
      '<br>Strength: ' + (d.strength||0).toFixed(3) +
      '<br>Accesses: ' + (d.access_count||0) +
      '<br>Entities: ' + (d.entities||[]).join(', ');
  }).on('mousemove', e => {
    tooltip.style.left = (e.pageX + 12) + 'px';
    tooltip.style.top = (e.pageY - 10) + 'px';
  }).on('mouseout', () => { tooltip.style.display = 'none'; });

  sim.on('tick', () => {
    link.attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
        .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
    node.attr('cx', d=>d.x).attr('cy', d=>d.y);
    label.attr('x', d=>d.x).attr('y', d=>d.y);
  });

  function dragStart(e,d) { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }
  function dragging(e,d) { d.fx=e.x; d.fy=e.y; }
  function dragEnd(e,d) { if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }

  if (data.nodes.length === 0) {
    svg.append('text').attr('x', w/2).attr('y', h/2).attr('text-anchor', 'middle')
      .attr('fill', '#8b949e').attr('font-size', 16).text('No memories yet. Use retain() to add some.');
  }
}

// ── Statistics ──
const CARD_CATEGORIES = {
  'Active Units': 'active_units',
  'Observations': 'observations',
  'Procedurals': 'procedurals',
  'Archive': 'archive_units',
  'Avg Strength': 'avg_strength',
  'Graph Edges': 'total_edges',
};

async function loadStats() {
  const s = await fetch('/api/stats').then(r => r.json());
  const panel = document.getElementById('stats-panel');
  panel.innerHTML = '<div class="stats-grid">' +
    card(s.active_units, 'Active Units') +
    card(s.observations, 'Observations') +
    card(s.procedurals || 0, 'Procedurals') +
    card(s.archive_units || 0, 'Archive') +
    card(s.avg_strength, 'Avg Strength') +
    card(s.total_edges, 'Graph Edges') +
    '</div>';
  panel.querySelectorAll('.stat-card').forEach(el => {
    el.addEventListener('click', () => openDrilldown(el.dataset.category, el.dataset.label));
  });
}
function card(v, l) {
  const cat = CARD_CATEGORIES[l] || '';
  return '<div class="stat-card" data-category="' + cat + '" data-label="' + l + '">' +
    '<div class="value">' + v + '</div>' +
    '<div class="label">' + l + '</div>' +
    '<div class="hint">click to view details</div></div>';
}

// ── Drilldown ──
async function openDrilldown(category, label) {
  const overlay = document.getElementById('drilldown-overlay');
  const title = document.getElementById('drilldown-title');
  const count = document.getElementById('drilldown-count');
  const body = document.getElementById('drilldown-body');

  title.textContent = label;
  body.innerHTML = '<div style="color:#8b949e;padding:20px">Loading...</div>';
  count.textContent = '';
  overlay.classList.add('active');

  const items = await fetch('/api/drilldown/' + category).then(r => r.json());
  count.textContent = items.length + ' items';

  if (items.length === 0) {
    body.innerHTML = '<div style="color:#8b949e;padding:20px">No data</div>';
    return;
  }

  if (category === 'total_edges') {
    body.innerHTML = items.map(i => {
      const sTags = (i.source_entities||[]).map(e => '<span class="entity-tag">' + e + '</span>').join('');
      const tTags = (i.target_entities||[]).map(e => '<span class="entity-tag">' + e + '</span>').join('');
      return '<div class="drilldown-item">' +
        '<div class="meta" style="margin-bottom:8px"><span style="color:#f0883e;font-weight:600">' + i.type + '</span><span>weight: ' + i.weight + '</span><span>co-activations: ' + i.co_activations + '</span></div>' +
        '<div style="display:flex;gap:12px;flex-wrap:wrap">' +
          '<div style="flex:1;min-width:200px;padding:8px 10px;background:#1c2128;border-radius:6px;border-left:3px solid #58a6ff">' +
            '<div style="font-size:11px;color:#58a6ff;margin-bottom:4px">Source</div>' +
            '<div class="content">' + (i.source_content || i.source_id.substring(0,8) + '...') + '</div>' +
            (sTags ? '<div class="entities" style="margin-top:4px">' + sTags + '</div>' : '') +
          '</div>' +
          '<div style="flex:1;min-width:200px;padding:8px 10px;background:#1c2128;border-radius:6px;border-left:3px solid #d2a8ff">' +
            '<div style="font-size:11px;color:#d2a8ff;margin-bottom:4px">Target</div>' +
            '<div class="content">' + (i.target_content || i.target_id.substring(0,8) + '...') + '</div>' +
            (tTags ? '<div class="entities" style="margin-top:4px">' + tTags + '</div>' : '') +
          '</div>' +
        '</div>' +
      '</div>';
    }).join('');
  } else {
    body.innerHTML = items.map(i => {
      const tags = (i.entities||[]).map(e => '<span class="entity-tag">' + e + '</span>').join('');
      return '<div class="drilldown-item">' +
        '<div class="content">' + (i.content||'') + '</div>' +
        '<div class="meta">' +
          (i.tier ? '<span>Tier: ' + i.tier + '</span>' : '') +
          (i.dimension ? '<span>Dim: ' + i.dimension + '</span>' : '') +
          '<span>Strength: ' + (i.strength||0) + '</span>' +
          '<span>Accesses: ' + (i.access_count||0) + '</span>' +
          '<span>' + (i.created||'').substring(0,19) + '</span>' +
        '</div>' +
        (tags ? '<div class="entities">' + tags + '</div>' : '') +
        '</div>';
    }).join('');
  }
}

function closeDrilldown() {
  document.getElementById('drilldown-overlay').classList.remove('active');
}
document.getElementById('drilldown-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeDrilldown();
});

// ── Timeline ──
async function loadTimeline() {
  const items = await fetch('/api/timeline').then(r => r.json());
  const panel = document.getElementById('timeline-panel');
  panel.innerHTML = '<div class="timeline-list">' + items.map(i =>
    '<div class="timeline-item ' + i.type + '">' +
    '<div>' + i.content + '</div>' +
    '<div class="meta">str=' + (i.strength||0).toFixed(3) + ' | access=' + i.access_count + ' | ' + (i.created||'').substring(0,19) + '</div></div>'
  ).join('') + '</div>';
  if (items.length === 0) panel.innerHTML = '<div style="padding:24px;color:#8b949e">No memories yet.</div>';
}

// ── Decay Curves ──
async function loadDecay() {
  const curves = await fetch('/api/decay').then(r => r.json());
  const panel = document.getElementById('decay-panel');
  const w = 600, h = 350, m = {t:30, r:30, b:40, l:50};

  const svg = d3.select('#decay-panel').append('svg').attr('width', w).attr('height', h);
  const x = d3.scaleLinear().domain([0, 90]).range([m.l, w-m.r]);
  const y = d3.scaleLinear().domain([0, 1]).range([h-m.b, m.t]);

  svg.append('g').attr('transform', 'translate(0,' + (h-m.b) + ')').call(d3.axisBottom(x).ticks(9))
    .selectAll('text').attr('fill', '#8b949e');
  svg.append('g').attr('transform', 'translate(' + m.l + ',0)').call(d3.axisLeft(y).ticks(5))
    .selectAll('text').attr('fill', '#8b949e');
  svg.selectAll('.domain, .tick line').attr('stroke', '#30363d');

  svg.append('text').attr('x', w/2).attr('y', h-5).attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 12).text('Days');
  svg.append('text').attr('x', -h/2).attr('y', 14).attr('text-anchor', 'middle').attr('fill', '#8b949e').attr('font-size', 12).attr('transform', 'rotate(-90)').text('Strength');

  const colors = {episodic: '#f78166', semantic: '#58a6ff', procedural: '#3fb950'};
  const line = d3.line().x(d => x(d.day)).y(d => y(d.strength)).curve(d3.curveMonotoneX);

  for (const [tier, pts] of Object.entries(curves)) {
    svg.append('path').datum(pts).attr('fill', 'none').attr('stroke', colors[tier]).attr('stroke-width', 2).attr('d', line);
    svg.append('text').attr('x', x(92)).attr('y', y(pts[pts.length-1].strength)).attr('fill', colors[tier]).attr('font-size', 11).text(tier);
  }

  // Threshold lines
  [{v:0.3, l:'Archive'}, {v:0.1, l:'Cold'}].forEach(({v,l}) => {
    svg.append('line').attr('x1', m.l).attr('x2', w-m.r).attr('y1', y(v)).attr('y2', y(v))
      .attr('stroke', '#484f58').attr('stroke-dasharray', '4,4');
    svg.append('text').attr('x', m.l+4).attr('y', y(v)-4).attr('fill', '#484f58').attr('font-size', 10).text(l + ' threshold');
  });
}

// Load all panels
loadGraph();
loadStats();
loadTimeline();
loadDecay();
</script>
</body>
</html>"""


class VizHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the visualization panel."""

    api: VizAPI  # set by server factory

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._respond(200, "text/html", FRONTEND_HTML)
        elif path == "/api/graph":
            self._json_response(self.api.get_graph_data())
        elif path == "/api/stats":
            self._json_response(self.api.get_stats())
        elif path == "/api/timeline":
            self._json_response(self.api.get_timeline())
        elif path == "/api/decay":
            self._json_response(self.api.get_decay_curves())
        elif path.startswith("/api/drilldown/"):
            category = path.split("/api/drilldown/", 1)[1]
            self._json_response(self.api.get_drilldown(category))
        else:
            self._respond(404, "text/plain", "Not found")

    def _respond(self, status: int, content_type: str, body: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def _json_response(self, data):
        self._respond(200, "application/json", json.dumps(data, default=str))

    def log_message(self, format, *args):
        pass  # suppress default logging


def run_viz_server(
    db_path: str = "cortiloop.db",
    namespace: str = "default",
    host: str = "localhost",
    port: int = 8765,
):
    """Start the visualization web server."""
    config = CortiLoopConfig(db_path=db_path, namespace=namespace)
    store = SQLiteStore(config)
    api = VizAPI(store)

    # Bind API to handler class
    VizHandler.api = api

    server = HTTPServer((host, port), VizHandler)
    print(f"🧠 CortiLoop Visualization — http://{host}:{port}")
    print(f"   Database: {db_path} (namespace: {namespace})")
    print(f"   Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CortiLoop Memory Visualization")
    parser.add_argument("--db", default="cortiloop.db", help="Database path")
    parser.add_argument("--namespace", default="default", help="Namespace")
    parser.add_argument("--host", default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    args = parser.parse_args()

    run_viz_server(args.db, args.namespace, args.host, args.port)


if __name__ == "__main__":
    main()
