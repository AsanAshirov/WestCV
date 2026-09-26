/* Antigradient site: renders data/site.json (built by tools/build_site_data.py) and config.json. */
(function () {
  "use strict";

  const CLASS_ORDER = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle", "jaywalking",
    "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
  ];
  // Box colours of the annotated videos (viz.TRACK_COLORS), so charts and videos agree.
  const GROUPS = [["vehicle", "#ffb43c", "vehicles"], ["two_wheeler", "#3ca0ff", "two-wheelers"], ["person", "#50dc50", "people"]];
  const pct = (x) => `${Math.round(x * 100)}%`;
  const RULES = {
    stopped_vehicle: (c) => `Stationary for ≥ ${c.min_stop_s} s on the carriageway while traffic in the same direction keeps passing it ≥ ${pct(c.min_flow_fraction)} of the time. A car waiting in a queue counts only after ${c.queue_max_s} s. Cars parked for the whole clip are ignored.`,
    jaywalking: (c) => `A pedestrian's feet on the learned carriageway outside the drawn crossings for ≥ ${c.min_duration_s} s. People riding a bike or scooter, or getting in and out of a car, are excluded.`,
    wrong_way: (c) => `Heading against the lane's learned direction of travel for ≥ ${pct(c.min_opposite_fraction)} of a path of at least ${c.min_path_bs} box sizes, for ≥ ${c.min_duration_s} s.`,
    congestion: (c) => `Per direction of travel, ≥ ${c.min_vehicles} vehicles slower than ${c.slow_speed} BS/s for ≥ ${c.min_duration_s} s. Shorter queues are ordinary red-light stops.`,
    accident: (c) => `A vehicle doing ≥ ${c.pre_speed} BS/s stops within ${c.max_decel_s} s while touching another road user at the same depth, after the two were closing in.`,
  };
  const RULES_PLAIN = {
    stopped_vehicle: "Stationary on the carriageway while traffic in the same direction keeps passing it; queues at a red light are excluded.",
    jaywalking: "A pedestrian's feet on the carriageway outside the drawn crossings; riders and passengers excluded.",
    wrong_way: "Sustained heading against the lane's learned direction of travel.",
    congestion: "Per direction of travel, many vehicles crawling for longer than a signal cycle.",
    accident: "An abrupt stop while touching another road user at the same depth, after the two were closing in.",
  };
  const WITHHELD = {
    near_miss: "Telling a near miss from ordinary dense traffic needs labelled examples we do not have.",
    red_light: "Needs the signal phase at every moment; we do not estimate it.",
    stop_line: "Needs the signal phase and the stop line position.",
    failure_to_yield: "Needs right-of-way rules at each conflict point.",
    solid_line_crossing: "Needs every lane marking mapped and tracked through occlusions.",
    illegal_turn: "Needs the list of allowed manoeuvres at this junction.",
    illegal_u_turn: "Needs the allowed manoeuvres; U-turns at the median are legal in places.",
    road_obstacle: "COCO has no class for debris or cones; a generic detector adds false alarms.",
    fire_smoke: "No fire or smoke class in COCO; a class that never fires is safer than a false one.",
  };

  // ---------- helpers ----------
  const $ = (sel, root = document) => root.querySelector(sel);
  function h(tag, attrs, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false) node.append(kid);
    return node;
  }
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clock = (s) => {
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
  };
  const nice = (label) => label.replace(/_/g, " ");
  const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  async function loadJSON(url) {
    try {
      const r = await fetch(url, { cache: "no-cache" });
      return r.ok ? await r.json() : null;
    } catch (e) {
      return null;
    }
  }
  function tiou(a, b) {
    const inter = Math.max(0, Math.min(a[1], b[1]) - Math.max(a[0], b[0]));
    const union = Math.max(a[1], b[1]) - Math.min(a[0], b[0]);
    return union > 0 ? inter / union : 0;
  }
  /** Greedy one-to-one matching per class by descending tIoU, as evaluate.py does. */
  function matchEvents(pred, gt, thr) {
    const predHit = new Array(pred.length).fill(-1);
    const gtHit = new Array(gt.length).fill(-1);
    const pairs = [];
    pred.forEach((p, i) => gt.forEach((g, j) => {
      if (p[2] === g[2]) {
        const v = tiou(p, g);
        if (v >= thr) pairs.push([v, i, j]);
      }
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    for (const [, i, j] of pairs) if (predHit[i] < 0 && gtHit[j] < 0) { predHit[i] = j; gtHit[j] = i; }
    return { predHit, gtHit };
  }

  const charts = [];
  function chart(node, option) {
    const c = echarts.init(node, null, { renderer: "canvas" });
    c.setOption(option);
    charts.push(c);
    return c;
  }
  function disposeCharts() {
    while (charts.length) charts.pop().dispose();
  }
  window.addEventListener("resize", () => charts.forEach((c) => c.resize()));
  function axisStyle(dark) {
    const ink = dark ? "#c3cad1" : token("--muted");
    const line = dark ? "rgba(232,235,238,0.18)" : token("--line");
    return {
      axisLabel: { color: ink, fontFamily: "Overpass Mono, monospace", fontSize: 11 },
      axisLine: { lineStyle: { color: line } },
      axisTick: { lineStyle: { color: line } },
      splitLine: { lineStyle: { color: line } },
      nameTextStyle: { color: ink, fontFamily: "Overpass Mono, monospace", fontSize: 11 },
    };
  }
  const tooltipStyle = () => ({
    backgroundColor: token("--surface"), borderColor: token("--line"),
    textStyle: { color: token("--ink"), fontFamily: "Overpass Mono, monospace", fontSize: 12 },
  });

  // ---------- state ----------
  let SITE = null;
  let CFG = null;
  let current = 0;
  let selectedEvent = -1;

  function readHash() {
    const id = (location.hash || "").slice(1);
    if (!SITE || !id) return;
    const i = SITE.videos.findIndex((v) => v.id.split(".")[0] === id);
    if (i >= 0) current = i;
  }

  // ---------- hero ----------
  function renderHero() {
    if (!SITE || !SITE.videos.length) return;
    const v = SITE.videos[0];
    $("#hero-img").src = v.heatmap;
    $("#hero-cam-id").textContent = `CAM ${v.id.split(".")[0]}`;
    $("#hero-cam-res").textContent = v.resolution ? `${v.resolution[0]}×${v.resolution[1]} · ${v.fps}p` : `${v.fps} fps`;
    $("#hero-cam-dur").textContent = clock(v.duration);
    $("#hero-cam").hidden = false;
    const total = SITE.videos.reduce((s, x) => s + x.duration, 0);
    const events = SITE.videos.reduce((s, x) => s + x.events.length, 0);
    const tracks = SITE.videos.reduce((s, x) => s + Object.values(x.n_tracks).reduce((a, b) => a + b, 0), 0);
    $("#facts").replaceChildren(
      h("div", { class: "fact" }, h("b", {}, `${(total / 60).toFixed(1)} min`), h("span", {}, `of sample video in ${SITE.videos.length} files, analysed end to end`)),
      h("div", { class: "fact" }, h("b", {}, tracks.toLocaleString("en")), h("span", {}, "road users tracked")),
      h("div", { class: "fact" }, h("b", {}, events), h("span", {}, "events predicted by the submitted model")),
    );
  }

  // ---------- demo ----------
  function renderDemo() {
    const slot = $("#demo-slot");
    const embed = CFG && CFG.space_embed;
    const page = CFG && (CFG.space_url || CFG.space_embed);
    if (!embed) {
      slot.replaceChildren(h("div", { class: "empty" }, "The demo address is set in site/config.json (space_url, space_embed)."));
      return;
    }
    slot.replaceChildren(
      h("iframe", { class: "demo-frame", src: embed, title: "Live demo on Hugging Face Spaces", loading: "lazy",
        allow: "clipboard-write", referrerpolicy: "no-referrer-when-downgrade" }),
      h("p", { class: "caption" }, "Hosted on Hugging Face Spaces. If the frame stays blank, ",
        h("a", { href: page, target: "_blank", rel: "noopener" }, "open the demo in its own tab"), "."),
    );
  }

  // ---------- samples ----------
  function renderSamples() {
    const slot = $("#samples-slot");
    if (!SITE || !SITE.videos.length) {
      slot.replaceChildren(h("div", { class: "empty" }, "No results yet: run tools/build_site_data.py after the T4 run and commit site/data."));
      return;
    }
    const v = SITE.videos[current];
    const { predHit, gtHit } = matchEvents(v.events, v.gt, 0.5);
    const hasGt = v.gt.length > 0;
    const tabs = h("div", { class: "tabs", role: "tablist", "aria-label": "Sample video" },
      SITE.videos.map((x, i) => h("button", {
        class: "tab", role: "tab", "aria-selected": String(i === current), id: `tab-${i}`,
        onclick: () => { current = i; selectedEvent = -1; history.replaceState(null, "", `#${x.id.split(".")[0]}`); renderData(); },
      }, x.id.split(".")[0], h("small", {}, `${x.events.length} events`))));

    const t = v.timing;
    const x = t ? t.total_sec / t.duration : null;
    const matched = predHit.filter((j) => j >= 0).length;
    const stats = h("div", { class: "stats" },
      h("div", { class: "stat" }, h("b", {}, clock(v.duration)), h("span", {}, "video length")),
      h("div", { class: "stat" }, h("b", {}, v.events.length), h("span", {}, "events predicted")),
      hasGt ? h("div", { class: "stat" }, h("b", {}, `${matched} / ${v.gt.length}`), h("span", {}, "labelled events found at tIoU ≥ 0.5")) : null,
      x !== null ? h("div", { class: "stat" }, h("b", {}, `${x.toFixed(2)}×`),
        h("span", {}, "run time / video length, limit 3× ",
          h("span", { class: `pill ${x <= 2.4 ? "good" : x < 3 ? "neutral" : "bad"}` }, x <= 2.4 ? "20% spare" : x < 3 ? "tight" : "over"))) : null,
      h("div", { class: "stat" }, h("b", {}, v.risk.length ? fmt(Math.max(...v.risk.map((r) => r[1])), 2) : "–"), h("span", {}, "peak accident risk (Part B)")),
    );

    const road = h("div", { class: "road" },
      h("div", { class: "road-head" }, h("h3", {}, "Event timeline"),
        h("div", { class: "legend" }, h("span", {}, h("i", { class: "pred" }), "predicted"), hasGt ? h("span", {}, h("i", { class: "gt" }), "our labels") : null)),
      h("div", { class: "chart", id: "timeline" }));

    const player = h("div", { class: "player", id: "player" });
    if (v.video_url) {
      player.append(h("video", { id: "video", src: v.video_url, controls: true, preload: "metadata", playsinline: true, muted: true }));
    } else {
      player.append(h("div", { class: "none" }, "Pick an event to see the frame from its middle, with tracked road users boxed."));
    }

    const rows = [];
    v.events.forEach((e, i) => rows.push({ e, i, kind: "pred", status: hasGt ? (predHit[i] >= 0 ? "found" : "false alarm") : null }));
    v.gt.forEach((e, j) => { if (gtHit[j] < 0) rows.push({ e, i: -1, kind: "gt", status: "missed" }); });
    rows.sort((a, b) => a.e[0] - b.e[0]);
    const table = h("div", { class: "events" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "class"), h("th", {}, "start"), h("th", {}, "end"), hasGt ? h("th", {}, "vs labels") : null)),
      h("tbody", {}, rows.length ? rows.map((r) => h("tr", {
        "data-t": r.e[0], "data-i": r.i, tabindex: "0",
        onclick: () => selectEvent(r.i, r.e), onkeydown: (ev) => { if (ev.key === "Enter") selectEvent(r.i, r.e); },
      },
      h("td", {}, h("span", { class: "chip", style: `background:${SITE.classes[r.e[2]] || "#888"}` }), nice(r.e[2])),
      h("td", {}, clock(r.e[0])), h("td", {}, clock(r.e[1])),
      hasGt ? h("td", {}, h("span", { class: `pill ${r.status === "found" ? "good" : "bad"}` }, r.status)) : null,
      )) : h("tr", {}, h("td", { colspan: "4", class: "muted" }, "No events in this video.")))));

    const risk = h("div", { class: "panel" }, h("h3", {}, "Accident risk, Part B"),
      h("p", { class: "caption" }, "Causal: each value uses only frames up to that moment. Shaded: the 5 s before a labelled accident, where the metric wants the risk high."),
      h("div", { class: "chart", id: "risk" }));

    slot.replaceChildren(tabs, stats, road, h("div", { class: "viewer" }, player, table), h("div", { style: "margin-top:20px" }, risk), renderMetrics());
    drawTimeline(v);
    drawRisk(v);
    const video = $("#video");
    if (video) video.addEventListener("timeupdate", () => moveCursor(video.currentTime));
  }

  function selectEvent(i, e) {
    const v = SITE.videos[current];
    selectedEvent = i;
    document.querySelectorAll(".events tbody tr").forEach((tr) => tr.classList.toggle("on", Number(tr.dataset.t) === e[0]));
    const video = $("#video");
    if (video) {
      video.currentTime = Math.max(0, e[0] - 1);
      video.play().catch(() => {});
      return;
    }
    const player = $("#player");
    const src = i >= 0 && v.thumbs ? v.thumbs[i] : null;
    if (src) {
      player.replaceChildren(h("img", { src, alt: `${nice(e[2])}, ${clock(e[0])}–${clock(e[1])}` }));
    } else {
      player.replaceChildren(h("div", { class: "none" }, i < 0 ? `Missed ${nice(e[2])} at ${clock(e[0])}–${clock(e[1])}: the model found nothing here.` : "No frame saved for this event."));
    }
    moveCursor((e[0] + e[1]) / 2);
  }

  let timelineChart = null;
  function drawTimeline(v) {
    const node = $("#timeline");
    const labels = CLASS_ORDER.filter((c) => v.events.some((e) => e[2] === c) || v.gt.some((e) => e[2] === c)).reverse();
    node.style.height = `${90 + Math.max(labels.length, 1) * 46}px`;
    const data = [];
    v.gt.forEach((e) => data.push({ value: [labels.indexOf(e[2]), e[0], e[1], 0], ev: e, idx: -1 }));
    v.events.forEach((e, i) => data.push({ value: [labels.indexOf(e[2]), e[0], e[1], 1], ev: e, idx: i }));
    const ax = axisStyle(true);
    const narrow = node.clientWidth < 560;
    timelineChart = chart(node, {
      animation: false,
      grid: { left: narrow ? 100 : 130, right: narrow ? 12 : 24, top: 10, bottom: 64 },
      tooltip: { ...tooltipStyle(), formatter: (p) => `${nice(p.data.ev[2])} · ${p.data.value[3] ? "predicted" : "label"}<br>${clock(p.data.ev[0])} – ${clock(p.data.ev[1])}` },
      xAxis: { type: "value", min: 0, max: v.duration, ...ax, axisLabel: { ...ax.axisLabel, formatter: (s) => clock(s).replace(/\.0$/, "") }, splitLine: { show: false } },
      yAxis: { type: "category", data: labels.map(nice), ...ax, axisTick: { show: false }, axisLine: { show: false },
        splitLine: { show: true, interval: 0, lineStyle: { color: "rgba(232,235,238,0.55)", type: [14, 10], width: 1.5 } },
        axisLabel: { ...ax.axisLabel, color: "#e8ebee", fontSize: narrow ? 10 : 12 } },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, filterMode: "weakFilter" },
        { type: "slider", xAxisIndex: 0, height: 18, bottom: 14, filterMode: "weakFilter", borderColor: "transparent",
          backgroundColor: "rgba(255,255,255,0.05)", fillerColor: "rgba(227,160,8,0.25)", handleStyle: { color: "#e3a008" },
          textStyle: { color: "#c3cad1", fontFamily: "Overpass Mono, monospace" }, labelFormatter: (s) => clock(s) },
      ],
      series: [{
        type: "custom",
        encode: { x: [1, 2], y: 0 },
        data,
        renderItem: (params, api) => {
          const row = api.value(0);
          const a = api.coord([api.value(1), row]);
          const b = api.coord([api.value(2), row]);
          const band = api.size([0, 1])[1];
          const isPred = api.value(3) === 1;
          const hgt = band * 0.3;
          const y = isPred ? a[1] + 1 : a[1] - hgt - 1;
          const color = SITE.classes[labels[row]] || "#e3a008";
          const shape = echarts.graphic.clipRectByRect({ x: a[0], y, width: Math.max(b[0] - a[0], 2), height: hgt },
            { x: params.coordSys.x, y: params.coordSys.y, width: params.coordSys.width, height: params.coordSys.height });
          if (!shape) return null;
          return { type: "rect", shape, style: isPred ? { fill: color, stroke: "rgba(0,0,0,0.25)" } : { fill: "rgba(0,0,0,0)", stroke: "#e8ebee", lineWidth: 1.5 } };
        },
        markLine: { silent: true, symbol: "none", data: [], lineStyle: { color: "#ffffff", width: 1, type: "solid" }, label: { show: false } },
      }],
    });
    timelineChart.on("click", (p) => { if (p.data) selectEvent(p.data.idx, p.data.ev); });
  }
  function moveCursor(t) {
    if (timelineChart) timelineChart.setOption({ series: [{ markLine: { data: [{ xAxis: t }] } }] });
  }

  function drawRisk(v) {
    const ax = axisStyle(false);
    const windows = v.gt.filter((e) => e[2] === "accident").map((e) => [{ xAxis: Math.max(0, e[0] - 5) }, { xAxis: e[0] }]);
    chart($("#risk"), {
      animation: false,
      grid: { left: 44, right: 16, top: 16, bottom: 56 },
      tooltip: { ...tooltipStyle(), trigger: "axis", valueFormatter: (x) => fmt(x, 3) },
      xAxis: { type: "value", min: 0, max: v.duration, ...ax, splitLine: { show: false }, axisLabel: { ...ax.axisLabel, formatter: (s) => clock(s).replace(/\.0$/, "") } },
      yAxis: { type: "value", min: 0, max: 1, interval: 0.25, ...ax },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 16, bottom: 10, borderColor: "transparent", fillerColor: "rgba(227,160,8,0.2)",
        textStyle: { color: token("--muted") }, labelFormatter: (s) => clock(s) }],
      series: [{
        type: "line", data: v.risk, showSymbol: false, sampling: "lttb", lineStyle: { color: "#d62728", width: 1.5 },
        areaStyle: { color: "rgba(214,39,40,0.12)" },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 0.5 }], lineStyle: { color: token("--muted"), type: "dashed" },
          label: { formatter: "alarm", position: "insideEndTop", color: token("--muted"), fontFamily: "Overpass Mono, monospace" } },
        markArea: { silent: true, itemStyle: { color: "rgba(227,160,8,0.18)" }, data: windows },
      }],
    });
    if (!v.risk.length) $("#risk").replaceChildren(h("div", { class: "empty" }, "Part B was off for this video."));
  }

  function renderMetrics() {
    const m = SITE.metrics;
    if (!m || !m.part_a) return null;
    const a = m.part_a;
    const b = m.part_b;
    const rows = a.classes.map((c) => {
      const pc = a.per_class[c];
      return h("tr", {},
        h("td", {}, h("span", { class: "chip", style: `background:${SITE.classes[c] || "#888"}` }), nice(c)),
        ...["0.3", "0.5", "0.7"].map((k) => h("td", { class: "num" }, fmt(pc[k].f1))),
        h("td", { class: "num" }, h("b", {}, fmt(pc.f1_mean))),
        h("td", { class: "num" }, `${pc["0.5"].tp} / ${pc["0.5"].fp} / ${pc["0.5"].fn}`));
    });
    return h("div", { style: "margin-top:28px" },
      h("div", { class: "stats" },
        h("div", { class: "stat" }, h("b", {}, fmt(a.score_a)), h("span", {}, "Score A, macro-F1 over tIoU 0.3/0.5/0.7")),
        b ? h("div", { class: "stat" }, h("b", {}, fmt(b.score_b)), h("span", {}, `Score B · AP ${fmt(b.ap, 2)} · alarm F1 ${fmt(b.f1_alarm, 2)} · mTTA ${fmt(b.mtta_sec, 1)} s`)) : null,
        h("div", { class: "stat" }, h("b", {}, fmt(m.model_score)), h("span", {}, b ? "model score M = 0.7·A + 0.3·B" : "model score M = A (no accidents labelled)"))),
      h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "class"), h("th", { class: "num" }, "F1 @0.3"), h("th", { class: "num" }, "F1 @0.5"),
          h("th", { class: "num" }, "F1 @0.7"), h("th", { class: "num" }, "mean"), h("th", { class: "num" }, "TP / FP / FN @0.5"))),
        h("tbody", {}, rows))),
      h("p", { class: "caption" }, "Scored with the official evaluate.py against our labels of all sample videos. These are the videos the thresholds were tuned on, so treat the numbers as an upper bound."));
  }

  // ---------- EDA ----------
  function renderEda() {
    const slot = $("#eda-slot");
    if (!SITE || !SITE.videos.length) {
      slot.replaceChildren(h("div", { class: "empty" }, "No EDA data yet."));
      return;
    }
    const v = SITE.videos[current];
    const table = h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ["video", "length", "frame", "codec", "bitrate", "size", "analysed", "tracks: vehicles / two-wheelers / people"]
        .map((x, i) => h("th", { class: i === 4 || i === 5 ? "num" : null }, x)))),
      h("tbody", {}, SITE.videos.map((x) => {
        const s = x.stream || {};
        return h("tr", {},
          h("td", {}, x.id), h("td", {}, clock(x.duration)),
          h("td", {}, x.resolution ? `${x.resolution[0]}×${x.resolution[1]}` : "–", h("span", { class: "sub" }, `${x.fps} fps`)),
          h("td", {}, s.codec ? `${s.codec} ${s.profile || ""}` : "–", h("span", { class: "sub" }, s.pix_fmt || "")),
          h("td", { class: "num" }, s.mbps ? `${s.mbps} Mb/s` : "–"), h("td", { class: "num" }, s.size_gb ? `${s.size_gb} GB` : "–"),
          h("td", {}, `${x.analysed_fps} fps`),
          h("td", {}, GROUPS.map((g) => x.n_tracks[g[0]]).join(" / ")));
      }))));
    const name = v.id.split(".")[0];
    slot.replaceChildren(
      table,
      h("p", { class: "eyebrow", style: "margin:28px 0 14px" }, `Charts for ${name} · switch videos in the tabs above`),
      h("div", { class: "grid-2" },
        h("div", { class: "panel" }, h("h3", {}, "Road users in view"), h("p", { class: "caption" }, "Distinct tracks per 10 s window"), h("div", { class: "chart", id: "counts" })),
        h("div", { class: "panel" }, h("h3", {}, "Vehicle speeds"), h("p", { class: "caption" }, "All tracked vehicle samples, BS/s; 1 BS/s ≈ 9 km/h for a car"), h("div", { class: "chart", id: "speeds" }))),
      h("div", { class: "grid-2", style: "margin-top:20px" },
        h("figure", { class: "panel", style: "margin:0" }, h("h3", {}, "Where road users were"),
          h("p", { class: "caption" }, "Foot points of every track, log scale. Bright bands are lanes and crossings."),
          h("img", { src: v.heatmap, alt: `Heat map of road-user positions in ${name}`, loading: "lazy" })),
        h("figure", { class: "panel", style: "margin:0" }, h("h3", {}, "Which way each lane flows"),
          h("p", { class: "caption" }, "Learned from moving vehicles; colour is direction. Wrong-way and congestion rules read this map."),
          h("img", { src: v.flow, alt: `Direction of travel learned in ${name}`, loading: "lazy" }))),
      v.brightness ? h("div", { class: "panel", style: "margin-top:20px" }, h("h3", {}, "Scene brightness"),
        h("p", { class: "caption" }, "Mean grey level per second (0–255): lighting changes and exposure jumps"), h("div", { class: "chart", id: "brightness" })) : null,
    );
    const ax = axisStyle(false);
    chart($("#counts"), {
      animation: false, grid: { left: 40, right: 12, top: 30, bottom: 30 },
      legend: { top: 0, textStyle: { color: token("--muted"), fontFamily: "Overpass Mono, monospace" } },
      tooltip: { ...tooltipStyle(), trigger: "axis" },
      xAxis: { type: "category", data: v.counts.t.map((s) => clock(s).replace(/\.0$/, "")), ...ax, splitLine: { show: false } },
      yAxis: { type: "value", ...ax },
      series: GROUPS.map(([key, color, label]) => ({ name: label, type: "line", stack: "all", showSymbol: false, data: v.counts[key],
        lineStyle: { color, width: 1 }, itemStyle: { color }, areaStyle: { color, opacity: 0.35 } })),
    });
    chart($("#speeds"), {
      animation: false, grid: { left: 48, right: 12, top: 16, bottom: 36 },
      tooltip: { ...tooltipStyle(), trigger: "axis" },
      xAxis: { type: "category", data: v.speed_hist.edges.map((e) => e.toFixed(2)), name: "BS/s", nameLocation: "middle", nameGap: 24, ...ax, splitLine: { show: false } },
      yAxis: { type: "value", ...ax },
      series: [{ type: "bar", data: v.speed_hist.counts, itemStyle: { color: GROUPS[0][1] }, barCategoryGap: "8%" }],
    });
    if (v.brightness) {
      chart($("#brightness"), {
        animation: false, grid: { left: 40, right: 12, top: 16, bottom: 30 },
        tooltip: { ...tooltipStyle(), trigger: "axis" },
        xAxis: { type: "category", data: v.brightness.t.map((s) => clock(s).replace(/\.0$/, "")), ...ax, splitLine: { show: false } },
        yAxis: { type: "value", ...ax, scale: true },
        series: [{ type: "line", data: v.brightness.v, showSymbol: false, lineStyle: { color: token("--amber"), width: 1.5 } }],
      });
    }
  }

  // ---------- approach: class table ----------
  function renderRules() {
    const slot = $("#rules-slot");
    const cls = SITE && SITE.config ? SITE.config.classes : null;
    const crossings = SITE ? SITE.videos.some((v) => v.has_crosswalks) : false;
    const rows = CLASS_ORDER.map((c) => {
      const conf = cls && cls[c];
      let status = "withheld";
      let text = WITHHELD[c];
      if (RULES[c]) {
        text = conf ? RULES[c](conf) : RULES_PLAIN[c];
        status = !conf || conf.enabled ? "predicted" : "switched off";
        if (c === "jaywalking" && conf && conf.require_crosswalks && !crossings) status = "off: no crossing map";
      }
      return h("tr", {},
        h("td", {}, h("span", { class: "chip", style: `background:${(SITE && SITE.classes[c]) || "#888"}` }), nice(c)),
        h("td", {}, h("span", { class: `pill ${status === "predicted" ? "good" : "neutral"}` }, status)),
        h("td", { style: "font-family:var(--serif);font-size:0.95rem" }, text));
    });
    slot.replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "class"), h("th", {}, "status"), h("th", {}, "rule, with the thresholds used in the submission"))),
      h("tbody", {}, rows))));
  }

  // ---------- team ----------
  function renderTeam() {
    const slot = $("#team-slot");
    const team = (CFG && CFG.team ? CFG.team : []).filter((m) => m.name);
    if (!team.length) {
      slot.replaceChildren(h("div", { class: "empty" }, "Team members are listed in site/config.json."));
      return;
    }
    slot.replaceChildren(...team.map((m) => {
      const initials = m.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
      return h("article", { class: "member" },
        h("div", { class: "who" },
          m.photo ? h("img", { class: "avatar", src: m.photo, alt: m.name }) : h("div", { class: "avatar", "aria-hidden": "true" }, initials),
          h("div", {}, h("h3", {}, m.name), h("div", { class: "role" }, m.role || ""))),
        m.did && m.did.length ? h("ul", {}, m.did.map((d) => h("li", {}, d))) : null,
        h("div", { class: "links" },
          m.github ? h("a", { href: m.github, target: "_blank", rel: "noopener" }, "GitHub") : null,
          m.linkedin ? h("a", { href: m.linkedin, target: "_blank", rel: "noopener" }, "LinkedIn") : null));
    }));
  }

  // ---------- report page: live numbers ----------
  function renderReport() {
    const slot = $("#report-results");
    if (!slot) return;
    if (!SITE || !SITE.videos.length) {
      slot.replaceChildren(h("div", { class: "empty" }, "Results appear here once site/data/site.json is built."));
      return;
    }
    const timing = h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, ["video", "length", "Part A", "Part B", "total", "× length", "events"].map((x, i) => h("th", { class: i ? "num" : null }, x)))),
      h("tbody", {}, SITE.videos.map((v) => {
        const t = v.timing || {};
        return h("tr", {}, h("td", {}, v.id), h("td", { class: "num" }, clock(v.duration)),
          h("td", { class: "num" }, t.part_a_sec !== undefined ? `${t.part_a_sec} s` : "–"),
          h("td", { class: "num" }, t.part_b_sec !== undefined ? `${t.part_b_sec} s` : "–"),
          h("td", { class: "num" }, t.total_sec !== undefined ? `${t.total_sec} s` : "–"),
          h("td", { class: "num" }, t.total_sec ? `${(t.total_sec / t.duration).toFixed(2)}×` : "–"),
          h("td", { class: "num" }, v.events.length));
      }))));
    slot.replaceChildren(timing, h("p", { class: "caption" }, "Official harness, one T4 GPU. The limit is 3× the video length."), renderMetrics() || h("span"));
  }

  function renderData() {
    disposeCharts();
    timelineChart = null;
    if (document.body.dataset.page === "report") { renderReport(); return; }
    renderHero();
    renderSamples();
    renderEda();
    renderRules();
  }

  function renderFooter() {
    const info = $("#build-info");
    if (info && SITE) info.textContent = `results built ${SITE.generated} from commit ${SITE.commit}`;
    const repo = CFG && CFG.repo;
    if (repo) document.querySelectorAll("a#repo-link, a[data-repo]").forEach((a) => { a.href = repo; });
  }

  async function main() {
    [SITE, CFG] = await Promise.all([loadJSON("data/site.json"), loadJSON("config.json")]);
    readHash();
    if (document.body.dataset.page === "home") {
      renderDemo();
      renderTeam();
    }
    renderData();
    renderFooter();
    // Charts take their colours from the theme: redraw when it changes.
    const redraw = () => renderData();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
    new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }
  main();
})();
