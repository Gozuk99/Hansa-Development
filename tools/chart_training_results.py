"""Create a compact HTML dashboard from curriculum-training CSV results."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from dataclasses import dataclass, field
import html
import json
import math
from pathlib import Path
import statistics
import webbrowser


DEFAULT_INPUT = Path("training_output/curriculum/results.csv")
DEFAULT_OUTPUT = Path("training_output/curriculum/results_chart.html")
CHART_COLUMNS = (
    ("latest_loss", "Latest training loss"),
    ("rolling_mean_loss", "Rolling mean loss"),
)
COLORS = {"training": "#2563eb", "evaluation": "#f97316"}
MOVEMENT_COUNT_FIELDS = (
    "move_action_count",
    "spent_action_count",
    "pointless_move_workflows",
    "repeated_move_penalties",
    "all_move_turn_penalties",
    "moves_creating_claimable_route",
    "move_claim_conversions",
)


def _chart_ceiling(value, interval, *, minimum, maximum=None):
    ceiling = max(minimum, (math.floor(value / interval) + 1) * interval)
    return min(ceiling, maximum) if maximum is not None else ceiling


DASHBOARD_SCRIPT = r"""
document.querySelectorAll('.chart').forEach(section => {
  const canvas = section.querySelector('canvas');
  const context = canvas.getContext('2d');
  const tooltip = section.querySelector('.tooltip');
  const data = JSON.parse(section.dataset.series);
  const state = {log: true, focus: true, left: 0, right: 1, plotted: []};
  const MAX_VISIBLE_GROUPS = 750;

  const percentile = (values, fraction) => {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * fraction;
    const lower = Math.floor(position), upper = Math.min(lower + 1, sorted.length - 1);
    return sorted[lower] * (upper - position) + sorted[upper] * (position - lower);
  };
  const format = value => value.toLocaleString(undefined, {maximumFractionDigits: 2});
  const aggregate = points => {
    if (points.length <= MAX_VISIBLE_GROUPS) {
      return points.map(point => ({
        game: point[0], value: point[1], minimum: point[1], maximum: point[1],
        firstGame: point[0], lastGame: point[0], count: 1,
      }));
    }
    const groupSize = Math.ceil(points.length / MAX_VISIBLE_GROUPS);
    const groups = [];
    for (let start = 0; start < points.length; start += groupSize) {
      const chunk = points.slice(start, start + groupSize);
      const values = chunk.map(point => point[1]);
      groups.push({
        game: (chunk[0][0] + chunk[chunk.length - 1][0]) / 2,
        value: values.reduce((total, value) => total + value, 0) / values.length,
        minimum: Math.min(...values), maximum: Math.max(...values),
        firstGame: chunk[0][0], lastGame: chunk[chunk.length - 1][0], count: chunk.length,
      });
    }
    return groups;
  };

  function draw() {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = 390;
    canvas.width = width * ratio; canvas.height = height * ratio;
    canvas.style.height = `${height}px`; context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const margin = {left: 68, right: 20, top: 18, bottom: 38};
    const all = [...data.training, ...data.evaluation];
    const fullMinX = Math.min(...all.map(point => point[0]));
    const fullMaxX = Math.max(...all.map(point => point[0]));
    const span = fullMaxX - fullMinX || 1;
    const minX = fullMinX + span * state.left, maxX = fullMinX + span * state.right;
    const visible = all.filter(point => point[0] >= minX && point[0] <= maxX);
    const rawValues = visible.map(point => point[1]);
    const positiveValues = rawValues.filter(value => value > 0);
    const cap = state.focus ? percentile(rawValues, .95) : Math.max(...rawValues);
    const floor = Math.max(
      state.focus ? percentile(positiveValues, .05) : Math.min(...positiveValues), 1e-9
    );
    const transform = value => state.log ? Math.log10(Math.max(value, floor)) : value;
    const minY = state.log ? transform(floor) : Math.min(0, ...rawValues);
    const maxY = transform(cap) === minY ? minY + 1 : transform(cap);
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const px = x => margin.left + (x - minX) / (maxX - minX || 1) * plotWidth;
    const pyTransformed = y => margin.top + (maxY - y) / (maxY - minY) * plotHeight;
    const py = y => pyTransformed(transform(Math.max(floor, Math.min(y, cap))));

    context.clearRect(0, 0, width, height);
    context.strokeStyle = '#334155'; context.fillStyle = '#cbd5e1'; context.font = '12px system-ui';
    for (let tick = 0; tick <= 5; tick++) {
      const y = margin.top + plotHeight * tick / 5;
      const transformed = maxY - (maxY - minY) * tick / 5;
      const label = state.log ? 10 ** transformed : transformed;
      context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width-margin.right, y);
      context.stroke(); context.fillText(format(label), 4, y + 4);
    }
    context.fillText(`game ${Math.round(minX)}`, margin.left, height - 10);
    context.fillText(`game ${Math.round(maxX)}`, width - 92, height - 10);
    state.plotted = [];

    const drawSeries = (name, points, color) => {
      const rawShown = points.filter(point => point[0] >= minX && point[0] <= maxX);
      const shown = aggregate(rawShown);
      if (shown.some(point => point.count > 1)) {
        context.fillStyle = color + '18'; context.beginPath();
        shown.forEach((point, index) => {
          const x = px(point.game), y = py(point.maximum);
          index ? context.lineTo(x, y) : context.moveTo(x, y);
        });
        [...shown].reverse().forEach(point => context.lineTo(px(point.game), py(point.minimum)));
        context.closePath(); context.fill();
      }
      context.strokeStyle = color; context.lineWidth = 1.5; context.beginPath();
      shown.forEach((point, index) => {
        const x = px(point.game), y = py(point.value);
        index ? context.lineTo(x, y) : context.moveTo(x, y);
      });
      context.stroke();
      shown.forEach(point => {
        const x = px(point.game), y = py(point.value);
        const outlier = point.maximum > cap || point.minimum < floor;
        context.fillStyle = outlier ? '#dc2626' : color;
        context.beginPath(); context.arc(x, y, outlier ? 4 : 2.5, 0, Math.PI * 2); context.fill();
        state.plotted.push({...point, x, y, name, outlier});
      });
      if (name === 'training' && rawShown.length > 2) {
        context.strokeStyle = '#16a34a'; context.lineWidth = 2; context.beginPath();
        aggregate(data.median.filter(point => point[0] >= minX && point[0] <= maxX))
          .forEach((point, index) => {
          const x = px(point.game), y = py(point.value);
          index ? context.lineTo(x, y) : context.moveTo(x, y);
        });
        context.stroke();
      }
    };
    drawSeries('training', data.training, '#2563eb');
    drawSeries('evaluation', data.evaluation, '#f97316');
    const trendPoints = data.training.filter(point => point[1] > 0);
    if (trendPoints.length > 1) {
      const transformed = trendPoints.map(point => [point[0], transform(point[1])]);
      const meanX = transformed.reduce((total, point) => total + point[0], 0) / transformed.length;
      const meanY = transformed.reduce((total, point) => total + point[1], 0) / transformed.length;
      const denominator = transformed.reduce((total, point) => total + (point[0] - meanX) ** 2, 0);
      const slope = denominator ? transformed.reduce(
        (total, point) => total + (point[0] - meanX) * (point[1] - meanY), 0
      ) / denominator : 0;
      const intercept = meanY - slope * meanX;
      const startX = Math.max(minX, transformed[0][0]);
      const endX = Math.min(maxX, transformed[transformed.length - 1][0]);
      const trendY = x => Math.max(minY, Math.min(maxY, intercept + slope * x));
      context.strokeStyle = '#7c3aed'; context.lineWidth = 3; context.setLineDash([9, 6]);
      context.beginPath(); context.moveTo(px(startX), pyTransformed(trendY(startX)));
      context.lineTo(px(endX), pyTransformed(trendY(endX))); context.stroke();
      context.setLineDash([]);
    }
  }

  section.querySelector('[data-action="scale"]').onclick = event => {
    state.log = !state.log; event.target.textContent = `Scale: ${state.log ? 'log' : 'linear'}`; draw();
  };
  section.querySelector('[data-action="focus"]').onclick = event => {
    state.focus = !state.focus;
    event.target.textContent = `Range: ${state.focus ? 'focus 95%' : 'all values'}`; draw();
  };
  section.querySelector('[data-action="reset"]').onclick = () => {
    state.left = 0; state.right = 1; draw();
  };
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const position = event.offsetX / canvas.clientWidth;
    const factor = event.deltaY < 0 ? .8 : 1.25;
    const span = Math.min(1, (state.right - state.left) * factor);
    const center = state.left + (state.right - state.left) * position;
    state.left = Math.max(0, Math.min(1 - span, center - span * position));
    state.right = state.left + span; draw();
  }, {passive: false});
  canvas.addEventListener('mousemove', event => {
    let nearest = null, distance = 12;
    state.plotted.forEach(point => {
      const candidate = Math.hypot(point.x - event.offsetX, point.y - event.offsetY);
      if (candidate < distance) { nearest = point; distance = candidate; }
    });
    if (!nearest) { tooltip.hidden = true; return; }
    tooltip.hidden = false; tooltip.style.left = `${event.offsetX + 14}px`;
    tooltip.style.top = `${event.offsetY + 14}px`;
    const games = nearest.count === 1 ? `Game ${format(nearest.firstGame)}`
      : `Games ${format(nearest.firstGame)}–${format(nearest.lastGame)}`;
    tooltip.textContent = nearest.count === 1
      ? `${games}: ${format(nearest.value)}${nearest.outlier ? ' (outside focused range)' : ''}`
      : `${games}: average ${format(nearest.value)}, minimum ${format(nearest.minimum)}, `
        + `maximum ${format(nearest.maximum)} (${format(nearest.count)} games)`;
  });
  canvas.addEventListener('mouseleave', () => { tooltip.hidden = true; });
  new ResizeObserver(draw).observe(canvas); draw();
});
document.querySelectorAll('.evaluation-performance').forEach(container => {
  const update = () => container.querySelectorAll('[data-evaluation-panel]').forEach(panel => {
    const map = container.querySelector('[data-evaluation-map]').value;
    const players = container.querySelector('[data-evaluation-players]').value;
    panel.hidden = panel.dataset.map !== map || panel.dataset.players !== players;
  });
  container.querySelector('[data-evaluation-map]').addEventListener('change', update);
  container.querySelector('[data-evaluation-players]').addEventListener('change', update);
  update();
});
"""


@dataclass
class Series:
    """Complete ordered plot data; the browser groups only the visible window."""

    max_points: int
    points: list[tuple[float, float]] = field(default_factory=list)

    def add(self, x_value: float, y_value: float) -> None:
        self.points.append((x_value, y_value))


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _row_value(row: dict[str, str], column: str) -> float | None:
    return _number(row.get(column))


def _json_list(value: str | None):
    try:
        result = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return result if isinstance(result, list) else []


def read_results(path: Path, max_points: int):
    series = {
        (column, run_type): Series(max_points)
        for column, _title in CHART_COLUMNS
        for run_type in ("training", "evaluation")
    }
    series["rolling_median_50", "training"] = Series(max_points)
    recent_training_losses = deque(maxlen=50)
    counts = {
        "run_type": Counter(),
        "curriculum_stage": Counter(),
        "completion_reason": Counter(),
        "map": Counter(),
        "player_count": Counter(),
        "tier_games": Counter(),
        "tier_wins": Counter(),
        "tier_games_by_player_count": Counter(),
        "tier_wins_by_player_count": Counter(),
        "tier_score_total_by_player_count": Counter(),
        "tier_score_games_by_player_count": Counter(),
        "winning_score_total": Counter(),
        "winning_score_count": Counter(),
        "winning_score_min": Counter(),
        "winning_score_max": Counter(),
        "losing_score_total": Counter(),
        "losing_score_count": Counter(),
        "losing_score_min": Counter(),
        "losing_score_max": Counter(),
        "ties_by_player_count": Counter(),
        "evaluation_batches": {},
        "evaluation_map_batches": {},
        "evaluation_player_batches": {},
        "evaluation_map_player_batches": {},
        "evaluation_versions": {},
        "current_evaluation_suite_version": 0,
    }
    row_count = 0
    training_game_number = 0
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"game#", "run_type"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required CSV column(s): {', '.join(sorted(missing))}")

        for row_count, row in enumerate(reader, start=1):
            run_type = row.get("run_type", "").strip().lower()
            if run_type not in ("training", "training_timeout", "evaluation"):
                continue
            if run_type == "training_timeout":
                counts["run_type"][run_type] += 1
                counts["completion_reason"]["action_limit"] += 1
                continue
            game_number = _number(row.get("game#")) or float(row_count)
            if run_type == "training":
                training_game_number += 1
            chart_game_number = (
                float(training_game_number) if run_type == "training" else game_number
            )
            latest_loss = _row_value(row, "latest_loss")
            if run_type == "training" and latest_loss is not None:
                recent_training_losses.append(latest_loss)
                series["rolling_median_50", "training"].add(
                    chart_game_number, statistics.median(recent_training_losses)
                )
            for name in counts:
                if name in row:
                    counts[name][row.get(name, "unknown") or "unknown"] += 1
            player_count = row.get("player_count", "unknown") or "unknown"
            assigned_tiers = _json_list(row.get("tier_to_seat_assignments"))
            winner_tiers = _json_list(row.get("winner_tier"))
            final_scores = _json_list(row.get("final_player_scores"))
            if run_type == "evaluation":
                batch = int(_number(row.get("batch#")) or 0)
                map_num = row.get("map", "unknown") or "unknown"
                suite_version = int(_number(row.get("evaluation_suite_version")) or 1)
                version = counts["evaluation_versions"].setdefault(
                    suite_version,
                    {
                        "batches": {},
                        "map_batches": {},
                        "player_batches": {},
                        "map_player_batches": {},
                    },
                )
                targets = (
                    (version["batches"], batch),
                    (version["map_batches"], (map_num, batch)),
                    (version["player_batches"], (player_count, batch)),
                    (
                        version["map_player_batches"],
                        (map_num, player_count, batch),
                    ),
                )
                movement_values = {field: _row_value(row, field) for field in MOVEMENT_COUNT_FIELDS}
                for collection, key in targets:
                    evaluation = collection.setdefault(
                        key,
                        {
                            "games": 0,
                            "completed": 0,
                            "completion_reasons": Counter(),
                            "failure_reasons": Counter(),
                            "random": 0.0,
                            "tier_games": Counter(),
                            "tier_wins": Counter(),
                            "tier_score": Counter(),
                            "tier_score_games": Counter(),
                            "actions": 0,
                            "loss_total": 0.0,
                            "loss_games": 0,
                            "expected": 0,
                            "movement_totals": Counter(),
                            "movement_games": Counter(),
                        },
                    )
                    evaluation["games"] += 1
                    evaluation["random"] += 1 / int(player_count)
                    completion_reason = row.get("completion_reason", "normal") or "normal"
                    evaluation["completion_reasons"][completion_reason] += 1
                    completed = row.get("completed", "true").strip().lower() != "false"
                    if completed:
                        evaluation["completed"] += 1
                        evaluation["actions"] += int(_number(row.get("action_count")) or 0)
                    else:
                        evaluation["failure_reasons"][completion_reason] += 1
                    evaluation_loss = _row_value(row, "latest_loss")
                    if evaluation_loss is not None:
                        evaluation["loss_total"] += evaluation_loss
                        evaluation["loss_games"] += 1
                    evaluation["expected"] = max(
                        evaluation["expected"],
                        int(_number(row.get("evaluation_suite_size")) or 0),
                    )
                    for field, value in movement_values.items():
                        if value is not None:
                            evaluation["movement_totals"][field] += value
                            evaluation["movement_games"][field] += 1
                    for tier, score in zip(assigned_tiers, final_scores):
                        evaluation["tier_games"][tier] += 1
                        if isinstance(score, (int, float)):
                            evaluation["tier_score"][tier] += score
                            evaluation["tier_score_games"][tier] += 1
                    evaluation_win_share = 1 / len(winner_tiers) if winner_tiers else 0
                    for tier in winner_tiers:
                        evaluation["tier_wins"][tier] += evaluation_win_share
            for tier in assigned_tiers:
                counts["tier_games"][str(tier)] += 1
                counts["tier_games_by_player_count"][(player_count, str(tier))] += 1
            win_share = 1 / len(winner_tiers) if winner_tiers else 0
            for tier in winner_tiers:
                counts["tier_wins"][str(tier)] += win_share
                counts["tier_wins_by_player_count"][(player_count, str(tier))] += win_share
            if len(winner_tiers) > 1:
                counts["ties_by_player_count"][player_count] += 1
            winning_tier_ids = {str(tier) for tier in winner_tiers}
            for tier, score in zip(assigned_tiers, final_scores):
                if isinstance(score, (int, float)):
                    tier = str(tier)
                    key = (player_count, tier)
                    counts["tier_score_total_by_player_count"][key] += score
                    counts["tier_score_games_by_player_count"][key] += 1
                    result = "winning" if tier in winning_tier_ids else "losing"
                    counts[f"{result}_score_total"][key] += score
                    counts[f"{result}_score_count"][key] += 1
                    minimums = counts[f"{result}_score_min"]
                    maximums = counts[f"{result}_score_max"]
                    minimums[key] = score if key not in minimums else min(minimums[key], score)
                    maximums[key] = score if key not in maximums else max(maximums[key], score)
            for column, _title in CHART_COLUMNS:
                value = _row_value(row, column)
                if value is not None:
                    series[column, run_type].add(chart_game_number, value)
    if counts["evaluation_versions"]:
        latest_version = max(counts["evaluation_versions"])
        latest = counts["evaluation_versions"][latest_version]
        counts["current_evaluation_suite_version"] = latest_version
        counts["evaluation_batches"] = latest["batches"]
        counts["evaluation_map_batches"] = latest["map_batches"]
        counts["evaluation_player_batches"] = latest["player_batches"]
        counts["evaluation_map_player_batches"] = latest["map_player_batches"]
    return row_count, series, counts


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _statistics(points):
    values = [value for _game, value in points]
    p95 = _percentile(values, 0.95)
    comparison_size = max(1, len(values) // 10)
    starting_median = statistics.median(values[:comparison_size])
    latest_median = statistics.median(values[-comparison_size:])
    change = (latest_median - starting_median) / starting_median * 100 if starting_median else 0.0
    entries = (
        ("Median", statistics.median(values)),
        ("Average", statistics.fmean(values)),
        ("95% of losses below", p95),
        ("Starting 10% median", starting_median),
        ("Latest 10% median", latest_median),
        ("Beginning-to-latest change (%)", change),
    )
    return "".join(
        f"<div><strong>{html.escape(label)}</strong><span>{value:,.1f}%</span></div>"
        if label.endswith("(%)")
        else f"<div><strong>{html.escape(label)}</strong><span>{value:,.0f}</span></div>"
        for label, value in entries
    )


def _chart(chart_id, title, training, evaluation, median=()):
    if not training and not evaluation:
        return ""
    payload = html.escape(
        json.dumps(
            {"training": training, "evaluation": evaluation, "median": median},
            separators=(",", ":"),
        ),
        quote=True,
    )
    median_legend = (
        '<span class="median-key">Rolling median line (latest 50 games)</span>' if median else ""
    )
    return f"""
    <section class="card chart" data-series="{payload}">
      <div class="chart-heading"><h2>{html.escape(title)}</h2><div>
        <button type="button" data-action="scale">Scale: log</button>
        <button type="button" data-action="focus">Range: focus 95%</button>
        <button type="button" data-action="reset">Reset zoom</button>
      </div></div>
      <div class="statistics">{_statistics(training or evaluation)}</div>
      <canvas id="{html.escape(chart_id)}" aria-label="{html.escape(title)}"></canvas>
      <p class="chart-legend"><span class="loss-key">Loss line and points</span>
        <span class="range-key">Grouped minimum-to-maximum range</span>
        {median_legend}
        <span class="trend-key">Overall trend line</span></p>
      <div class="tooltip" hidden></div>
      <p class="hint">At most 750 groups are drawn. Hover for each group's average and range;
        zoom in to reveal finer detail down to individual games.</p>
    </section>"""


def _summary_table(title: str, values: Counter):
    rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{count:,}</td></tr>"
        for name, count in values.most_common()
    )
    return f"<section class='summary'><h2>{html.escape(title)}</h2><table>{rows}</table></section>"


def _tier_chart(games: Counter, wins: Counter):
    tiers = sorted(
        games, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    )
    rows = []
    for tier in tiers:
        played = games[tier]
        won = wins[tier]
        rate = won / played * 100 if played else 0
        rows.append(
            f"<tr><td>Tier {html.escape(tier)}</td><td>{won:,.1f} / {played:,}</td>"
            f"<td>{rate:.1f}%</td><td><div class='bar' style='width:{rate:.1f}%'></div></td></tr>"
        )
    return (
        "<section class='card'><h2>Winner percentage by tier</h2>"
        "<table><tr><th>Tier</th><th>Win share / games</th><th>Win rate</th><th></th></tr>"
        + "".join(rows)
        + "</table></section>"
    )


def _svg_grouped_bar_chart(
    values, *, maximum, suffix="", baselines=None, group_by_player_count=False
):
    """Render one narrow bar for every tier/player-count combination."""
    width, height = 1000, 320
    left, right, top, bottom = 52, 986, 18, 240
    plot_height = bottom - top
    colors = {"3": "#2563eb", "4": "#7c3aed", "5": "#16a34a"}

    def y_position(value):
        return bottom - min(max(value, 0), maximum) / maximum * plot_height

    grid = []
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        value = maximum * fraction
        y_value = y_position(value)
        grid.append(
            f'<line x1="{left}" y1="{y_value:.2f}" x2="{right}" y2="{y_value:.2f}" '
            f'class="svg-grid"/><text x="{left - 6}" y="{y_value + 4:.2f}" '
            f'text-anchor="end">{value:.0f}{suffix}</text>'
        )

    groups = []
    for tier, player_count, _value, _detail in values:
        group = player_count if group_by_player_count else tier
        if group not in groups:
            groups.append(group)
    gap_units = 0.55
    total_units = len(values) + max(len(groups) - 1, 0) * gap_units
    spacing = (right - left) / max(total_units, 1)
    bar_width = min(34, spacing * 0.62)
    bars = []
    group_centers = {group: [] for group in groups}
    group_positions = {group: 0 for group in groups}
    group_sizes = {
        group: sum(
            (player_count if group_by_player_count else tier) == group
            for tier, player_count, _value, _detail in values
        )
        for group in groups
    }
    for index, (tier, player_count, value, detail) in enumerate(values):
        group = player_count if group_by_player_count else tier
        group_index = groups.index(group)
        if group_by_player_count:
            group_width = (right - left) / len(groups)
            cluster_center = left + group_width * (group_index + 0.5)
            cluster_spacing = min(46, group_width * 0.7 / group_sizes[group])
            position = group_positions[group]
            center = cluster_center + (position - (group_sizes[group] - 1) / 2) * cluster_spacing
            group_positions[group] += 1
            bar_width = min(34, cluster_spacing * 0.72)
        else:
            center = left + spacing * (index + group_index * gap_units + 0.5)
        group_centers[group].append(center)
        if value is not None:
            y_value = y_position(value)
            bars.append(
                f"<g><title>{html.escape(detail)}</title><rect "
                f'x="{center - bar_width / 2:.2f}" y="{y_value:.2f}" '
                f'width="{bar_width:.2f}" height="{bottom - y_value:.2f}" '
                f'fill="{colors.get(player_count, "#64748b")}" rx="3"/>'
                f'<text x="{center:.2f}" y="{max(top + 11, y_value - 5):.2f}" '
                f'text-anchor="middle">{value:.1f}{suffix}</text></g>'
            )
        else:
            bars.append(
                f'<g><title>{html.escape(detail)}</title><text x="{center:.2f}" y="232" '
                'text-anchor="middle" class="no-data">N/A</text></g>'
            )
        if baselines is not None:
            baseline_y = y_position(baselines[player_count])
            bars.append(
                f'<line x1="{center - bar_width * 0.65:.2f}" y1="{baseline_y:.2f}" '
                f'x2="{center + bar_width * 0.65:.2f}" y2="{baseline_y:.2f}" '
                'class="svg-baseline"/>'
            )
        bars.append(
            f'<text x="{center:.2f}" y="263" text-anchor="middle" class="tier-label">'
            f"T{html.escape(tier)}</text>"
            + (
                ""
                if group_by_player_count
                else f'<text x="{center:.2f}" y="279" text-anchor="middle">'
                f"{html.escape(player_count)}p</text>"
            )
        )

    if group_by_player_count:
        for player_count, centers in group_centers.items():
            center = sum(centers) / len(centers)
            bars.append(
                f'<text x="{center:.2f}" y="286" text-anchor="middle" '
                f'class="chart-group-label">{html.escape(player_count)} players</text>'
            )

    return (
        f'<svg class="tier-svg" viewBox="0 0 {width} {height}" role="img">'
        + "".join(grid)
        + "".join(bars)
        + "</svg>"
    )


def _score_result_summary(player_count, tiers, counts):
    rows = []
    for tier in tiers:
        key = (player_count, tier)
        cells = []
        for result in ("winning", "losing"):
            sample_count = counts[f"{result}_score_count"][key]
            if sample_count:
                average = counts[f"{result}_score_total"][key] / sample_count
                minimum = counts[f"{result}_score_min"][key]
                maximum = counts[f"{result}_score_max"][key]
                cells.append(
                    f"<strong>{average:.1f} average</strong><br>"
                    f"<span>{minimum:g}&ndash;{maximum:g} range &middot; "
                    f"{sample_count} game{'s' if sample_count != 1 else ''}</span>"
                )
            else:
                cells.append("&mdash;")
        rows.append(
            f"<tr><th>Tier {html.escape(tier)}</th>"
            + "".join(f"<td>{value}</td>" for value in cells)
            + "</tr>"
        )
    return (
        '<div class="score-summary"><table><thead><tr><th>Tier</th>'
        "<th>When this tier wins</th><th>When this tier loses</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _tier_player_count_charts(counts):
    games = counts["tier_games_by_player_count"]
    player_counts = ["3", "4", "5"] if games else []
    tiers = sorted(
        {tier for _count, tier in games},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    baselines = {
        player_count: 100 / int(player_count) if player_count.isdigit() else 0
        for player_count in player_counts
    }
    win_values = []
    score_values = []
    for player_count in player_counts:
        for tier in tiers:
            key = (player_count, tier)
            played = games[key]
            won = counts["tier_wins_by_player_count"][key]
            rate = won / played * 100 if played else None
            if played:
                win_values.append(
                    (
                        tier,
                        player_count,
                        rate,
                        f"Tier {tier}, {player_count} players: "
                        f"{won:.1f} win share / {played} games ({rate:.1f}%)",
                    )
                )
    for player_count in player_counts:
        for tier in tiers:
            key = (player_count, tier)
            score_games = counts["tier_score_games_by_player_count"][key]
            score_total = counts["tier_score_total_by_player_count"][key]
            score = score_total / score_games if score_games else None
            if score_games:
                score_values.append(
                    (
                        tier,
                        player_count,
                        score,
                        f"Tier {tier}, {player_count} players: "
                        f"average score {score:.2f} / {score_games} games",
                    )
                )
    if not win_values:
        return ""
    highest_win_rate = max(
        rate for _tier, _players, rate, _detail in win_values if rate is not None
    )
    win_maximum = _chart_ceiling(highest_win_rate, 10, minimum=40, maximum=100)
    highest_score = max(
        (score for _tier, _players, score, _detail in score_values if score is not None),
        default=1,
    )
    score_maximum = _chart_ceiling(highest_score * 1.15, 5, minimum=5)
    shared = ", ".join(
        f"{player_count}p: {counts['ties_by_player_count'][player_count]}"
        for player_count in player_counts
    )
    summaries = "".join(
        f"<div><h3>{html.escape(player_count)} players</h3>"
        f"{_score_result_summary(player_count, tiers, counts)}</div>"
        for player_count in player_counts
    )
    return (
        '<section class="card tier-performance"><h2>Tier performance by player count</h2>'
        "<p>Blue bars are 3-player games, purple bars are 4-player games, and green bars "
        "are 5-player games. Orange ticks show the matching random-win baseline "
        "(33.3%, 25%, or 20%). Only tiers assigned at that player count are shown. "
        "Rulebook tie-breakers are applied; remaining shared victories split one chart win. "
        f'Shared victories: {html.escape(shared)}.</p><div class="performance-grid"><div>'
        f"<h3>Win percentage</h3>{_svg_grouped_bar_chart(win_values, maximum=win_maximum, suffix='%', baselines=baselines, group_by_player_count=True)}"
        "</div><div><h3>Average final score</h3>"
        f"{_svg_grouped_bar_chart(score_values, maximum=score_maximum, group_by_player_count=True)}</div></div>"
        '<div class="score-results"><h3>Scores when each tier wins or loses</h3>'
        "<p>Each result shows the average score, the lowest-to-highest range, and "
        f'the number of games.</p><div class="score-summary-grid">{summaries}</div></div>'
        "</section>"
    )


def _evaluation_chart(
    batches,
    map_batches,
    player_batches,
    map_player_batches,
    suite_version=1,
):
    if not batches:
        return ""
    width, height = 920, 300
    left, right, top, bottom = 55, 20, 20, 40
    tier_colors = {
        1: "#2563eb",
        2: "#16a34a",
        3: "#7c3aed",
        4: "#db2777",
        5: "#64748b",
    }
    datasets = {("all", "all"): batches}
    for (map_num, batch), entry in map_batches.items():
        datasets.setdefault((str(map_num), "all"), {})[batch] = entry
    for (players, batch), entry in player_batches.items():
        datasets.setdefault(("all", str(players)), {})[batch] = entry
    for (map_num, players, batch), entry in map_player_batches.items():
        datasets.setdefault((str(map_num), str(players)), {})[batch] = entry

    def line_chart(
        title,
        explanation,
        ordered,
        series,
        suffix,
        maximum=None,
        baseline=None,
        focus_range=False,
        point_details=None,
        tick_step=None,
    ):
        available = [value for values in series.values() for value in values if value is not None]
        scale_values = available + ([*baseline] if baseline else [])
        axis_minimum = 0.0
        if focus_range and scale_values:
            low = min(scale_values)
            high = max(scale_values)
            padding = max((high - low) * 0.15, high * 0.05, 1.0)
            axis_minimum = max(0.0, low - padding)
            maximum = min(maximum, high + padding) if maximum is not None else high + padding
        maximum = maximum or ((max(scale_values) * 1.1) if scale_values else 1)
        maximum = max(maximum, 1)
        if maximum <= axis_minimum:
            maximum = axis_minimum + 1
        if tick_step is None:
            rough_step = (maximum - axis_minimum) / 5
            magnitude = 10 ** math.floor(math.log10(max(rough_step, 1e-9)))
            normalized = rough_step / magnitude
            factor = (
                1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
            )
            tick_step = factor * magnitude
        axis_minimum = math.floor(axis_minimum / tick_step) * tick_step
        maximum = math.ceil(maximum / tick_step) * tick_step
        x_span = max(len(ordered) - 1, 1)

        def point(index, value):
            return (
                left + index / x_span * (width - left - right),
                top + (maximum - value) / (maximum - axis_minimum) * (height - top - bottom),
            )

        grid = []
        tick_values = []
        value = axis_minimum
        while value <= maximum + tick_step / 2:
            tick_values.append(value)
            value += tick_step
        for value in tick_values:
            _x, y = point(0, value)
            grid.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                f'class="svg-grid"/><text x="5" y="{y + 4:.1f}">{value:,.0f}</text>'
            )
        x_tick_indices = sorted({round(tick * (len(ordered) - 1) / 5) for tick in range(6)})
        for index in x_tick_indices:
            x, _y = point(index, 0)
            grid.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom}" '
                f'class="svg-x-grid"/><text x="{x:.1f}" y="{height - 10}" '
                f'text-anchor="middle">batch {ordered[index][0]}</text>'
            )
        lines = []
        legend = []
        named_colors = ("#2563eb", "#dc2626", "#16a34a", "#7c3aed")
        for series_index, (tier, values) in enumerate(series.items()):
            color = tier_colors.get(tier, named_colors[series_index % len(named_colors)])
            series_label = f"Tier {tier}" if isinstance(tier, int) else str(tier)
            coordinates = [
                (index, *point(index, value), value)
                for index, value in enumerate(values)
                if value is not None
            ]
            if not coordinates:
                continue
            path = " ".join(
                f"{'M' if path_index == 0 else 'L'}{x:.1f},{y:.1f}"
                for path_index, (_index, x, y, _value) in enumerate(coordinates)
            )
            circles = "".join(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}">'
                f"<title>{series_label}, batch {ordered[index][0]}: "
                f"{value:.1f}{suffix}"
                f"{html.escape(point_details[index]) if point_details else ''}</title></circle>"
                for index, x, y, value in coordinates
            )
            lines.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>{circles}'
            )
            legend.append(f'<span style="color:{color}">{series_label}</span>')
        baseline_path = ""
        if baseline:
            coordinates = [point(index, value) for index, value in enumerate(baseline)]
            path = " ".join(
                f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}"
                for index, (x, y) in enumerate(coordinates)
            )
            baseline_path = (
                f'<path d="{path}" fill="none" stroke="#f97316" stroke-width="2" '
                'stroke-dasharray="8 5"/>'
            )
            legend.append('<span style="color:#c2410c">Random-win baseline</span>')
        return (
            f'<div class="evaluation-chart"><h3>{title}</h3><p>{explanation}</p>'
            f'<p class="chart-legend">{"".join(legend)}</p>'
            f'<svg class="tier-svg" viewBox="0 0 {width} {height}">{"".join(grid)}'
            f"{''.join(lines)}{baseline_path}"
            "</svg></div>"
        )

    loss_ordered = sorted(batches.items())
    loss_values = [
        entry["loss_total"] / entry["loss_games"] if entry["loss_games"] else None
        for _batch, entry in loss_ordered
    ]
    rolling_loss_values = []
    for index, value in enumerate(loss_values):
        window = [
            candidate
            for candidate in loss_values[max(0, index - 4) : index + 1]
            if candidate is not None
        ]
        rolling_loss_values.append(sum(window) / len(window) if value is not None else None)
    loss_details = [
        f"; {entry['completed']}/{entry['expected'] or entry['games']} boards completed"
        for _batch, entry in loss_ordered
    ]
    evaluation_loss_chart = line_chart(
        "Evaluation loss by batch",
        "Lower is better. Each point averages the completed fixed evaluation boards; "
        "hover to see how many boards completed.",
        loss_ordered,
        {
            "Evaluation loss": loss_values,
            "Five-batch average": rolling_loss_values,
        },
        "",
        focus_range=True,
        point_details=loss_details,
    )

    panels = []
    for (map_value, player_value), current_batches in datasets.items():
        ordered = sorted(current_batches.items())
        tiers = sorted({tier for _batch, entry in ordered for tier in entry["tier_games"]})

        def movement_ratio(entry, numerator, denominator):
            numerator_total = entry["movement_totals"][numerator]
            denominator_total = entry["movement_totals"][denominator]
            if not denominator_total:
                return None
            return numerator_total / denominator_total

        def movement_average(entry, field):
            games = entry["movement_games"][field]
            if not games:
                return None
            return entry["movement_totals"][field] / games

        def movement_percentage(entry, numerator, denominator):
            ratio = movement_ratio(entry, numerator, denominator)
            return ratio * 100 if ratio is not None else None

        def display_metric(value, *, percentage=False):
            if value is None:
                return "&mdash;"
            return f"{value * 100:.1f}%" if percentage else f"{value:.2f}"

        def rolling(values, window=10):
            result = []
            for index in range(len(values)):
                current = [
                    value
                    for value in values[max(0, index - window + 1) : index + 1]
                    if value is not None
                ]
                result.append(statistics.fmean(current) if current else None)
            return result

        win_series = {
            tier: rolling(
                [
                    entry["tier_wins"][tier] / entry["tier_games"][tier] * 100
                    if entry["tier_games"][tier]
                    else None
                    for _batch, entry in ordered
                ]
            )
            for tier in tiers
        }
        score_series = {
            tier: rolling(
                [
                    entry["tier_score"][tier] / entry["tier_score_games"][tier]
                    if entry["tier_score_games"][tier]
                    else None
                    for _batch, entry in ordered
                ]
            )
            for tier in tiers
        }
        action_series = {
            "Game length": [
                entry["actions"] / entry["completed"] if entry["completed"] else None
                for _batch, entry in ordered
            ]
        }
        move_percentage_series = {
            "Move %": [
                movement_percentage(entry, "move_action_count", "spent_action_count")
                for _batch, entry in ordered
            ]
        }
        pointless_move_series = {
            "Pointless Moves/game": [
                movement_average(entry, "pointless_move_workflows") for _batch, entry in ordered
            ]
        }
        repeated_move_series = {
            "Repeated-Move penalties/game": [
                movement_average(entry, "repeated_move_penalties") for _batch, entry in ordered
            ]
        }
        all_move_turn_series = {
            "All-Move-turn penalties/game": [
                movement_average(entry, "all_move_turn_penalties") for _batch, entry in ordered
            ]
        }
        move_claim_series = {
            "Move to Claim conversion": [
                movement_percentage(
                    entry,
                    "move_claim_conversions",
                    "moves_creating_claimable_route",
                )
                for _batch, entry in ordered
            ]
        }
        baseline = rolling([entry["random"] / entry["games"] * 100 for _batch, entry in ordered])
        latest = ordered[-1][1]
        tier_one_games = latest["tier_games"][1]
        tier_one_scores = latest["tier_score_games"][1]
        tier_one_win = latest["tier_wins"][1] / tier_one_games * 100 if tier_one_games else 0
        tier_one_score = latest["tier_score"][1] / tier_one_scores if tier_one_scores else 0
        average_actions = latest["actions"] / latest["completed"] if latest["completed"] else 0
        latest_move_ratio = movement_ratio(latest, "move_action_count", "spent_action_count")
        latest_pointless_moves = movement_average(latest, "pointless_move_workflows")
        latest_move_claim_rate = movement_ratio(
            latest,
            "move_claim_conversions",
            "moves_creating_claimable_route",
        )
        failures = [f"{reason}: {count}" for reason, count in latest["failure_reasons"].items()]
        warning = (
            '<p class="evaluation-warning"><strong>Evaluation failures:</strong> '
            + html.escape(", ".join(failures))
            + "</p>"
            if failures
            else '<p class="evaluation-success">All latest evaluation games completed normally.</p>'
        )
        panels.append(
            f'<div data-evaluation-panel data-map="{map_value}" data-players="{player_value}" hidden>'
            '<div class="statistics">'
            f"<div><strong>Completed</strong><span>{latest['completed']}/{latest['games']}</span></div>"
            f"<div><strong>Tier 1 win rate</strong><span>{tier_one_win:.1f}%</span></div>"
            f"<div><strong>Tier 1 average score</strong><span>{tier_one_score:.1f}</span></div>"
            f"<div><strong>Average game length</strong><span>{average_actions:.0f} actions</span></div>"
            f"<div><strong>Move %</strong><span>{display_metric(latest_move_ratio, percentage=True)}</span></div>"
            f"<div><strong>Pointless Moves/game</strong><span>{display_metric(latest_pointless_moves)}</span></div>"
            f"<div><strong>Move &rarr; Claim rate</strong><span>{display_metric(latest_move_claim_rate, percentage=True)}</span></div>"
            f"</div>{warning}"
            + line_chart(
                "Win rate by tier",
                "Higher is better. Each colored line is a rolling 10-batch policy-tier rate.",
                ordered,
                win_series,
                "%",
                maximum=100,
                baseline=baseline,
            )
            + line_chart(
                "Average final score by tier",
                "Higher is generally better. Lines show rolling 10-batch averages across "
                "the same fixed positions.",
                ordered,
                score_series,
                " points",
                focus_range=True,
                tick_step=5,
            )
            + line_chart(
                "Average completed-game length",
                "Lower is generally better, provided games still finish normally.",
                ordered,
                action_series,
                " actions",
            )
            + line_chart(
                "Move % of paid actions",
                "Lower generally indicates less reliance on Move, but Move remains a legal "
                "and sometimes necessary action.",
                ordered,
                move_percentage_series,
                "%",
                maximum=100,
            )
            + line_chart(
                "Pointless Move workflows per game",
                "Lower is better. Counts movement workflows receiving the existing "
                "pointless-movement target.",
                ordered,
                pointless_move_series,
                " workflows/game",
            )
            + line_chart(
                "Repeated-Move penalties per game",
                "Lower is better. Counts applications of the existing consecutive-Move penalty.",
                ordered,
                repeated_move_series,
                " penalties/game",
            )
            + line_chart(
                "All-Move-turn penalties per game",
                "Lower is better. Counts turns where every paid action was a normal Move.",
                ordered,
                all_move_turn_series,
                " penalties/game",
            )
            + line_chart(
                "Move &rarr; Claim conversion rate",
                "Higher means more route-creating normal Moves were followed by an immediate "
                "paid claim. This is diagnostic, not a requirement for every Move.",
                ordered,
                move_claim_series,
                "%",
                maximum=100,
            )
            + "</div>"
        )
    return f"""
    <section class="card evaluation-performance"><div class="chart-heading">
      <h2>Evaluation results — suite version {suite_version}</h2>
      <label>Board: <select data-evaluation-map><option value="all">All maps</option>
      <option value="1">Map 1</option><option value="2">Map 2</option>
      <option value="3">Map 3</option></select></label>
      <label>Players: <select data-evaluation-players><option value="all">All players</option>
      <option value="3">3 players</option><option value="4">4 players</option>
      <option value="5">5 players</option></select></label></div>
      {evaluation_loss_chart}
      {"".join(panels)}
      <p class="hint">Each point summarizes one batch of fixed evaluation positions.
        Use the filters to compare matching boards and player counts.</p>
    </section>"""


def build_dashboard(row_count, series, counts, source_path):
    charts = "".join(
        _chart(
            f"chart-{index}",
            title,
            series[column, "training"].points,
            (),
            series["rolling_median_50", "training"].points if column == "latest_loss" else (),
        )
        for index, (column, title) in enumerate(CHART_COLUMNS)
    )
    summaries = "".join(
        _summary_table(title, counts[key])
        for key, title in (
            ("completion_reason", "Completion results"),
            ("run_type", "Game types"),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Hansa training results</title><style>
body {{ margin: 0 auto; max-width: 1100px; padding: 24px; color: #e2e8f0;
font-family: system-ui, sans-serif; background: #0f172a; color-scheme:dark; }}
.card,.summary {{ background:#1e293b; margin:16px 0; padding:16px; border-radius:10px;
border:1px solid #334155; box-shadow:0 4px 14px #02061780; }} .summaries {{ display:grid;
grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
.summaries .summary {{ margin:0; }} h1,h2 {{ margin:0 0 12px; }} h2 {{ font-size:17px; }}
canvas {{ width:100%; display:block; }} .chart {{ position:relative; }}
.chart-heading {{ display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
button {{ padding:6px 10px; margin-left:5px; color:#e2e8f0; border:1px solid #475569;
border-radius:6px; background:#334155; cursor:pointer; }}
button:hover {{ background:#475569; }}
select {{ padding:6px 28px 6px 8px; color:#e2e8f0; border:1px solid #475569;
border-radius:6px; background:#334155; }}
.statistics {{ display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin-bottom:8px; }}
.statistics div {{ flex:0 1 max-content; min-width:88px; max-width:210px; background:#0f172a;
padding:7px 10px; border-radius:5px; }}
.statistics strong,.statistics span {{ display:block; }} .statistics strong {{ font-size:11px;
color:#94a3b8; }} .tooltip {{ position:absolute; pointer-events:none; background:#020617;
color:#f8fafc; border:1px solid #475569; padding:7px 9px; border-radius:5px;
font-size:12px; z-index:2; }}
.hint {{ margin:4px 0 0; color:#94a3b8; font-size:12px; }}
.chart-legend {{ margin:4px 0; font-size:12px; }} .chart-legend span {{ margin-right:16px; }}
.loss-key {{ color:#60a5fa; }} .range-key {{ color:#94a3b8; }}
.median-key {{ color:#4ade80; }} .trend-key {{ color:#c084fc; }}
table {{ width:100%; border-collapse:collapse; }} td {{ padding:4px; border-bottom:1px solid #334155; }}
td:last-child {{ text-align:right; }} .legend span {{ margin-right:18px; }}
.bar {{ height:14px; background:#22c55e; min-width:2px; }} th {{ text-align:left; }}
.performance-grid {{ display:grid; grid-template-columns:1fr; gap:24px; }}
.performance-grid h3 {{ margin:0 0 8px; font-size:14px; }}
.tier-svg {{ width:100%; height:auto; }} .tier-svg text {{ font:11px system-ui; fill:#cbd5e1; }}
.tier-svg .tier-label {{ font-weight:600; }} .svg-grid {{ stroke:#475569; stroke-width:1; }}
.svg-x-grid {{ stroke:#334155; stroke-width:1; }}
.svg-baseline {{ stroke:#f97316; stroke-width:2; stroke-dasharray:8 5; }}
.tier-legend {{ margin:2px 0 0; font-size:12px; }} .tier-legend span {{ margin-right:18px; }}
.win-rate-key {{ color:#60a5fa; }} .random-key {{ color:#fb923c; }}
.score-summary {{ overflow-x:auto; margin-top:18px; }} .score-summary th,
.score-summary td {{ text-align:center; white-space:nowrap; }}
.score-results {{ margin-top:24px; }} .score-results > h3 {{ margin-bottom:4px; }}
.score-results > p {{ margin-top:0; }} .score-summary-grid {{ display:grid;
grid-template-columns:1fr; gap:20px; margin-top:18px; }}
.score-summary-grid > div > h3 {{ margin:0 0 8px; }}
.score-summary-grid .score-summary {{ margin-top:0; overflow-x:auto; }}
.tier-svg .no-data {{ fill:#94a3b8; font-size:9px; }}
.evaluation-chart {{ margin-top:24px; padding-top:18px; border-top:1px solid #334155; }}
.evaluation-chart h3 {{ margin:0 0 4px; }}
.evaluation-warning {{ color:#fecaca; background:#450a0a; padding:10px; border-radius:6px; }}
.evaluation-success {{ color:#bbf7d0; background:#052e16; padding:10px; border-radius:6px; }}
</style></head><body>
<h1>Hansa training results</h1>
<p>{row_count:,} rows read from {html.escape(str(source_path))}</p>
<p class="legend"><span style="color:{COLORS["training"]}">Training</span>
<span style="color:{COLORS["evaluation"]}">Evaluation</span></p>
{_tier_chart(counts["tier_games"], counts["tier_wins"])}
{_tier_player_count_charts(counts)}
{
        _evaluation_chart(
            counts["evaluation_batches"],
            counts["evaluation_map_batches"],
            counts["evaluation_player_batches"],
            counts["evaluation_map_player_batches"],
            counts["current_evaluation_suite_version"],
        )
    }
{charts}<div class="summaries">{summaries}</div>
<script>{DASHBOARD_SCRIPT}</script></body></html>"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="results CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="HTML output path")
    parser.add_argument(
        "--max-points",
        type=int,
        default=750,
        help="maximum visible groups per line chart (default: 750)",
    )
    parser.add_argument("--open", action="store_true", help="open the finished chart in a browser")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_points < 10:
        raise SystemExit("--max-points must be at least 10")
    if not args.input.is_file():
        raise SystemExit(f"Results file not found: {args.input}")
    row_count, series, counts = read_results(args.input, args.max_points)
    dashboard = build_dashboard(row_count, series, counts, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dashboard, encoding="utf-8")
    print(f"Charted {row_count:,} rows: {args.output.resolve()}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
