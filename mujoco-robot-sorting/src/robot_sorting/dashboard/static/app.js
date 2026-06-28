const data = window.DASHBOARD_DATA;
const summary = data.summary;

document.getElementById("run-id").textContent = `Run: ${summary.run_id}`;
const slaBadge = document.getElementById("sla-badge");
slaBadge.textContent = summary.sla_passed ? "SLA PASS" : "SLA FAIL";
slaBadge.classList.add(summary.sla_passed ? "pass" : "fail");

const cardValues = [
  ["Total", summary.total_objects],
  ["Normal", summary.normal_count],
  ["Defect", summary.defect_count],
  ["Placed", summary.placed_count],
  ["Failed", summary.failed_count],
  ["Success Rate", `${(summary.success_rate * 100).toFixed(1)}%`],
  ["Avg Latency", `${summary.average_command_latency_seconds.toFixed(4)}s`],
  ["Max Latency", `${summary.max_command_latency_seconds.toFixed(4)}s`],
  ["Conveyor Stops", summary.conveyor_stop_count],
  ["Overhead Rotate", summary.overhead_rotate_count],
  ["Level Parallel", summary.level_parallel_count],
  ["Table Failures", summary.table_penetration_failure_count],
  ["Collision Failures", summary.collision_failure_count],
  ["No Bin Space", summary.no_free_space_inside_bin_count],
];

document.getElementById("cards").innerHTML = cardValues
  .map(([label, value]) => `<article class="card"><span>${label}</span><strong>${value}</strong></article>`)
  .join("");

document.getElementById("timeline").innerHTML = data.timeline
  .map((row) => `<div class="timeline-item"><span>${row.timestamp}s</span><strong>${row.state}</strong><em>${row.event}</em></div>`)
  .join("");

document.getElementById("objects").innerHTML = data.objects
  .map((row) => `
    <tr>
      <td>${row.object_id}</td>
      <td>${row.label}</td>
      <td>${row.status}</td>
      <td>${row.target_bin ?? ""}</td>
      <td>${row.transport_mode ?? ""}</td>
      <td>${row.command_latency_seconds === null ? "" : `${row.command_latency_seconds.toFixed(4)}s`}</td>
    </tr>`)
  .join("");

document.getElementById("events").innerHTML = data.events
  .map((row) => `
    <tr>
      <td>${row.timestamp}</td>
      <td>${row.state}</td>
      <td>${row.object_id ?? ""}</td>
      <td>${row.event}</td>
    </tr>`)
  .join("");
