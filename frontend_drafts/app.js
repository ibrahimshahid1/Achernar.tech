const fallbackStatus = {
  connected: true,
  daemon_id: "draft-offline-fixture",
  hardware_id: "achernar-local-daemon",
  camera: {
    name: "ASI2600MM Pro",
    connected: true,
    cooling: true,
    temperature: -10.1,
    exposing: true,
    streaming: true,
    gain: 100,
    exposure_s: 2,
    latest_frame: {
      frame_counter: 1842,
      width: 1304,
      height: 976,
      format: "Mjpeg",
    },
  },
  mount: {
    name: "AP 1100GTO",
    connected: true,
    tracking: true,
    ra: 5.5753,
    dec: 22.0144,
    parked: false,
    pier_side: "East",
    guide_rms: 0.48,
  },
  focuser: {
    name: "Optec Gemini",
    connected: true,
    position: 9900,
    moving: false,
    absolute: true,
    max_range: 50000,
    hfr: 3.61,
    temperature: 4.8,
  },
  filter_wheel: {
    name: "EFW 7 x 36 mm",
    connected: true,
    filter: "L",
    slot: 1,
    moving: false,
    offset: 0,
  },
};

let equipment = {};
const selected = new Set(readSelectedEquipment());

const equipmentGrid = document.querySelector("#equipmentGrid");
const dropZone = document.querySelector("#dropZone");
const startRunButton = document.querySelector("#startRunButton");
const backToDashboard = document.querySelector("#backToDashboard");
const selectedRunStrip = document.querySelector("#selectedRunStrip");
const deviceCount = document.querySelector("#deviceCount");

function connectionTone(connected) {
  return connected === false ? "warn" : "good";
}

function statusLabel(connected, activeLabel) {
  if (connected === false) return "Offline";
  return activeLabel;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return replacements[char];
  });
}

function readSelectedEquipment() {
  try {
    const stored = JSON.parse(sessionStorage.getItem("achernarDraftSelectedEquipment") || "[]");
    if (Array.isArray(stored) && stored.length > 0) return stored;
  } catch {
    // Keep the static prototype forgiving if storage is unavailable.
  }
  return ["daemon-camera", "daemon-mount"];
}

function saveSelectedEquipment() {
  try {
    sessionStorage.setItem("achernarDraftSelectedEquipment", JSON.stringify([...selected]));
  } catch {
    // The static draft should still work as a visual prototype without storage.
  }
}

async function loadDaemonStatus() {
  try {
    const response = await fetch("./daemon-status.sample.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`fixture ${response.status}`);
    return await response.json();
  } catch {
    return fallbackStatus;
  }
}

function cardsFromDaemonStatus(status) {
  const cards = {};
  const source = status.daemon_id ? "Django relay / daemon sample" : "Draft fixture";
  const site = status.hardware_id || "local daemon";

  if (status.camera) {
    const camera = status.camera;
    const frame = camera.latest_frame || {};
    const mode = camera.streaming ? "Streaming" : camera.exposing ? "Exposing" : camera.status || "Idle";
    cards["daemon-camera"] = {
      id: "daemon-camera",
      type: "camera",
      kind: "CAM",
      name: camera.name || "Local Camera",
      role: "Primary imager",
      source,
      site,
      status: statusLabel(camera.connected, mode),
      connected: camera.connected !== false,
      metrics: [
        ["Sensor", `${Number(camera.temperature || 0).toFixed(1)} C`],
        ["Exposure", `${Number(camera.exposure_s || 0).toFixed(1)} s`],
        ["Gain", camera.gain ?? "--"],
        ["Frame", frame.width && frame.height ? `${frame.width} x ${frame.height}` : "--"],
      ],
    };
  }

  if (status.mount) {
    const mount = status.mount;
    cards["daemon-mount"] = {
      id: "daemon-mount",
      type: "mount",
      kind: "MNT",
      name: mount.name || "Local Mount",
      role: "Tracking mount",
      source,
      site,
      status: statusLabel(mount.connected, mount.tracking ? "Tracking" : "Idle"),
      connected: mount.connected !== false,
      metrics: [
        ["RA", Number(mount.ra || 0).toFixed(2)],
        ["Dec", Number(mount.dec || 0).toFixed(2)],
        ["Guide", `${Number(mount.guide_rms || 0).toFixed(2)}\"`],
        ["Side", mount.pier_side || "--"],
      ],
    };
  }

  if (status.focuser) {
    const focuser = status.focuser;
    cards["daemon-focuser"] = {
      id: "daemon-focuser",
      type: "focuser",
      kind: "FOC",
      name: focuser.name || "Local Focuser",
      role: "Absolute focuser",
      source,
      site,
      status: statusLabel(focuser.connected, focuser.moving ? "Moving" : "Idle"),
      connected: focuser.connected !== false,
      metrics: [
        ["Position", focuser.position ?? "--"],
        ["HFR", focuser.hfr ?? "--"],
        ["Temp", focuser.temperature == null ? "--" : `${Number(focuser.temperature).toFixed(1)} C`],
        ["Range", focuser.max_range ?? "--"],
      ],
    };
  }

  if (status.filter_wheel) {
    const wheel = status.filter_wheel;
    cards["daemon-filter-wheel"] = {
      id: "daemon-filter-wheel",
      type: "filter-wheel",
      kind: "FLT",
      name: wheel.name || "Filter Wheel",
      role: "Filter wheel",
      source,
      site,
      status: statusLabel(wheel.connected, wheel.filter || "Ready"),
      connected: wheel.connected !== false,
      metrics: [
        ["Slot", wheel.slot ?? "--"],
        ["Filter", wheel.filter || "--"],
        ["Offset", wheel.offset ?? 0],
        ["Motion", wheel.moving ? "Moving" : "Stopped"],
      ],
    };
  }

  return cards;
}

function renderEquipmentGrid() {
  if (!equipmentGrid) return;
  const cards = Object.values(equipment);
  equipmentGrid.innerHTML = cards.map(renderEquipmentCard).join("");
  attachEquipmentCardHandlers();
  syncCardState();
}

function renderEquipmentCard(item) {
  const metrics = item.metrics
    .map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`)
    .join("");
  return `
    <article class="equipment-card" draggable="true" data-equipment-id="${escapeHtml(item.id)}" data-kind="${escapeHtml(item.type)}">
      <div class="card-topline">
        <span class="kind-token">${escapeHtml(item.kind)}</span>
        <span class="status-token ${connectionTone(item.connected)}">${escapeHtml(item.status)}</span>
      </div>
      <h3>${escapeHtml(item.name)}</h3>
      <p class="card-subtitle">${escapeHtml(item.role)} / ${escapeHtml(item.source)}</p>
      <dl class="metric-grid">${metrics}</dl>
    </article>
  `;
}

function attachEquipmentCardHandlers() {
  document.querySelectorAll(".equipment-card").forEach((card) => {
    card.addEventListener("dragstart", (event) => {
      card.classList.add("dragging");
      event.dataTransfer.setData("text/plain", card.dataset.equipmentId);
    });

    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
    });

    card.addEventListener("click", () => {
      selected.add(card.dataset.equipmentId);
      saveSelectedEquipment();
      renderSelectedEquipment();
      syncCardState();
    });
  });
}

function pruneSelection() {
  [...selected].forEach((id) => {
    if (!equipment[id]) selected.delete(id);
  });

  if (selected.size === 0) {
    ["daemon-camera", "daemon-mount"].forEach((id) => {
      if (equipment[id]) selected.add(id);
    });
  }

  saveSelectedEquipment();
}

function renderSelectedEquipment() {
  if (!dropZone) return;
  dropZone.innerHTML = "";

  selected.forEach((id) => {
    const item = equipment[id];
    if (!item) return;

    const row = document.createElement("div");
    row.className = "selected-item";
    row.dataset.equipmentId = item.id;
    row.innerHTML = `
      <span class="kind-token">${escapeHtml(item.kind)}</span>
      <div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.role)}</small></div>
      <button type="button" class="icon-button remove-item" aria-label="Remove ${escapeHtml(item.name)}">x</button>
    `;
    row.querySelector(".remove-item").addEventListener("click", () => {
      selected.delete(item.id);
      saveSelectedEquipment();
      renderSelectedEquipment();
      syncCardState();
    });
    dropZone.appendChild(row);
  });
}

function syncCardState() {
  document.querySelectorAll(".equipment-card").forEach((card) => {
    card.classList.toggle("selected", selected.has(card.dataset.equipmentId));
  });
}

function renderSelectedRunStrip() {
  if (!selectedRunStrip) return;
  selectedRunStrip.innerHTML = "";

  selected.forEach((id) => {
    const item = equipment[id];
    if (!item) return;

    const chip = document.createElement("div");
    chip.className = "run-equipment-chip";
    chip.innerHTML = `
      <span class="kind-token">${escapeHtml(item.kind)}</span>
      <div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.role)}</small></div>
    `;
    selectedRunStrip.appendChild(chip);
  });
}

function updateSummary() {
  if (deviceCount) {
    deviceCount.textContent = `${Object.values(equipment).filter((item) => item.connected).length} devices online`;
  }
}

if (dropZone) {
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("is-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("is-over");
  });

  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-over");
    const id = event.dataTransfer.getData("text/plain");
    if (equipment[id]) {
      selected.add(id);
      saveSelectedEquipment();
      renderSelectedEquipment();
      syncCardState();
    }
  });
}

function showRunRoom() {
  saveSelectedEquipment();
  window.location.href = "./run.html";
}

function showDashboard() {
  window.location.href = "./index.html";
}

if (startRunButton) {
  startRunButton.addEventListener("click", showRunRoom);
}

if (backToDashboard) {
  backToDashboard.addEventListener("click", showDashboard);
}

function activatePanel(panelId) {
  document.querySelectorAll(".tab-button").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.panel === panelId);
  });

  document.querySelectorAll(".control-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === panelId);
  });

  document.querySelectorAll(".mini-window").forEach((windowEl) => {
    windowEl.classList.toggle("active", windowEl.dataset.panelShortcut === panelId);
  });
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => activatePanel(button.dataset.panel));
});

document.querySelectorAll(".mini-window").forEach((windowEl) => {
  windowEl.addEventListener("click", () => activatePanel(windowEl.dataset.panelShortcut));
});

async function init() {
  equipment = cardsFromDaemonStatus(await loadDaemonStatus());
  pruneSelection();
  renderEquipmentGrid();
  renderSelectedEquipment();
  renderSelectedRunStrip();
  updateSummary();
}

init();
