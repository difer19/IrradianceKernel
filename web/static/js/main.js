const modelASelect = document.getElementById("model-a");
const modelBSelect = document.getElementById("model-b");
const compareButton = document.getElementById("compare-button");
const statusElement = document.getElementById("status");
let maps = { a: null, b: null };
let modelCatalog = [];

const palette = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626", "#0891b2"];
const metricDefinitions = [
  ["Accuracy", "accuracy"],
  ["F1 macro", "f1_macro"],
  ["AUC OVR", "auc_ovr"],
  ["MCC", "mcc"],
];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(message, isError = false) {
  statusElement.textContent = message;
  statusElement.className = isError ? "status error" : "status";
}

function setLoading(isLoading) {
  compareButton.classList.toggle("is-loading", isLoading);
  compareButton.dataset.loading = isLoading ? "true" : "false";
  compareButton.disabled = isLoading || modelCatalog.length < 2;
}

function metric(value) {
  return value === null || value === undefined ? "No definida" : Number(value).toFixed(4);
}

function compactConfiguration(configuration) {
  const labels = {
    standard: "Standard",
    minmax: "MinMax",
    normalizer: "Norm",
    uniform: "Uniforme",
    kmeans: "KMeans",
    dbscan: "DBSCAN",
    agglomerative: "Jerárquico",
    pca: "PCA",
    lda: "LDA",
    ksvc: "KSVC",
    kann: "KANNC",
    kridge: "KRidge",
    linear: "Lineal",
    poly: "Polinomial",
    rbf: "RBF",
    hyperbolic: "Hiperbólico",
    triangular: "Triangular",
    radial_basic: "Radial básico",
    rational_quadratic: "Racional cuadrático",
    canberra: "Canberra",
    truncated: "Truncado",
  };
  return configuration.split("__").map((part) => labels[part] || part.replaceAll("_", " ")).join(" · ");
}

function modelName(model) {
  return `${model.satellite.toUpperCase()} · ${compactConfiguration(model.configuration)}`;
}

async function loadModels() {
  const response = await fetch("/api/models");
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "No se pudieron cargar los modelos.");
  modelCatalog = payload.models;
  modelASelect.innerHTML = "";
  modelBSelect.innerHTML = "";
  modelCatalog.forEach((model, index) => {
    const optionA = new Option(modelName(model), model.id);
    const optionB = new Option(modelName(model), model.id);
    optionA.title = modelName(model);
    optionB.title = modelName(model);
    modelASelect.add(optionA);
    modelBSelect.add(optionB);
    if (index === 0) optionA.selected = true;
    if (index === 1) optionB.selected = true;
  });
  if (modelCatalog.length < 2) {
    setLoading(false);
    throw new Error("Se necesitan al menos dos modelos en /models.");
  }
  setLoading(false);
  await compareModels();
}

function renderMap(model, key) {
  if (maps[key]) maps[key].remove();
  const map = L.map(`map-${key}`);
  maps[key] = map;

  // Capas base cartográficas reales. Esri suele mostrar mejor las carreteras,
  // poblaciones y límites que una capa vacía cuando se visualiza a escala regional.
  const streetMap = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles © Esri — Source: Esri, HERE, Garmin, USGS, NGA",
    }
  );
  const satelliteMap = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
    }
  );
  const openStreetMap = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap",
  });
  streetMap.addTo(map);
  L.control.layers({
    "Mapa de calles (Esri)": streetMap,
    "Imagen satelital (Esri)": satelliteMap,
    "OpenStreetMap": openStreetMap,
  }, null, { collapsed: false }).addTo(map);

  const bounds = [];
  model.points.forEach((point) => {
    const position = [point.lat, point.lon];
    bounds.push(position);
    const color = palette[point.predicted_class % palette.length];
    const marker = L.circleMarker(position, {
      radius: 5,
      color,
      fillColor: color,
      fillOpacity: 0.78,
      weight: 1,
    }).bindPopup(
      `<strong>${escapeHtml(point.class_label)}</strong><br>`
      + `Irradiancia registrada: ${point.irradiance}<br>`
      + `EPSG:3857: (${point.x_epsg3857}, ${point.y_epsg3857})`
    ).addTo(map);
    marker.on("click", (event) => {
      L.DomEvent.stopPropagation(event);
      queryPoint(model.satellite, point.lat, point.lon);
    });
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [18, 18] });
  else map.setView([0, 0], 2);
  map.on("click", (event) => queryPoint(model.satellite, event.latlng.lat, event.latlng.lng));

  document.getElementById(`map-${key}-title`).textContent = modelName(model);
  document.getElementById(`map-${key}-subtitle`).textContent =
    `${model.point_count} observaciones · ${model.classes.length} clases`;
  renderLegend(model, key);
}

function renderLegend(model, key) {
  const legend = document.getElementById(`legend-${key}`);
  legend.innerHTML = model.classes.map((item) => {
    const color = palette[item.id % palette.length];
    return `<span class="legend-item"><i class="legend-dot" style="background:${color}"></i>${escapeHtml(`Clase ${item.id}`)}</span>`;
  }).join("");
}

function renderMetrics(modelA, modelB) {
  const rows = [...metricDefinitions, ["Observaciones holdout", "holdout_size"]];
  const highlights = metricDefinitions.map(([label, key]) => {
    const valueA = modelA.metrics[key];
    const valueB = modelB.metrics[key];
    const validA = valueA !== null && valueA !== undefined;
    const validB = valueB !== null && valueB !== undefined;
    const winner = validA && validB && valueA !== valueB ? (valueA > valueB ? "A" : "B") : "—";
    return `<div class="metric-highlight ${winner !== "—" ? "is-winner" : ""}">
      <span class="metric-highlight-label">${label}</span>
      <div class="metric-values"><strong>${metric(valueA)}</strong><span>A&nbsp;&nbsp;·&nbsp;&nbsp;${metric(valueB)} B</span></div>
      <span class="metric-winner">${winner === "—" ? "Sin diferencia" : `Ventaja modelo ${winner}`}</span>
    </div>`;
  }).join("");
  document.getElementById("metric-highlights").innerHTML = highlights;
  const body = rows.map(([label, key]) => {
    const valueA = key === "holdout_size" ? modelA.holdout_size : metric(modelA.metrics[key]);
    const valueB = key === "holdout_size" ? modelB.holdout_size : metric(modelB.metrics[key]);
    return `<tr><th>${label}</th><td>${valueA}</td><td>${valueB}</td></tr>`;
  }).join("");
  document.getElementById("metrics-table").innerHTML = `
    <table><thead><tr><th>Métrica</th><th>${escapeHtml(modelName(modelA))}</th>
    <th>${escapeHtml(modelName(modelB))}</th></tr></thead><tbody>${body}</tbody></table>`;
}

function renderConfusion(model, key) {
  const matrix = model.confusion_matrix;
  const header = model.classes.map((item) => `<th>${item.id}</th>`).join("");
  const rows = matrix.map((row, index) =>
    `<tr><th>${index}</th>${row.map((value) => `<td>${value}</td>`).join("")}</tr>`
  ).join("");
  document.getElementById(`confusion-${key}-title`).textContent = modelName(model);
  document.getElementById(`confusion-${key}`).innerHTML = `
    <table class="matrix"><thead><tr><th>Real \ Pred.</th>${header}</tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function drawCurve(canvasId, model) {
  const canvas = document.getElementById(canvasId);
  const context = canvas.getContext("2d");
  const curve = model.curve;
  const width = canvas.width;
  const height = canvas.height;
  const padding = 42;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#cbd5e1";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(padding, padding);
  context.lineTo(padding, height - padding);
  context.lineTo(width - padding, height - padding);
  context.stroke();
  context.fillStyle = "#475569";
  context.font = "12px sans-serif";
  context.fillText(curve.x_label, width / 2 - 25, height - 10);
  context.save();
  context.translate(13, height / 2 + 25);
  context.rotate(-Math.PI / 2);
  context.fillText(curve.y_label, 0, 0);
  context.restore();
  context.fillText(curve.type === "roc" ? "ROC" : "Precision–Recall", padding, 18);
  if (!curve.x.length) return;
  context.strokeStyle = "#2563eb";
  context.lineWidth = 2;
  context.beginPath();
  curve.x.forEach((x, index) => {
    const px = padding + Math.max(0, Math.min(1, x)) * (width - 2 * padding);
    const py = height - padding - Math.max(0, Math.min(1, curve.y[index])) * (height - 2 * padding);
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  });
  context.stroke();
}

function renderCurves(modelA, modelB) {
  document.getElementById("curve-a-title").textContent = modelName(modelA);
  document.getElementById("curve-b-title").textContent = modelName(modelB);
  drawCurve("curve-a", modelA);
  drawCurve("curve-b", modelB);
}

async function compareModels() {
  const modelA = modelASelect.value;
  const modelB = modelBSelect.value;
  if (!modelA || !modelB || modelA === modelB) {
    setStatus("Selecciona dos modelos diferentes.", true);
    return;
  }
  setStatus("Cargando comparación…");
  setLoading(true);
  try {
    const response = await fetch(`/api/compare?model_a=${encodeURIComponent(modelA)}&model_b=${encodeURIComponent(modelB)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "No se pudo comparar.");
    renderMap(payload.model_a, "a");
    renderMap(payload.model_b, "b");
    renderMetrics(payload.model_a, payload.model_b);
    renderConfusion(payload.model_a, "a");
    renderConfusion(payload.model_b, "b");
    renderCurves(payload.model_a, payload.model_b);
    setStatus(payload.same_satellite
      ? "Comparación entre modelos del mismo satélite."
      : "Comparación exploratoria entre satélites; cada mapa conserva sus observaciones.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setLoading(false);
  }
}

async function queryPoint(satellite, lat, lon) {
  const params = new URLSearchParams({
    satellite,
    lat,
    lon,
    model_a: modelASelect.value,
    model_b: modelBSelect.value,
  });
  setStatus("Consultando la observación más cercana…");
  try {
    const response = await fetch(`/api/point?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "No se pudo consultar el punto.");
    const predictionRows = payload.predictions.map((prediction) => `
      <tr><td>${escapeHtml(prediction.model_id)}</td>
      <td>${escapeHtml(prediction.satellite)}</td>
      <td>${escapeHtml(prediction.class_label)}</td>
      <td>${prediction.native_dataset ? "Sí" : "Exploratoria"}</td></tr>`).join("");
    document.getElementById("point-placeholder").textContent = "Resultado de la consulta:";
    document.getElementById("point-result").innerHTML = `
      <div class="point-summary">
        <span><b>Satélite:</b> ${payload.query.satellite}</span>
        <span><b>Latitud:</b> ${Number(payload.point.lat).toFixed(6)}</span>
        <span><b>Longitud:</b> ${Number(payload.point.lon).toFixed(6)}</span>
        <span><b>Distancia:</b> ${payload.distance_km} km</span>
        <span><b>Irradiancia:</b> ${payload.point.irradiance}</span>
        <span><b>Clase observada por el mapa:</b> ${escapeHtml(payload.point.class_label)}</span>
      </div>
      <table><thead><tr><th>Modelo</th><th>Satélite</th><th>Clase predicha</th><th>Datos nativos</th></tr></thead>
      <tbody>${predictionRows}</tbody></table>
      <p class="small-note">${escapeHtml(payload.note)}</p>`;
    setStatus("Punto consultado correctamente.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

compareButton.addEventListener("click", compareModels);
modelASelect.addEventListener("change", compareModels);
modelBSelect.addEventListener("change", compareModels);
loadModels().catch((error) => setStatus(error.message, true));
