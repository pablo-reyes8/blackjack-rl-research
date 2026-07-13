const state = {
  payload: null,
  busy: false,
  autoLoop: false,
};

const ACTION_ORDER = [
  "bet_1x",
  "bet_2x",
  "bet_3x",
  "bet_4x",
  "stand",
  "hit",
  "double",
  "split",
  "surrender",
  "insurance",
];

const ACTION_TEXT = {
  bet_1x: "Bet 1x",
  bet_2x: "Bet 2x",
  bet_3x: "Bet 3x",
  bet_4x: "Bet 4x",
  stand: "Stand",
  hit: "Hit",
  double: "Double",
  split: "Split",
  surrender: "Surrender",
  insurance: "Insurance",
};

const SUITS = [
  { symbol: "♠", name: "spade", color: "black" },
  { symbol: "♥", name: "heart", color: "red" },
  { symbol: "♣", name: "club", color: "black" },
  { symbol: "♦", name: "diamond", color: "red" },
];

function $(id) {
  return document.getElementById(id);
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function refresh() {
  setBusy(true);
  try {
    state.payload = await api("/api/state");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    render();
  }
}

async function post(path, body = {}) {
  setBusy(true);
  try {
    state.payload = await api(path, {
      method: "POST",
      body: JSON.stringify(body),
    });
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    render();
  }
}

function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("busy", value);
}

function suitFor(rank, index) {
  const text = String(rank || "");
  const rankScore = [...text].reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return SUITS[(rankScore + index) % SUITS.length];
}

function cardElement(rank, hidden = false, index = 0) {
  const card = document.createElement("div");
  card.className = "card";
  card.style.animationDelay = `${Math.min(index * 70, 420)}ms`;
  if (hidden) {
    card.classList.add("hidden");
    card.textContent = "?";
    return card;
  }
  const suit = suitFor(rank, index);
  card.classList.add(suit.color, suit.name);
  card.innerHTML = `
    <span class="corner top"><b>${rank || "--"}</b><i>${suit.symbol}</i></span>
    <span class="pip">${suit.symbol}</span>
    <span class="corner bottom"><b>${rank || "--"}</b><i>${suit.symbol}</i></span>
  `;
  return card;
}

function renderCards(container, cards, hiddenHole = false) {
  container.innerHTML = "";
  if (!cards || cards.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-seat";
    empty.textContent = "Las cartas se reparten despues de apostar";
    container.appendChild(empty);
    return;
  }
  cards.forEach((rank, index) => container.appendChild(cardElement(rank, false, index)));
  if (hiddenHole) container.appendChild(cardElement("?", true, cards.length));
}

function renderHands(publicState) {
  const holder = $("playerHands");
  holder.innerHTML = "";
  const hands = publicState.player_hands || [];
  if (!hands.length) {
    const empty = document.createElement("div");
    empty.className = "hand";
    empty.innerHTML = `<div class="hand-cards"><div class="empty-seat">Selecciona Bet 1x, 2x, 3x o 4x</div></div><div class="hand-meta"><span>Esperando apuesta</span><strong>Bet --</strong></div>`;
    holder.appendChild(empty);
    return;
  }
  hands.forEach((hand) => {
    const handNode = document.createElement("article");
    handNode.className = `hand ${hand.index === publicState.current_hand_index ? "active" : ""}`;
    const cards = document.createElement("div");
    cards.className = "hand-cards";
    renderCards(cards, hand.cards || []);
    const status = hand.settlement || hand.close_reason || (hand.closed ? "closed" : "active");
    handNode.appendChild(cards);
    handNode.insertAdjacentHTML(
      "beforeend",
      `<div class="hand-meta"><span>Total ${hand.total ?? "--"} ${hand.is_soft ? "soft" : ""} · ${status}</span><strong>Bet ${fmt(hand.bet)}</strong></div>`
    );
    holder.appendChild(handNode);
  });
}

function renderTable(payload) {
  const publicState = payload.public_state || {};
  const dealer = publicState.dealer || {};
  renderCards($("dealerCards"), dealer.cards || [], dealer.hole_card_hidden);
  $("dealerTotal").textContent = dealer.total
    ? `Total ${dealer.total}`
    : dealer.visible_total
      ? `Visible ${dealer.visible_total}`
      : "Total --";
  renderHands(publicState);

  const phase = publicState.decision_phase || "betting";
  $("phaseChip").textContent = payload.response.done ? "Round over" : phase;
  $("roundReward").textContent = `Reward ${fmt((publicState.round_reward ?? payload.response.reward), 2)}`;
  $("tableHint").textContent = hintText(payload);

  const shoe = publicState.shoe || {};
  const pct = Math.max(0, Math.min(100, Number(shoe.penetration_used || 0) * 100));
  $("shoeFill").style.width = `${pct}%`;
  $("shoeText").textContent = `${fmt(pct, 0)}%`;
  $("discardText").textContent = `${shoe.cards_used ?? "--"} used`;
  renderResultBanner(payload);
}

function resultText(result) {
  const labels = {
    win: "Ganaste",
    blackjack: "Blackjack",
    loss: "Perdiste",
    push: "Push",
    surrender: "Surrender",
  };
  return labels[result] || "Mano terminada";
}

function renderResultBanner(payload) {
  const banner = $("resultBanner");
  const last = (payload.session || {}).last_result;
  if (!payload.response.done || !last) {
    banner.className = "result-banner";
    banner.textContent = "";
    return;
  }
  const result = last.result || "push";
  banner.className = `result-banner show ${result}`;
  banner.innerHTML = `<strong>${resultText(result)}</strong><span>${fmt(last.reward, 2)}</span>`;
}

function renderActions(payload) {
  const holder = $("actionButtons");
  const rec = payload.recommendation || {};
  const legal = new Set(payload.response.legal_actions || []);
  holder.innerHTML = "";

  const groups = [
    { title: "Apuesta", className: "bet-actions", actions: ["bet_1x", "bet_2x", "bet_3x", "bet_4x"] },
    { title: "Jugada", className: "play-actions", actions: ["stand", "hit", "double", "split", "surrender", "insurance"] },
  ];

  groups.forEach((group) => {
    const groupNode = document.createElement("div");
    groupNode.className = `action-group ${group.className}`;
    groupNode.innerHTML = `<div class="action-group-title">${group.title}</div>`;
    const row = document.createElement("div");
    row.className = "action-row";
    group.actions.forEach((name) => {
      const qItem = (rec.actions || []).find((item) => item.name === name);
      const isLegal = legal.has(name) && !payload.response.done;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `action-btn ${qItem && qItem.is_best ? "best" : ""} ${isLegal ? "legal" : ""}`;
      button.disabled = state.busy || !isLegal;
      button.innerHTML = `<span>${ACTION_TEXT[name]}</span><strong>${qItem && qItem.is_best ? "AGENTE" : qItem && qItem.q !== null ? fmt(qItem.q, 1) : "--"}</strong>`;
      button.addEventListener("click", () => post("/api/action", { action: name }));
      row.appendChild(button);
    });
    groupNode.appendChild(row);
    holder.appendChild(groupNode);
  });

  $("suggestedBtn").disabled = state.busy || !rec.suggested_action || payload.response.done;
  $("autoBtn").disabled = state.busy || payload.response.done;
  $("autoLoopBtn").disabled = state.busy && !state.autoLoop;
  $("autoLoopBtn").textContent = state.autoLoop ? "Parar auto" : "Auto continuo";
  $("autoLoopBtn").classList.toggle("danger", state.autoLoop);
  $("newRoundBtn").textContent = payload.response.done ? "Siguiente mano" : "Repartir nueva";
}

function hintText(payload) {
  const publicState = payload.public_state || {};
  const rec = payload.recommendation || {};
  if (payload.response.done) {
    return "Mano terminada. Siguiente mano conserva el shoe y la memoria observada.";
  }
  if (publicState.decision_phase === "betting") {
    return `Fase apuesta: el agente recomienda ${rec.suggested_label || "--"}. Tu click reparte las cartas.`;
  }
  return `Fase jugada: el agente recomienda ${rec.suggested_label || "--"} para la mano activa.`;
}

function agentSentence(payload) {
  const rec = payload.recommendation || {};
  const publicState = payload.public_state || {};
  const action = rec.suggested_label || "--";
  if (payload.response.done) {
    return "La mano termino. Puedes seguir en la misma mesa para conservar historial y shoe.";
  }
  if (publicState.decision_phase === "betting") {
    return `El agente apostaria ${action}. Puedes aceptarlo o elegir otro tamano de apuesta.`;
  }
  const hand = publicState.current_hand || {};
  const dealer = (publicState.dealer || {}).upcard || "--";
  return `Con ${((hand.cards || []).join("-") || "--")} contra ${dealer}, el agente jugaria ${action}.`;
}

function renderQ(payload) {
  const rec = payload.recommendation || {};
  $("suggestionText").textContent = rec.suggested_label || "--";
  $("agentCallout").textContent = agentSentence(payload);
  const holder = $("qList");
  holder.innerHTML = "";
  const legalValues = (rec.actions || []).filter((item) => item.legal && item.q !== null).map((item) => item.q);
  const min = Math.min(...legalValues, 0);
  const max = Math.max(...legalValues, 1);
  const span = Math.max(0.0001, max - min);
  (rec.actions || []).forEach((item) => {
    const width = item.q === null ? 0 : Math.max(4, ((item.q - min) / span) * 100);
    const row = document.createElement("div");
    row.className = `q-row ${item.is_best ? "best" : ""} ${item.legal ? "" : "illegal"}`;
    row.innerHTML = `<span>${item.is_best ? "AGENTE: " : ""}${item.label}</span><div class="bar"><i style="width:${width}%"></i></div><strong>${item.q === null ? "--" : fmt(item.q, 2)}</strong>`;
    holder.appendChild(row);
  });
}

function renderCount(payload) {
  const count = (payload.recommendation || {}).count_bucket;
  $("countBest").textContent = count ? count.best_label : "--";
  const holder = $("countList");
  holder.innerHTML = "";
  if (!count) {
    holder.innerHTML = `<div class="q-row illegal"><span>No head</span><div class="bar"><i></i></div><strong>--</strong></div>`;
    return;
  }
  count.labels.forEach((label, index) => {
    const probability = count.probabilities[index] || 0;
    const row = document.createElement("div");
    row.className = "count-row";
    row.innerHTML = `<span>${label}</span><div class="bar"><i style="width:${probability * 100}%"></i></div><strong>${fmt(probability * 100, 0)}%</strong>`;
    holder.appendChild(row);
  });
}

function renderStats(payload) {
  const model = payload.model || {};
  const session = payload.session || {};
  const outcomes = session.outcomes || {};
  $("modelPill").textContent = `${model.key || "--"} · ${model.architecture || "--"} · dim ${model.state_dim || "--"}`;
  $("bankroll").textContent = fmt(session.total_reward, 2);
  $("rounds").textContent = session.completed_rounds || 0;
  $("avgReward").textContent = fmt(session.average_reward, 3);
  $("seed").textContent = model.seed || "--";
  $("wins").textContent = (outcomes.win || 0) + (outcomes.blackjack || 0);
  $("losses").textContent = (outcomes.loss || 0) + (outcomes.surrender || 0);
  $("pushes").textContent = outcomes.push || 0;
  $("ev100").textContent = fmt(session.ev_per_100_hands, 2);
}

function renderHistory(payload) {
  const info = payload.response.info || {};
  $("lastAction").textContent = info.action || "--";
  const history = (((payload.public_state || {}).history || {}).recent_actions || []).slice(-10).reverse();
  const holder = $("history");
  holder.innerHTML = "";
  history.forEach((event) => {
    const item = document.createElement("li");
    const extra = event.card ? ` · ${event.card}` : event.amount ? ` · ${fmt(event.amount)}` : "";
    item.textContent = `R${event.round_index}: ${event.actor} ${event.action}${extra}`;
    holder.appendChild(item);
  });
}

function render() {
  const payload = state.payload;
  if (!payload) return;
  renderTable(payload);
  renderActions(payload);
  renderQ(payload);
  renderCount(payload);
  renderStats(payload);
  renderHistory(payload);
}

$("suggestedBtn").addEventListener("click", () => post("/api/play-suggestion"));
$("autoBtn").addEventListener("click", () => post("/api/autoplay", { max_steps: 30 }));
$("autoLoopBtn").addEventListener("click", () => {
  state.autoLoop = !state.autoLoop;
  render();
  if (state.autoLoop) runAutoLoop();
});
$("newRoundBtn").addEventListener("click", () => post("/api/new-round"));
$("newTableBtn").addEventListener("click", () => {
  const current = state.payload && state.payload.model ? state.payload.model.key : "05A";
  const next = current === "05A" ? "04D" : "05A";
  const useNext = window.confirm(`OK cambia a ${next}. Cancel deja ${current} y solo reinicia la mesa.`);
  post("/api/new-table", { model_key: useNext ? next : current });
});

refresh();

async function runAutoLoop() {
  while (state.autoLoop) {
    if (!state.payload) {
      await refresh();
      continue;
    }
    if (state.payload.response && state.payload.response.done) {
      await post("/api/new-round");
    } else {
      await post("/api/autoplay", { max_steps: 30 });
    }
    await new Promise((resolve) => window.setTimeout(resolve, 650));
  }
}
