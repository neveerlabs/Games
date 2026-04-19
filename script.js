const API_BASE = 'http://localhost:3000/api';
let EPSILON = 0.0, LEARNING_RATE = 0.25, DISCOUNT = 0.98;
const PLAYER = 'X', AI = 'O';
let board = Array(9).fill('_');
let gameActive = true, playerTurn = true;
let playerScore = 0, aiScore = 0, draws = 0;
let moveHistory = [];
let trainingActive = false, stopTraining = false;
let modelMeta = { version: 0, gamesTrained: 0 };

function stateStr(b) { return b.join(''); }
function availableMoves(b) { let m = []; b.forEach((v, i) => { if (v === '_') m.push(i); }); return m; }

function checkWinner(b) {
  const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (let [a,bc,c] of lines) if (b[a] !== '_' && b[a] === b[bc] && b[bc] === b[c]) return { winner: b[a], indices: [a,bc,c] };
  if (b.every(cell => cell !== '_')) return { winner: 'draw', indices: [] };
  return { winner: null, indices: [] };
}

function findWinningMove(b, sym) {
  let avail = availableMoves(b);
  for (let idx of avail) { let copy = b.slice(); copy[idx] = sym; if (checkWinner(copy).winner === sym) return idx; }
  return -1;
}

function findForkMove(b, sym) {
  let forks = [];
  for (let idx of availableMoves(b)) {
    let copy = b.slice();
    copy[idx] = sym;
    let winPaths = 0;
    const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
    for (let [a,bc,c] of lines) {
      if (copy[a] === sym && copy[bc] === sym && copy[c] === '_') winPaths++;
      if (copy[a] === sym && copy[c] === sym && copy[bc] === '_') winPaths++;
      if (copy[bc] === sym && copy[c] === sym && copy[a] === '_') winPaths++;
    }
    if (winPaths >= 2) forks.push(idx);
  }
  return forks.length ? forks[0] : -1;
}

function getHeuristicMove(b, sym, opp) {
  let win = findWinningMove(b, sym);
  if (win !== -1) return win;
  let block = findWinningMove(b, opp);
  if (block !== -1) return block;
  let fork = findForkMove(b, sym);
  if (fork !== -1) return fork;
  let blockFork = findForkMove(b, opp);
  if (blockFork !== -1) return blockFork;
  if (b[4] === '_') return 4;
  const corners = [0,2,6,8];
  for (let c of corners) if (b[c] === '_') return c;
  let avail = availableMoves(b);
  return avail.length ? avail[Math.floor(Math.random() * avail.length)] : -1;
}

function renderBoardUI() {
  const container = document.getElementById('board');
  container.innerHTML = '';
  board.forEach((val, idx) => {
    const cell = document.createElement('div');
    cell.className = `cell ${val === 'X' ? 'x' : val === 'O' ? 'o' : ''}`;
    cell.dataset.index = idx;
    if (val !== '_') cell.innerText = val;
    cell.addEventListener('click', () => onUserClick(idx));
    container.appendChild(cell);
  });
}

function highlightWin(indices) {
  const cells = document.querySelectorAll('.cell');
  indices.forEach(i => { if (cells[i]) cells[i].classList.add('win'); });
}
function clearHighlights() { document.querySelectorAll('.cell.win').forEach(c => c.classList.remove('win')); }

function showResultModal(title, msg, isWin) {
  document.getElementById('resultTitle').innerText = title;
  document.getElementById('resultMsg').innerText = msg;
  const crownDiv = document.getElementById('crownArea');
  crownDiv.innerHTML = isWin ? '<img src="https://img.icons8.com/ios-filled/90/FFD966/crown.png" width="52" style="filter: drop-shadow(0 0 6px gold);">' : '';
  document.getElementById('resultModal').classList.add('active');
}
function hideResultModal() { document.getElementById('resultModal').classList.remove('active'); }

async function fetchScores() {
  const res = await fetch(`${API_BASE}/scores`);
  const data = await res.json();
  playerScore = data.playerWins;
  aiScore = data.aiWins;
  draws = data.draws;
  updateUIStats();
}

async function updateScoresOnServer() {
  await fetch(`${API_BASE}/scores/reset`, { method: 'POST' });
}

async function recordGame(moves, winner) {
  await fetch(`${API_BASE}/game/record`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves, winner })
  });
}

async function handleGameOver(result, winIdx = []) {
  clearHighlights();
  const moveRecords = moveHistory.map(m => ({ state: m.state, action: m.action }));
  await recordGame(moveRecords, result);
  if (result === PLAYER) {
    highlightWin(winIdx);
    showResultModal('HUMAN VICTORY', 'AI will learn from this loss', false);
  } else if (result === AI) {
    highlightWin(winIdx);
    showResultModal('AI DOMINATION', 'AI recorded a win', true);
    startParticleEffect();
  } else {
    showResultModal('STRATEGIC DRAW', 'Game ended in a draw', false);
  }
  await fetchScores();
}

async function makeMove(idx, symbol) {
  if (!gameActive) return;
  if (board[idx] !== '_') return;
  board[idx] = symbol;
  renderBoardUI();
  let res = checkWinner(board);
  if (res.winner) { gameActive = false; await handleGameOver(res.winner, res.indices); return; }
  if (res.winner === 'draw') { gameActive = false; await handleGameOver('draw'); return; }
  playerTurn = (symbol !== PLAYER);
  updateUIStats();
  if (!playerTurn && gameActive) setTimeout(() => makeAIMove(), 150);
}

async function recordAIMove(state, actionIdx) {
  board[actionIdx] = AI;
  let nextState = stateStr(board);
  moveHistory.push({ state, action: actionIdx, nextState });
  renderBoardUI();
  let res = checkWinner(board);
  if (res.winner) { gameActive = false; await handleGameOver(res.winner, res.indices); return; }
  if (res.winner === 'draw') { gameActive = false; await handleGameOver('draw'); return; }
  playerTurn = true;
  updateUIStats();
}

function setThinkingStatus(isThinking) {
  const turnLabel = document.getElementById('turnLabel');
  if (isThinking) {
    turnLabel.innerHTML = 'THINKING...';
    turnLabel.style.color = '#3c8eff';
  } else {
    turnLabel.innerHTML = playerTurn ? 'PLAYER (X)' : 'AI (O)';
    turnLabel.style.color = '';
  }
}

async function makeAIMove() {
  if (!gameActive) return;
  setThinkingStatus(true);
  let state = stateStr(board);
  let avail = availableMoves(board);
  if (avail.length === 0) { setThinkingStatus(false); return; }
  let win = findWinningMove(board, AI);
  if (win !== -1) { await recordAIMove(state, win); setThinkingStatus(false); return; }
  let block = findWinningMove(board, PLAYER);
  if (block !== -1) { await recordAIMove(state, block); setThinkingStatus(false); return; }
  let fork = findForkMove(board, AI);
  if (fork !== -1) { await recordAIMove(state, fork); setThinkingStatus(false); return; }
  let blockFork = findForkMove(board, PLAYER);
  if (blockFork !== -1) { await recordAIMove(state, blockFork); setThinkingStatus(false); return; }
  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ board })
    });
    const data = await response.json();
    const actions = data.actions;
    let chosen = actions[0].idx;
    await recordAIMove(state, chosen);
  } catch (e) {
    const fallback = getHeuristicMove(board, AI, PLAYER);
    await recordAIMove(state, fallback);
  }
  setThinkingStatus(false);
}

async function onUserClick(idx) {
  if (gameActive && playerTurn && board[idx] === '_') await makeMove(idx, PLAYER);
}

function newGameRound() {
  board = Array(9).fill('_');
  gameActive = true;
  playerTurn = true;
  moveHistory = [];
  clearHighlights();
  renderBoardUI();
  updateUIStats();
}

function updateUIStats() {
  document.getElementById('playerScore').innerText = playerScore;
  document.getElementById('aiScore').innerText = aiScore;
  document.getElementById('qStateCount').innerText = modelMeta.gamesTrained || 0;
  document.getElementById('turnLabel').innerHTML = playerTurn ? 'PLAYER (X)' : 'AI (O)';
  document.getElementById('epsilonValue').innerText = EPSILON.toFixed(2);
  document.getElementById('epsVal').innerText = EPSILON.toFixed(2);
  document.getElementById('lrVal').innerText = LEARNING_RATE.toFixed(2);
  document.getElementById('gammaVal').innerText = DISCOUNT.toFixed(2);
}

async function runTrainingSession(totalGames) {
  if (trainingActive) return;
  trainingActive = true;
  stopTraining = false;
  const progressDiv = document.getElementById('trainProgress');
  const progressFill = document.getElementById('progressFill');
  const stopBtn = document.getElementById('stopTrainBtn');
  progressDiv.style.display = 'block';
  stopBtn.style.display = 'inline-flex';
  try {
    const res = await fetch(`${API_BASE}/train/start?episodes=${totalGames}`, { method: 'POST' });
    const data = await res.json();
    alert(data.message);
  } catch (e) {
    alert('Training failed to start. Ensure server is running.');
  }
  progressDiv.style.display = 'none';
  stopBtn.style.display = 'none';
  trainingActive = false;
}

function startParticleEffect() {
  const canvas = document.getElementById('particleCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  let particles = [];
  for (let i = 0; i < 100; i++) particles.push({
    x: innerWidth / 2 + (Math.random() - 0.5) * 400,
    y: innerHeight / 2 + (Math.random() - 0.5) * 300,
    vx: (Math.random() - 0.5) * 8,
    vy: -Math.random() * 8 - 2,
    life: 60,
    size: Math.random() * 5 + 2,
    color: `rgba(43,122,255,${Math.random() * 0.7 + 0.3})`
  });
  function anim() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => { p.x += p.vx; p.y += p.vy; p.vy += 0.12; p.life--; ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = p.color; ctx.fill(); });
    particles = particles.filter(p => p.life > 0);
    if (particles.length) requestAnimationFrame(anim);
  }
  anim();
  setTimeout(() => ctx.clearRect(0, 0, canvas.width, canvas.height), 2200);
}
window.addEventListener('resize', () => { const c = document.getElementById('particleCanvas'); c.width = window.innerWidth; c.height = window.innerHeight; });

async function fetchModelMeta() {
  try {
    const res = await fetch(`${API_BASE}/model/meta`);
    modelMeta = await res.json();
    updateUIStats();
  } catch (e) {}
}

async function attachEvents() {
  document.getElementById('resetScoreBtn').onclick = async () => { await updateScoresOnServer(); await fetchScores(); };
  document.getElementById('homeBtn').onclick = () => location.reload();
  document.getElementById('rematchBtn').onclick = () => { hideResultModal(); newGameRound(); };
  document.getElementById('homeModalBtn').onclick = () => location.reload();
  document.getElementById('clearMemoryBtn').onclick = () => { alert('Memory is managed on server. Use server commands to reset model.'); };
  document.getElementById('exportQBtn').onclick = () => { alert('Model is stored on server. Use API to export.'); };
  document.getElementById('importQBtn').onclick = () => { alert('Import via server file system.'); };
  document.getElementById('train1k').onclick = () => runTrainingSession(1000);
  document.getElementById('train5k').onclick = () => runTrainingSession(5000);
  document.getElementById('train50k').onclick = () => runTrainingSession(50000);
  document.getElementById('stopTrainBtn').onclick = () => { if (trainingActive) stopTraining = true; };
  document.getElementById('epsilonSlider').oninput = (e) => { EPSILON = parseFloat(e.target.value); updateUIStats(); };
  document.getElementById('lrSlider').oninput = (e) => { LEARNING_RATE = parseFloat(e.target.value); updateUIStats(); };
  document.getElementById('gammaSlider').oninput = (e) => { DISCOUNT = parseFloat(e.target.value); updateUIStats(); };
  document.getElementById('licenseLink').onclick = (e) => { e.preventDefault(); document.getElementById('licenseModal').classList.add('active'); };
  document.getElementById('privacyLink').onclick = (e) => { e.preventDefault(); document.getElementById('privacyModal').classList.add('active'); };
  document.getElementById('closeLicenseBtn').onclick = () => document.getElementById('licenseModal').classList.remove('active');
  document.getElementById('closePrivacyBtn').onclick = () => document.getElementById('privacyModal').classList.remove('active');
  window.onclick = (e) => { if (e.target.classList.contains('modal')) document.querySelectorAll('.modal').forEach(m => m.classList.remove('active')); };
}

async function bootstrap() {
  await fetchScores();
  await fetchModelMeta();
  renderBoardUI();
  updateUIStats();
  await attachEvents();
}
bootstrap();