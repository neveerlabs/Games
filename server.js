const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const initSqlJs = require('sql.js');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = 3000;
const DB_PATH = './tictactoe.db';
const MODEL_DIR = './models';
const CURRENT_MODEL_PATH = path.join(MODEL_DIR, 'current_model');
const MODEL_META_PATH = path.join(MODEL_DIR, 'model_meta.json');

let db;
let model = null;
let modelMeta = { version: 0, gamesTrained: 0 };

async function initDB() {
  const SQL = await initSqlJs();
  try {
    const fileBuffer = await fs.readFile(DB_PATH);
    db = new SQL.Database(fileBuffer);
  } catch {
    db = new SQL.Database();
  }
  db.run(`
    CREATE TABLE IF NOT EXISTS games (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      moves TEXT NOT NULL,
      winner TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS scores (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      playerWins INTEGER DEFAULT 0,
      aiWins INTEGER DEFAULT 0,
      draws INTEGER DEFAULT 0
    );
    INSERT OR IGNORE INTO scores (id, playerWins, aiWins, draws) VALUES (1, 0, 0, 0);
  `);
  await saveDB();
}

async function saveDB() {
  const data = db.export();
  await fs.writeFile(DB_PATH, data);
}

async function ensureModelDir() {
  try {
    await fs.mkdir(MODEL_DIR, { recursive: true });
  } catch (e) {}
}

async function loadModel() {
  await ensureModelDir();
  try {
    const metaRaw = await fs.readFile(MODEL_META_PATH, 'utf8');
    modelMeta = JSON.parse(metaRaw);
    model = await tf.loadLayersModel(`file://${CURRENT_MODEL_PATH}/model.json`);
    console.log(`Model v${modelMeta.version} loaded.`);
  } catch (e) {
    console.log('No existing model. Creating a new one.');
    model = createFreshModel();
    modelMeta = { version: 1, gamesTrained: 0 };
    await saveModel();
  }
}

function createFreshModel() {
  const input = tf.input({ shape: [27] });
  const dense1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(input);
  const dense2 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(dense1);
  const output = tf.layers.dense({ units: 9, activation: 'linear' }).apply(dense2);
  const m = tf.model({ inputs: input, outputs: output });
  m.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  return m;
}

async function saveModel() {
  await ensureModelDir();
  await model.save(`file://${CURRENT_MODEL_PATH}`);
  await fs.writeFile(MODEL_META_PATH, JSON.stringify(modelMeta, null, 2));
}

function boardToTensor(boardArray) {
  const arr = new Array(27).fill(0);
  for (let i = 0; i < 9; i++) {
    if (boardArray[i] === 'X') arr[i] = 1;
    else if (boardArray[i] === 'O') arr[i + 9] = 1;
    else arr[i + 18] = 1;
  }
  return arr;
}

function availableMoves(boardArray) {
  return boardArray.reduce((acc, cell, idx) => cell === '_' ? acc.concat(idx) : acc, []);
}

function checkWinner(boardArray) {
  const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (let [a,b,c] of lines) {
    if (boardArray[a] !== '_' && boardArray[a] === boardArray[b] && boardArray[b] === boardArray[c]) {
      return boardArray[a];
    }
  }
  if (boardArray.every(c => c !== '_')) return 'draw';
  return null;
}

app.post('/api/predict', async (req, res) => {
  try {
    const { board } = req.body;
    if (!board || board.length !== 9) return res.status(400).json({ error: 'Invalid board' });
    const tensor = tf.tensor2d([boardToTensor(board)]);
    const prediction = model.predict(tensor);
    const qValues = await prediction.data();
    tensor.dispose();
    prediction.dispose();
    const avail = availableMoves(board);
    const masked = avail.map(idx => ({ idx, value: qValues[idx] }));
    masked.sort((a,b) => b.value - a.value);
    res.json({ actions: masked });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/game/record', async (req, res) => {
  try {
    const { moves, winner } = req.body;
    db.run('INSERT INTO games (moves, winner) VALUES (?, ?)', [JSON.stringify(moves), winner]);
    if (winner === 'X') {
      db.run('UPDATE scores SET playerWins = playerWins + 1 WHERE id = 1');
    } else if (winner === 'O') {
      db.run('UPDATE scores SET aiWins = aiWins + 1 WHERE id = 1');
    } else {
      db.run('UPDATE scores SET draws = draws + 1 WHERE id = 1');
    }
    await saveDB();
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/scores', async (req, res) => {
  try {
    const stmt = db.prepare('SELECT playerWins, aiWins, draws FROM scores WHERE id = 1');
    stmt.step();
    const row = stmt.getAsObject();
    stmt.free();
    res.json(row);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.post('/api/scores/reset', async (req, res) => {
  try {
    db.run('UPDATE scores SET playerWins = 0, aiWins = 0, draws = 0 WHERE id = 1');
    await saveDB();
    res.json({ success: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/model/meta', (req, res) => {
  res.json(modelMeta);
});

app.post('/api/train/start', async (req, res) => {
  try {
    const episodes = parseInt(req.query.episodes) || 1000;
    res.json({ message: `Training started for ${episodes} episodes. Check console for progress.` });
    startTraining(episodes).catch(console.error);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

async function startTraining(episodes) {
  console.log(`Starting training for ${episodes} episodes...`);
  const batchSize = 32;
  let gamesPlayed = 0;
  const trainingData = { states: [], targets: [] };
  const epsilon = 0.2;
  const gamma = 0.99;
  
  for (let ep = 0; ep < episodes; ep++) {
    const history = await selfPlayGame(epsilon);
    gamesPlayed++;
    const winner = history.winner;
    let reward = winner === 'O' ? 1 : winner === 'X' ? -1 : 0.1;
    for (let i = history.states.length - 1; i >= 0; i--) {
      const state = history.states[i];
      const action = history.actions[i];
      const nextState = i < history.states.length - 1 ? history.states[i+1] : null;
      let target = reward;
      if (nextState) {
        const nextTensor = tf.tensor2d([boardToTensor(nextState)]);
        const nextPred = model.predict(nextTensor);
        const nextQ = await nextPred.data();
        target = reward + gamma * Math.max(...nextQ);
        nextTensor.dispose();
        nextPred.dispose();
      }
      const stateTensor = tf.tensor2d([boardToTensor(state)]);
      const pred = model.predict(stateTensor);
      const qValues = await pred.data();
      qValues[action] = target;
      trainingData.states.push(state);
      trainingData.targets.push(qValues);
      stateTensor.dispose();
      pred.dispose();
      reward = 0;
    }
    if (trainingData.states.length >= batchSize || ep === episodes - 1) {
      const xs = tf.tensor2d(trainingData.states.map(s => boardToTensor(s)));
      const ys = tf.tensor2d(trainingData.targets);
      await model.fit(xs, ys, { epochs: 1, verbose: 0 });
      xs.dispose();
      ys.dispose();
      trainingData.states = [];
      trainingData.targets = [];
    }
    if ((ep+1) % 100 === 0) {
      console.log(`Episode ${ep+1}/${episodes}`);
    }
  }
  modelMeta.gamesTrained += gamesPlayed;
  modelMeta.version += 1;
  await saveModel();
  console.log('Training completed and model saved.');
}

async function selfPlayGame(epsilon) {
  let board = Array(9).fill('_');
  let turn = 'X';
  const states = [];
  const actions = [];
  while (true) {
    const winner = checkWinner(board);
    if (winner) return { states, actions, winner };
    const avail = availableMoves(board);
    if (avail.length === 0) return { states, actions, winner: 'draw' };
    let move;
    if (turn === 'O') {
      if (Math.random() < epsilon) {
        move = avail[Math.floor(Math.random() * avail.length)];
      } else {
        const tensor = tf.tensor2d([boardToTensor(board)]);
        const pred = model.predict(tensor);
        const qs = await pred.data();
        let best = -Infinity;
        for (let a of avail) {
          if (qs[a] > best) { best = qs[a]; move = a; }
        }
        tensor.dispose();
        pred.dispose();
      }
      states.push([...board]);
      actions.push(move);
    } else {
      move = avail[Math.floor(Math.random() * avail.length)];
    }
    board[move] = turn;
    turn = turn === 'X' ? 'O' : 'X';
  }
}

initDB().then(() => loadModel()).then(() => {
  app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
});