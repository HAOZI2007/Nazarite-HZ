'use strict';
const $ = (id) => document.getElementById(id);
const canvas = $('board');
const ctx = canvas.getContext('2d');
let model = null, position = null, busy = false, human = -1, nextHuman = -1;
let hover = null, keyboardCell = null, noticeTimer;
let geometry = { width: 0, height: 0, padding: 0, step: 0, rows: 15, cols: 15 };

async function api(path, data) {
  const response = await fetch('/api/' + path, data === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败');
  return result;
}

function notify(text, error = false) {
  clearTimeout(noticeTimer);
  $('message').textContent = text;
  $('message').className = error ? 'error' : '';
  $('message').hidden = false;
  noticeTimer = setTimeout(() => { $('message').hidden = true; }, error ? 8000 : 3500);
}

async function task(action) {
  if (busy) return;
  busy = true;
  hover = null;
  render();
  try { await action(); }
  catch (error) { notify(error.message, true); }
  finally { busy = false; render(); }
}

function coordinate(action) {
  return String.fromCharCode(65 + action % model.cols) + (model.rows - Math.floor(action / model.cols));
}

function render() {
  const loaded = !!model;
  const aiTurn = position && !position.terminated && position.current_player !== human;
  $('boardCover').hidden = loaded;
  $('loadModel').disabled = busy;
  $('refresh').disabled = busy;
  $('modelSelect').disabled = busy;
  $('modelPath').disabled = busy;
  $('newGame').disabled = busy || !loaded;
  $('undo').disabled = busy || !position?.moves.length;
  $('retry').hidden = busy || !aiTurn;
  $('simulations').disabled = busy;
  $('playBlack').disabled = busy;
  $('playWhite').disabled = busy;
  $('loadModel').textContent = busy && !position ? '正在加载…' : '加载模型';
  $('modelTag').textContent = loaded ? '已就绪' : '未加载';
  $('modelTag').className = loaded ? 'tag ready' : 'tag';
  $('boardSize').textContent = loaded ? `${model.rows} × ${model.cols}` : '— × —';
  $('moveCount').textContent = position?.moves.length || 0;
  $('blackName').textContent = human === -1 ? '你 · 黑棋' : 'AI · 黑棋';
  $('whiteName').textContent = human === 1 ? '你 · 白棋' : 'AI · 白棋';
  const current = position?.current_player ?? human;
  $('turnStone').className = 'turn-indicator' + (current === 1 ? ' white' : '');
  if (position?.terminated) {
    $('status').textContent = position.winner === 0 ? '和棋，棋逢对手' : position.winner === human ? '你赢了，精彩的一局！' : 'AI 获胜，再来一局？';
    $('statusDetail').textContent = `本局共 ${position.moves.length} 手 · 点击重新开局，继续切磋`;
  } else if (loaded) {
    $('status').textContent = aiTurn ? (busy ? 'AI 正在思考…' : '等待 AI 落子') : '轮到你落子';
    $('statusDetail').textContent = aiTurn ? (busy ? `正在进行 ${$('simulations').value} 次 MCTS 搜索` : '点击重试 AI 落子，继续对局') : `你执${human === -1 ? '黑' : '白'}棋 · 点击交叉点落子`;
  }
  const history = $('history');
  history.replaceChildren();
  if (!position?.moves.length) {
    const li = document.createElement('li');
    li.className = 'empty-record'; li.textContent = '每一步，都从这里开始。'; history.append(li);
  } else {
    position.moves.forEach((action, index) => {
      const li = document.createElement('li');
      const num = document.createElement('span'); num.className = 'number'; num.textContent = String(index + 1).padStart(2, '0');
      const stone = document.createElement('i'); stone.className = 'stone ' + (index % 2 ? 'white' : 'black');
      const who = document.createElement('span'); who.textContent = (index % 2 ? 1 : -1) === human ? '你' : 'AI';
      const coord = document.createElement('span'); coord.className = 'coordinate'; coord.textContent = coordinate(action);
      li.append(num, stone, who, coord); history.append(li);
    });
    history.scrollTop = history.scrollHeight;
  }
  draw();
}

async function refreshModels() {
  const result = await api('models');
  const select = $('modelSelect'), previous = select.value;
  select.replaceChildren();
  for (const item of result.models) {
    const option = document.createElement('option');
    option.value = item.path; option.textContent = item.name;
    option.title = item.path; select.append(option);
  }
  if (!result.models.length) {
    const option = document.createElement('option'); option.value = ''; option.textContent = '未找到模型，请填写本机路径'; select.append(option);
  } else if (result.models.some((item) => item.path === previous)) select.value = previous;
  return result;
}

function showModelInfo() {
  $('modelInfo').querySelector('p').textContent = `${model.task} · 已训练 ${model.iterations} 轮 · ${model.device.toUpperCase()}\n${model.name}`;
  $('modelInfo').title = model.path;
}

async function aiMove() {
  if (!position || position.terminated || position.current_player === human) return;
  render();
  const result = await api('move', { moves: position.moves, simulations: Number($('simulations').value) });
  position = result;
  $('thinkTime').textContent = `AI 用时 ${result.elapsed.toFixed(2)} s`;
}

async function startGame() {
  human = nextHuman;
  position = await api('position', { moves: [] });
  keyboardCell = null;
  $('thinkTime').textContent = '等待落子';
  await aiMove();
}

$('loadModel').addEventListener('click', () => {
  if (model && position?.moves.length && !position.terminated && !confirm('加载模型会结束当前对局，继续吗？')) return;
  task(async () => {
    const loaded = await api('load', { path: $('modelPath').value.trim() || $('modelSelect').value });
    model = loaded;
    showModelInfo();
    await startGame();
    notify('模型已入席，开始对弈吧。');
  });
});
$('refresh').addEventListener('click', () => task(refreshModels));
$('newGame').addEventListener('click', () => {
  if (position?.moves.length && !position.terminated && !confirm('结束当前对局并重新开局？')) return;
  task(startGame);
});
$('retry').addEventListener('click', () => task(aiMove));
$('undo').addEventListener('click', () => task(async () => {
  const moves = position.moves.slice();
  // 撤回最近一次人类落子及其后的 AI 应手。
  let index = moves.length - 1;
  while (index >= 0 && (index % 2 ? 1 : -1) !== human) index--;
  if (index < 0) { notify('你还没有落子，暂时无法悔棋。'); return; }
  position = await api('position', { moves: moves.slice(0, index) });
  $('thinkTime').textContent = '已悔棋';
}));
for (const [id, player] of [['playBlack', -1], ['playWhite', 1]]) {
  $(id).addEventListener('click', () => {
    nextHuman = player;
    $('playBlack').classList.toggle('selected', player === -1);
    $('playWhite').classList.toggle('selected', player === 1);
    $('playBlack').setAttribute('aria-pressed', String(player === -1));
    $('playWhite').setAttribute('aria-pressed', String(player === 1));
    if (model) notify('执子已选择，点击「重新开局」生效。');
  });
}
$('simulations').addEventListener('input', () => { $('simulationValue').textContent = $('simulations').value; });
$('showNumbers').addEventListener('change', draw);

function canPlay() {
  return model && position && !busy && !position.terminated && position.current_player === human;
}

function playAt(cell) {
  if (!canPlay() || !cell || position.board[cell[0]][cell[1]]) return;
  task(async () => {
    position = await api('position', { moves: [...position.moves, cell[0] * model.cols + cell[1]] });
    await aiMove();
  });
}

function cellAt(event) {
  const rect = canvas.getBoundingClientRect();
  const { padding, step, rows, cols } = geometry;
  const x = (event.clientX - rect.left) * geometry.width / rect.width;
  const y = (event.clientY - rect.top) * geometry.height / rect.height;
  const col = Math.round((x - padding) / step), row = Math.round((y - padding) / step);
  if (row < 0 || col < 0 || row >= rows || col >= cols) return null;
  if (Math.hypot(x - padding - col * step, y - padding - row * step) > step * .48) return null;
  return [row, col];
}
canvas.addEventListener('pointermove', (event) => { hover = canPlay() ? cellAt(event) : null; draw(); });
canvas.addEventListener('pointerleave', () => { hover = null; draw(); });
canvas.addEventListener('click', (event) => { keyboardCell = null; playAt(cellAt(event)); });
canvas.addEventListener('keydown', (event) => {
  if (!canPlay()) return;
  const directions = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (directions[event.key]) {
    event.preventDefault();
    if (!keyboardCell) keyboardCell = [Math.floor(model.rows / 2), Math.floor(model.cols / 2)];
    else keyboardCell = [Math.max(0, Math.min(model.rows - 1, keyboardCell[0] + directions[event.key][0])), Math.max(0, Math.min(model.cols - 1, keyboardCell[1] + directions[event.key][1]))];
    draw();
  } else if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault(); playAt(keyboardCell);
  }
});

function draw() {
  const width = canvas.clientWidth;
  if (!width) return;
  const rows = model?.rows || 15, cols = model?.cols || 15;
  const padding = Math.max(24, width * .065), step = (width - 2 * padding) / (cols - 1);
  const height = padding * 2 + step * (rows - 1);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = height + 'px';
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  geometry = { width, height, padding, step, rows, cols };
  ctx.clearRect(0, 0, width, height);
  // 淡木纹在各种屏幕尺寸下保持清晰，无外部图片依赖。
  ctx.strokeStyle = '#ae854b0b'; ctx.lineWidth = 1;
  for (let y = 3; y < height; y += 5) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.bezierCurveTo(width * .3, y - 3, width * .6, y + 3, width, y); ctx.stroke();
  }
  ctx.strokeStyle = '#95774799'; ctx.lineWidth = .8;
  for (let row = 0; row < rows; row++) {
    ctx.beginPath(); ctx.moveTo(padding, padding + row * step); ctx.lineTo(width - padding, padding + row * step); ctx.stroke();
  }
  for (let col = 0; col < cols; col++) {
    ctx.beginPath(); ctx.moveTo(padding + col * step, padding); ctx.lineTo(padding + col * step, height - padding); ctx.stroke();
  }
  ctx.fillStyle = '#8f764e'; ctx.font = `${Math.max(9, Math.min(11, step * .29))}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let col = 0; col < cols; col++) ctx.fillText(String.fromCharCode(65 + col), padding + col * step, height - padding * .4);
  for (let row = 0; row < rows; row++) ctx.fillText(String(rows - row), padding * .4, padding + row * step);
  const stars = rows === cols && rows % 2 ? [[Math.floor(rows / 2), Math.floor(cols / 2)]] : [];
  if (rows === cols && rows >= 11) stars.push([3, 3], [3, cols - 4], [rows - 4, 3], [rows - 4, cols - 4]);
  ctx.fillStyle = '#8b7047';
  for (const [row, col] of stars) { ctx.beginPath(); ctx.arc(padding + col * step, padding + row * step, Math.max(2, step * .065), 0, Math.PI * 2); ctx.fill(); }
  if (!position) return;
  const radius = step * .415;
  position.moves.forEach((action, index) => {
    const row = Math.floor(action / cols), col = action % cols;
    const x = padding + col * step, y = padding + row * step;
    const black = index % 2 === 0;
    ctx.save(); ctx.shadowColor = '#322a2445'; ctx.shadowBlur = radius * .3; ctx.shadowOffsetY = radius * .13;
    const gradient = ctx.createRadialGradient(x - radius * .35, y - radius * .4, 0, x, y, radius);
    gradient.addColorStop(0, black ? '#58605b' : '#fffef9'); gradient.addColorStop(1, black ? '#131b17' : '#e4e6da');
    ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    if ($('showNumbers').checked) {
      ctx.fillStyle = black ? '#e7ece4' : '#5e6959'; ctx.font = `500 ${radius * .85}px sans-serif`; ctx.fillText(String(index + 1), x, y + .5);
    } else if (index === position.moves.length - 1) {
      ctx.strokeStyle = '#b4684d'; ctx.lineWidth = Math.max(1.5, radius * .12); ctx.strokeRect(x - radius * .19, y - radius * .19, radius * .38, radius * .38);
    }
  });
  if (position.winner) drawWinningLine();
  const preview = hover || keyboardCell;
  if (canPlay() && preview && !position.board[preview[0]][preview[1]]) {
    ctx.fillStyle = human === -1 ? '#1f2c2250' : '#ffffffa0';
    ctx.beginPath(); ctx.arc(padding + preview[1] * step, padding + preview[0] * step, radius, 0, Math.PI * 2); ctx.fill();
  }
}

function drawWinningLine() {
  const { padding, step, rows, cols } = geometry;
  const action = position.moves.at(-1), r = Math.floor(action / cols), c = action % cols;
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const ends = [[r, c], [r, c]];
    for (let i = 0; i < 2; i++) {
      const sign = i === 0 ? -1 : 1;
      while (true) {
        const nr = ends[i][0] + dr * sign, nc = ends[i][1] + dc * sign;
        if (nr < 0 || nc < 0 || nr >= rows || nc >= cols || position.board[nr][nc] !== position.winner) break;
        ends[i] = [nr, nc];
      }
    }
    if (Math.max(Math.abs(ends[0][0] - ends[1][0]), Math.abs(ends[0][1] - ends[1][1])) < 4) continue;
    ctx.strokeStyle = '#b46d4f'; ctx.lineWidth = Math.max(2, step * .06); ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(padding + ends[0][1] * step, padding + ends[0][0] * step); ctx.lineTo(padding + ends[1][1] * step, padding + ends[1][0] * step); ctx.stroke();
  }
}

new ResizeObserver(draw).observe(canvas.parentElement);
render();
task(async () => {
  const result = await refreshModels();
  if (result.loaded) {
    model = result.loaded;
    showModelInfo();
    await startGame();
  }
});
