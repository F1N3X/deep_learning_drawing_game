// ═══════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════
const API_URL = 'http://localhost:8000/predict';
const CLASSES   = ['The Eiffel Tower','airplane','alarm clock','ant','apple','axe','bee','bicycle','bush','cactus','crown','dolphin','dragon','penguin','star'];
const WORDS     = ['The Eiffel Tower','airplane','alarm clock','ant','apple','axe','bee','bicycle','bush','cactus','crown','dolphin','dragon','penguin','star'];
const TIMES     = [20, 15, 10]; // per life
const PREDICT_INTERVAL = 400; // ms between predictions
const CONFIDENCE_THRESHOLD = 85; // minimum confidence % to validate a correct guess

// ═══════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════
let lives       = 3;
let score       = 0;
let life_index  = 0; // 0=first life=20s, 1=15s, 2=10s
let current_word= '';
let time_left   = 0;
let timer_id    = null;
let predict_id  = null;
let event_id    = null;
let active_events = {};
let drawing     = false;
let last_x = 0, last_y = 0;
let is_running  = false;
let correct_flag= false;

// ═══════════════════════════════════════════════════════
// DOM
// ═══════════════════════════════════════════════════════
const canvas    = document.getElementById('draw-canvas');
const ctx       = canvas.getContext('2d');
const wrapper   = document.getElementById('canvas-wrapper');
const border    = document.getElementById('canvas-border');
const overlay   = document.getElementById('overlay');
const overlayTitle = document.getElementById('overlay-title');
const overlaySub   = document.getElementById('overlay-sub');
const overlayBtn   = document.getElementById('overlay-btn');
const wordEl    = document.getElementById('word');
const guessEl   = document.getElementById('guess');
const confEl    = document.getElementById('confidence');
const timerBar  = document.getElementById('timer-bar');
const timerText = document.getElementById('timer-text');
const flashEl   = document.getElementById('flash');
const livesEls  = [document.getElementById('h1'), document.getElementById('h2'), document.getElementById('h3')];
const scoreEl   = document.getElementById('score');
const tickerEl  = document.getElementById('event-ticker');
const activeEvEl= document.getElementById('active-events');
const clearBtn  = document.getElementById('clear-btn');

// ═══════════════════════════════════════════════════════
// DRAW
// ═══════════════════════════════════════════════════════
function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * scaleX,
    y: (src.clientY - rect.top)  * scaleY
  };
}

canvas.addEventListener('mousedown', e => {
  if (!is_running) return;
  drawing = true;
  const p = getPos(e);
  last_x = p.x; last_y = p.y;
  ctx.beginPath();
  ctx.arc(last_x, last_y, 9, 0, Math.PI * 2);
  ctx.fillStyle = '#000'; ctx.fill();
});
canvas.addEventListener('mousemove', e => {
  if (!drawing || !is_running) return;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(last_x, last_y);
  ctx.lineTo(p.x, p.y);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();
  last_x = p.x; last_y = p.y;
});
canvas.addEventListener('mouseup',   () => drawing = false);
canvas.addEventListener('mouseleave',() => drawing = false);

canvas.addEventListener('touchstart', e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mousedown', {clientX: e.touches[0].clientX, clientY: e.touches[0].clientY})); }, {passive:false});
canvas.addEventListener('touchmove',  e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: e.touches[0].clientX, clientY: e.touches[0].clientY})); }, {passive:false});
canvas.addEventListener('touchend',   e => { e.preventDefault(); canvas.dispatchEvent(new MouseEvent('mouseup')); }, {passive:false});

clearBtn.addEventListener('click', () => { clearCanvas(); });

// ═══════════════════════════════════════════════════════
// PREDICTION
// ═══════════════════════════════════════════════════════
async function predict() {
  if (!is_running) return;
  try {
    // Check if canvas has any marks
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let hasDrawing = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] < 240) { hasDrawing = true; break; }
    }
    if (!hasDrawing) return;

    const imageData = canvas.toDataURL('image/png');
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    if (!res.ok) return;
    const data2 = await res.json();
    updateGuess(data2.guess, data2.confidence);
  } catch(e) {}
}

function updateGuess(guessWord, confidence) {
  if (!is_running) return;
  guessEl.textContent = guessWord.toUpperCase();
  confEl.textContent  = `${confidence.toFixed(1)}%`;

  const wordMatch = guessWord.toLowerCase() === current_word.toLowerCase();
  const isCorrect = wordMatch && confidence >= CONFIDENCE_THRESHOLD;

  if (isCorrect && !correct_flag) {
    correct_flag = true;
    onCorrect();
  }

  if (isCorrect) {
    // Right word + confident enough → validate
    guessEl.className = 'correct';
    confEl.style.color = 'var(--neon-green)';
    border.style.borderColor = 'var(--neon-green)';
    border.style.boxShadow = '0 0 20px var(--neon-green), inset 0 0 10px rgba(57,255,20,0.1)';
  } else if (wordMatch) {
    // Right word but below threshold → warming up
    guessEl.className = '';
    confEl.style.color = 'var(--neon-yellow)';
    border.style.borderColor = 'var(--neon-yellow)';
    border.style.boxShadow = '0 0 14px var(--neon-yellow)';
  } else {
    guessEl.className = '';
    confEl.style.color = confidence > 60 ? 'var(--neon-cyan)' : '#444';
    border.style.borderColor = 'var(--neon-cyan)';
    border.style.boxShadow = '0 0 10px var(--neon-cyan), inset 0 0 10px rgba(0,245,255,0.05)';
  }
}

// ═══════════════════════════════════════════════════════
// TIMER
// ═══════════════════════════════════════════════════════
function startTimer() {
  const total = TIMES[life_index] || 5;
  time_left = total;
  updateTimerUI(total, total);

  clearInterval(timer_id);
  timer_id = setInterval(() => {
    time_left -= 0.1;
    updateTimerUI(time_left, total);
    if (time_left <= 0) {
      clearInterval(timer_id);
      onTimeUp();
    }
  }, 100);
}

function updateTimerUI(left, total) {
  const pct = Math.max(0, left / total) * 100;
  timerBar.style.width = pct + '%';
  timerText.textContent = Math.ceil(Math.max(0, left)) + 's';
  if (pct > 50)       timerBar.style.background = 'var(--neon-yellow)';
  else if (pct > 25)  timerBar.style.background = '#ff9900';
  else                timerBar.style.background = 'var(--neon-pink)';
}

// ═══════════════════════════════════════════════════════
// WARIOWARE EVENTS
// ═══════════════════════════════════════════════════════
const EVENT_DEFS = [
  {
    id: 'shake',
    label: '📳 SHAKE',
    apply() {
      let t = 0;
      const id = setInterval(() => {
        const x = (Math.random() - 0.5) * 12;
        const y = (Math.random() - 0.5) * 12;
        wrapper.style.transform = `translate(${x}px,${y}px)`;
        if (++t > 40) { wrapper.style.transform = ''; removeEvent('shake'); clearInterval(id); }
      }, 40);
      active_events['shake'] = id;
    },
    remove() { wrapper.style.transform = ''; }
  },
  {
    id: 'flip_h',
    label: '↔️ FLIP H',
    apply() {
      canvas.style.transform = (canvas.style.transform || '') + ' scaleX(-1)';
    },
    remove() {
      canvas.style.transform = canvas.style.transform.replace(' scaleX(-1)', '').replace('scaleX(-1)', '');
    }
  },
  {
    id: 'flip_v',
    label: '↕️ FLIP V',
    apply() {
      canvas.style.transform = (canvas.style.transform || '') + ' scaleY(-1)';
    },
    remove() {
      canvas.style.transform = canvas.style.transform.replace(' scaleY(-1)', '').replace('scaleY(-1)', '');
    }
  },
  {
    id: 'rotate',
    label: '🔄 ROTATE',
    apply() {
      canvas.style.transform = (canvas.style.transform || '') + ' rotate(180deg)';
    },
    remove() {
      canvas.style.transform = canvas.style.transform.replace(' rotate(180deg)', '').replace('rotate(180deg)', '');
    }
  },
  {
    id: 'dark',
    label: '🌑 LIGHTS OUT',
    apply() {
      canvas.style.filter = 'brightness(0.15)';
    },
    remove() {
      canvas.style.filter = '';
    }
  },
  {
    id: 'shrink',
    label: '🔬 SHRINK',
    apply() {
      canvas.style.transform = (canvas.style.transform || '') + ' scale(0.55)';
    },
    remove() {
      canvas.style.transform = canvas.style.transform.replace(' scale(0.55)', '').replace('scale(0.55)', '');
    }
  },
  {
    id: 'stripe',
    label: '🟫 STRIPES',
    apply() {
      const el = document.createElement('div');
      el.id = 'ev-stripe';
      el.style.cssText = `
        position:absolute; inset:0; pointer-events:none; z-index:5;
        background: repeating-linear-gradient(
          45deg,
          rgba(0,0,0,0.35) 0px, rgba(0,0,0,0.35) 8px,
          transparent 8px, transparent 16px
        );
      `;
      wrapper.appendChild(el);
    },
    remove() {
      document.getElementById('ev-stripe')?.remove();
    }
  },
  {
    id: 'emoji_rain',
    label: '🎉 PARTY',
    apply() {
      const emojis = ['⭐','🌈','💥','🎊','🦄'];
      const els = [];
      for (let i = 0; i < 7; i++) {
        const el = document.createElement('div');
        el.className = 'overlay-element';
        el.textContent = emojis[Math.floor(Math.random() * emojis.length)];
        el.style.cssText = `
          font-size: ${20 + Math.random()*24}px;
          left: ${Math.random()*320}px;
          top: ${Math.random()*320}px;
          opacity: 0.65;
          animation: float_${i} 2s ease-in-out infinite alternate;
          transform: rotate(${(Math.random()-0.5)*30}deg);
        `;
        wrapper.appendChild(el);
        els.push(el);
      }
      active_events['emoji_rain_els'] = els;
    },
    remove() {
      (active_events['emoji_rain_els'] || []).forEach(e => e.remove());
      delete active_events['emoji_rain_els'];
    }
  },
  {
    id: 'blur',
    label: '😵 BLUR',
    apply() {
      canvas.style.filter = (canvas.style.filter || '') + ' blur(3px)';
    },
    remove() {
      canvas.style.filter = canvas.style.filter.replace(/ ?blur\(3px\)/, '').replace(/^blur\(3px\)/, '');
    }
  }
];

const eventMap = {};
EVENT_DEFS.forEach(e => eventMap[e.id] = e);

let activeEventIds = [];

function addRandomEvent() {
  const available = EVENT_DEFS.filter(e => !activeEventIds.includes(e.id));
  if (available.length === 0) return;
  const ev = available[Math.floor(Math.random() * available.length)];
  activeEventIds.push(ev.id);
  ev.apply();
  addBadge(ev.label);
  updateActiveEvDisplay();
}

function removeEvent(id) {
  const idx = activeEventIds.indexOf(id);
  if (idx !== -1) activeEventIds.splice(idx, 1);
  eventMap[id]?.remove();
  updateActiveEvDisplay();
}

function resetAllEvents() {
  [...activeEventIds].forEach(id => {
    eventMap[id]?.remove();
  });
  activeEventIds = [];
  canvas.style.transform = '';
  canvas.style.filter    = '';
  wrapper.style.transform = '';
  tickerEl.innerHTML = '';
  // Remove any overlay elements
  wrapper.querySelectorAll('.overlay-element, #ev-stripe').forEach(e => e.remove());
  updateActiveEvDisplay();
}

function addBadge(label) {
  const b = document.createElement('div');
  b.className = 'event-badge';
  b.textContent = label;
  tickerEl.appendChild(b);
}

function updateActiveEvDisplay() {
  activeEvEl.textContent = activeEventIds.length > 0 ? `${activeEventIds.length} EVENT${activeEventIds.length>1?'S':''}` : '';
}

function scheduleNextEvent() {
  clearTimeout(event_id);
  const delay = 3000 + Math.random() * 3000;
  event_id = setTimeout(() => {
    if (is_running && !correct_flag) {
      addRandomEvent();
      scheduleNextEvent();
    }
  }, delay);
}

// ═══════════════════════════════════════════════════════
// GAME FLOW
// ═══════════════════════════════════════════════════════
let usedWords = [];

function pickWord() {
  let pool = WORDS.filter(w => !usedWords.includes(w));
  if (pool.length === 0) { usedWords = []; pool = [...WORDS]; }
  const w = pool[Math.floor(Math.random() * pool.length)];
  usedWords.push(w);
  return w;
}

function startRound() {
  correct_flag = false;
  current_word = pickWord();
  wordEl.textContent = current_word.toUpperCase();
  wordEl.style.animation = 'none';
  void wordEl.offsetWidth;
  wordEl.style.animation = '';

  clearCanvas();
  guessEl.textContent = '...';
  guessEl.className   = '';
  confEl.textContent  = '';
  border.style.borderColor = 'var(--neon-cyan)';
  border.style.boxShadow   = '0 0 10px var(--neon-cyan)';
  resetAllEvents();

  clearInterval(predict_id);
  predict_id = setInterval(predict, PREDICT_INTERVAL);

  startTimer();
  scheduleNextEvent();
}

function onCorrect() {
  // Stop timer & events
  clearInterval(timer_id);
  clearTimeout(event_id);
  clearInterval(predict_id);
  is_running = false;

  score += Math.ceil(time_left * 10);
  scoreEl.textContent = score;

  flash('green');
  resetAllEvents();

  // Reset life timer for next round (life_index doesn't advance on success)
  setTimeout(() => {
    is_running = true;
    startRound();
  }, 1200);
}

function onTimeUp() {
  clearTimeout(event_id);
  clearInterval(predict_id);
  is_running = false;

  lives--;
  life_index = Math.min(life_index + 1, 2);
  updateLives();
  flash('red');
  resetAllEvents();

  if (lives <= 0) {
    setTimeout(showGameOver, 600);
    return;
  }

  setTimeout(() => {
    is_running = true;
    startRound();
  }, 1200);
}

function updateLives() {
  livesEls.forEach((el, i) => {
    el.classList.toggle('lost', i >= lives);
  });
}

function flash(type) {
  flashEl.className = type;
  flashEl.style.opacity = '1';
  setTimeout(() => { flashEl.style.opacity = '0'; flashEl.className = ''; }, 200);
}

// ═══════════════════════════════════════════════════════
// START / GAME OVER
// ═══════════════════════════════════════════════════════
function startGame() {
  lives      = 3;
  score      = 0;
  life_index = 0;
  usedWords  = [];
  scoreEl.textContent = 0;
  updateLives();
  overlay.classList.add('hidden');
  is_running = true;
  startRound();
}

function showGameOver() {
  is_running = false;
  clearInterval(timer_id);
  clearInterval(predict_id);
  clearTimeout(event_id);

  overlayTitle.innerHTML = `<span style="color:var(--neon-pink)">GAME OVER</span><br><span style="font-size:12px;color:#888">Score final</span>`;
  overlaySub.innerHTML   = `<span style="font-size:24px;color:var(--neon-yellow);font-family:'Press Start 2P',monospace">${score}</span><br><br>Tu as deviné <b>${usedWords.length - (lives > 0 ? 0 : 1)}</b> mots !`;
  overlayBtn.textContent = 'REJOUER';
  overlay.classList.remove('hidden');
}

overlayBtn.addEventListener('click', startGame);

// ═══════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════
clearCanvas();