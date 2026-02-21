// server.js — OSC→WebSocket bridge for RALF Sound Test
//
// Receives OSC from Gesture Studio on UDP port 12000,
// serves the browser audio engine, and bridges gestures via WebSocket.
//
// Usage: npm start  (then open http://localhost:8080)

const dgram = require("dgram");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { WebSocketServer } = require("ws");

const OSC_PORT = 12000;
const HTTP_PORT = 8080;

// =============================================================================
// Minimal OSC parser (handles /gesture/N with float arg)
// =============================================================================

function parseOsc(buf) {
  // Read null-terminated string, padded to 4-byte boundary
  function readString(offset) {
    let end = offset;
    while (end < buf.length && buf[end] !== 0) end++;
    const str = buf.toString("ascii", offset, end);
    // Advance past null + padding to 4-byte boundary
    const padded = end + 1;
    return { value: str, next: padded + (4 - (padded % 4)) % 4 };
  }

  const addr = readString(0);
  if (!addr.value.startsWith("/")) return null;

  const typetag = readString(addr.next);
  // We only care about the address — the float arg is always 1.0 for hits
  return { address: addr.value };
}

// =============================================================================
// HTTP server — serves index.html
// =============================================================================

const htmlPath = path.join(__dirname, "index.html");

const server = http.createServer((req, res) => {
  if (req.url === "/" || req.url === "/index.html") {
    res.writeHead(200, { "Content-Type": "text/html" });
    fs.createReadStream(htmlPath).pipe(res);
  } else {
    res.writeHead(404);
    res.end("Not found");
  }
});

// =============================================================================
// WebSocket server — bridges OSC to browser
// =============================================================================

const wss = new WebSocketServer({ server });
const clients = new Set();

wss.on("connection", (ws) => {
  clients.add(ws);
  console.log(`[ws] client connected (${clients.size} total)`);
  ws.on("close", () => {
    clients.delete(ws);
    console.log(`[ws] client disconnected (${clients.size} total)`);
  });
});

function broadcast(data) {
  const msg = JSON.stringify(data);
  for (const ws of clients) {
    if (ws.readyState === 1) ws.send(msg);
  }
}

// =============================================================================
// UDP/OSC receiver
// =============================================================================

const udp = dgram.createSocket("udp4");

udp.on("message", (buf) => {
  const msg = parseOsc(buf);
  if (msg && msg.address.startsWith("/gesture/")) {
    console.log(`[osc] ${msg.address}`);
    broadcast({ type: "gesture", address: msg.address });
  }
});

udp.on("error", (err) => {
  console.error(`[osc] error: ${err.message}`);
  if (err.code === "EADDRINUSE") {
    console.error(
      `[osc] Port ${OSC_PORT} is in use. Is Ableton/Max running? Close it or change the port.`
    );
  }
});

// =============================================================================
// Start
// =============================================================================

udp.bind(OSC_PORT, () => {
  console.log(`[osc] listening on UDP port ${OSC_PORT}`);
});

server.listen(HTTP_PORT, () => {
  console.log(`[http] serving at http://localhost:${HTTP_PORT}`);
  console.log(`\nRALF Sound Test ready.`);
  console.log(`  1. Start Gesture Studio (output: 127.0.0.1:${OSC_PORT})`);
  console.log(`  2. Open http://localhost:${HTTP_PORT} in your browser`);
  console.log(`  3. Click "Start" and perform gestures\n`);
});
