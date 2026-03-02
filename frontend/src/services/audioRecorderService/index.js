let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let zeroGainNode = null;
let buffers = [];

function flattenFloat32(chunks) {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i += 1) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate) {
  const pcm16 = floatTo16BitPCM(samples);
  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcm16.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, pcm16.length * 2, true);

  let offset = 44;
  for (let i = 0; i < pcm16.length; i += 1) {
    view.setInt16(offset, pcm16[i], true);
    offset += 2;
  }
  return buffer;
}

export function createAnalyser(fftSize = 128) {
  if (!audioContext || !sourceNode) return null;
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = fftSize;
  sourceNode.connect(analyser);
  return analyser;
}

export async function startRecording() {
  if (audioContext) return;

  buffers = [];
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  zeroGainNode = audioContext.createGain();
  zeroGainNode.gain.value = 0;

  processorNode.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    buffers.push(new Float32Array(input));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(zeroGainNode);
  zeroGainNode.connect(audioContext.destination);
}

export async function stopRecording() {
  if (!audioContext) return { blob: null };

  processorNode.disconnect();
  sourceNode.disconnect();
  zeroGainNode.disconnect();

  const tracks = mediaStream?.getTracks?.() || [];
  tracks.forEach((t) => t.stop());

  const sr = audioContext.sampleRate || 48000;
  await audioContext.close();

  audioContext = null;
  mediaStream = null;
  sourceNode = null;
  processorNode = null;
  zeroGainNode = null;

  const samples = flattenFloat32(buffers);
  buffers = [];

  const wav = encodeWAV(samples, sr);
  const blob = new Blob([wav], { type: "audio/wav" });
  return { blob, sampleRate: sr, numSamples: samples.length };
}
