const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const sendBtn = document.getElementById("sendBtn");

let latestHands = [];
let latestPoses = [];
const margin = 100;

let detectorHands, detectorPose;
const videoWidth = 960;
const videoHeight = 680;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: videoWidth, height: videoHeight },
  });
  video.srcObject = stream;
  await video.play();

  video.width = video.videoWidth;
  video.height = video.videoHeight;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function setupModels() {
  // --- Hands モデル ---
  const modelHands = handPoseDetection.SupportedModels.MediaPipeHands;
  detectorHands = await handPoseDetection.createDetector(modelHands, {
    runtime: "mediapipe",
    modelType: "full",
    maxHands: 2,
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
  });

  // --- Pose モデル ---
  const modelPose = poseDetection.SupportedModels.MoveNet;
  detectorPose = await poseDetection.createDetector(modelPose, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
  });
}


async function detect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const hands = await detectorHands.estimateHands(video);
  const poses = await detectorPose.estimatePoses(video);

  // 最新の検出結果を保存  
  latestHands = hands;
  latestPoses = poses;

  // 骨格を描画
  ctx.strokeStyle = "cyan";
  ctx.lineWidth = 2;

  // --- Pose 描画 ---
  if (poses.length > 0) {
    const keypoints = poses[0].keypoints;
    keypoints.forEach((pt) => {
      if (pt.score > 0.4) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = "lime";
        ctx.fill();
      }
    });
  }

  // --- Hands 描画 ---
  if (hands.length > 0) {
    hands.forEach((hand) => {
      hand.keypoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
      });
    });
    sendBtn.disabled = false;
    sendBtn.classList.add("bg-blue-500");
  } else {
    sendBtn.disabled = true;
    sendBtn.classList.remove("bg-blue-500");
  }

  requestAnimationFrame(detect);
}

// 🔘 上半身ごと送信
function sendUpperBody() {
  const bbox = getBoundingBox(latestHands, latestPoses);
  if (!bbox) {
    alert("手または上半身が検出されていません。");
    return;
  }

  const { x, y, w, h } = bbox;

  const sendCanvas = document.createElement("canvas");
  const sendCtx = sendCanvas.getContext("2d");
  sendCanvas.width = w;
  sendCanvas.height = h;

  sendCtx.drawImage(video, x, y, w, h, 0, 0, w, h);

  sendCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("file", blob, "cropped.jpg");

    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (res.ok) {
      let message = `${data.label} (${(data.confidence * 100).toFixed(2)}%)`;
      document.getElementById("res-class").textContent = message;
      const imgEl = document.getElementById("res-image");
      imgEl.src = `${data.image_url}?t=${Date.now()}`;
      imgEl.classList.remove("hidden");

      speachText(data.label);
    }
  }, "image/jpeg");
}

function speachText(text) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  speechSynthesis.speak(utterance);
}

function getBoundingBox(hands, poses) {
  const xs = [];
  const ys = [];

  // --- Hands の keypoints 座標 ---
  hands.forEach(hand => {
    hand.keypoints.forEach(pt => {
      xs.push(pt.x);
      ys.push(pt.y);
    });
  });

  // --- Pose の keypoints 座標 ---
  poses.forEach(pose => {
    pose.keypoints.forEach(pt => {
      if (pt.score > 0.4) {
        xs.push(pt.x);
        ys.push(pt.y);
      }
    });
  });

  if (xs.length === 0 || ys.length === 0) return null;

  // bbox計算（Pythonのmin_x, max_x, ... と同じ考え方）
  const minX = Math.max(0, Math.min(...xs) - margin);
  const maxX = Math.min(video.videoWidth, Math.max(...xs) + margin);
  const minY = Math.max(0, Math.min(...ys) - margin);
  const maxY = Math.min(video.videoHeight, Math.max(...ys) + margin);

  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}


sendBtn.addEventListener("click", sendUpperBody);
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter") sendUpperBody();
});

async function main() {
  await setupCamera();
  await setupModels();
  detect();
}

main();