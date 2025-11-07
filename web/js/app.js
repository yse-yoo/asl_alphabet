// ===============================
// 初期設定
// ===============================
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const T = 40;
const POSE_DIM = 33 * 3;
const HANDS_DIM = 21 * 2 * 3;
const LAND_DIM = POSE_DIM + HANDS_DIM;
const HAND_WEIGHT = 1.5;
const Z_SCALE = 0.3;

// ===============================
// ✅ WebSocket 接続
// ===============================
const ws = new WebSocket("ws://localhost:8000/ws");

// 受信（FastAPI → Web）
ws.onmessage = (msg) => {
    const res = JSON.parse(msg.data);
    if (res.ready) {
        document.getElementById("res-class").textContent =
            `${res.label} (${(res.prob * 100).toFixed(1)}%)`;
    }
};

ws.onopen = () => console.log("✅ WebSocket connected");
ws.onerror = (e) => console.error("WS Error:", e);

// ===============================
// モデル（TF.js）
// ===============================
let detectorHands, detectorPose;

async function setupModels() {
    // ---- 1) Hand Pose Detection (MediaPipe Hands) ----
    const modelHands = handPoseDetection.SupportedModels.MediaPipeHands;
    detectorHands = await handPoseDetection.createDetector(modelHands, {
        runtime: "mediapipe",
        modelType: "full",
        maxHands: 2,
        solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    });

    // ---- 2) Pose (MoveNet) ----
    const modelPose = poseDetection.SupportedModels.MoveNet;
    detectorPose = await poseDetection.createDetector(modelPose, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    });
}

// ===============================
// カメラ
// ===============================
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// ===============================
// ランドマーク → 225次元
// ===============================
function extract225(pose, hands) {
    let poseArr = [];
    let handArr = [];

    // ---- pose ----
    if (pose?.keypoints) {
        pose.keypoints.forEach((pt) => {
            poseArr.push(pt.x / video.videoWidth);
            poseArr.push(pt.y / video.videoHeight);
            poseArr.push((pt.z || 0) * Z_SCALE);
        });
    }
    while (poseArr.length < POSE_DIM) poseArr.push(0);

    // ---- hands ----
    hands.forEach(hand => {
        hand.keypoints.forEach(pt => {
            handArr.push(pt.x / video.videoWidth);
            handArr.push(pt.y / video.videoHeight);
            handArr.push((pt.z || 0) * Z_SCALE);
        });
    });
    while (handArr.length < HANDS_DIM) handArr.push(0);

    // 重みづけ
    handArr = handArr.map(v => v * HAND_WEIGHT);

    return [...poseArr.slice(0, POSE_DIM), ...handArr.slice(0, HANDS_DIM)];
}

// ===============================
// 推論ループ
// ===============================
async function detect() {
    const hands = await detectorHands.estimateHands(video);
    const poses = await detectorPose.estimatePoses(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 骨格描画
    if (poses.length > 0) {
        const kp = poses[0].keypoints;
        kp.forEach(pt => {
            if (pt.score > 0.4) {
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = "cyan";
                ctx.fill();
            }
        });
    }

    hands.forEach(hand => {
        hand.keypoints.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = "red";
            ctx.fill();
        });
    });

    // ---- WebSocket 送信 ----
    if (ws.readyState === WebSocket.OPEN) {
        const lm225 = extract225(poses[0] || null, hands || []);
        ws.send(JSON.stringify({ landmark: lm225 }));
    }

    requestAnimationFrame(detect);
}

// ===============================
// メイン
// ===============================
async function main() {
    await setupCamera();
    await setupModels();
    detect();
}

main();
