const video = document.getElementById("video");
const labelDiv = document.getElementById("label");

// -----------------------------
// 推論設定
// -----------------------------
const POSE_DIM = 33 * 3;
const HANDS_DIM = 21 * 2 * 3;
const LAND_DIM = POSE_DIM + HANDS_DIM;
const HAND_WEIGHT = 1.5;
const Z_SCALE = 0.3;

// WebSocket
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (msg) => {
    const res = JSON.parse(msg.data);
    if (res.ready) {
        labelDiv.textContent = `${res.label} (${res.prob.toFixed(2)})`;
    }
};

// -----------------------------
// ✅ MediaPipe 初期化（CDN 版）
// -----------------------------

// Pose
const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
});
pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
});

// Hands
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
    maxNumHands: 2,
    minDetectionConfidence: 0.5,
});

// 結果受け取り
let lastPose = null;
let lastHands = null;

pose.onResults((res) => (lastPose = res));
hands.onResults((res) => (lastHands = res));

// -----------------------------
// ✅ Landmark → 225
// -----------------------------
function extract225(poseRes, handsRes) {
    const poseArr = [];
    const handArr = [];

    // pose
    if (poseRes?.poseLandmarks) {
        for (const lm of poseRes.poseLandmarks) {
            poseArr.push(lm.x, lm.y, lm.z * Z_SCALE);
        }
    }
    while (poseArr.length < POSE_DIM) poseArr.push(0);

    // hands
    if (handsRes?.multiHandLandmarks) {
        for (const hand of handsRes.multiHandLandmarks) {
            for (const lm of hand) {
                handArr.push(lm.x, lm.y, lm.z * Z_SCALE);
            }
        }
    }
    while (handArr.length < HANDS_DIM) handArr.push(0);

    // weight hands
    const weightedHands = handArr.map((v) => v * HAND_WEIGHT);

    return [...poseArr.slice(0, POSE_DIM), ...weightedHands.slice(0, HANDS_DIM)];
}

// -----------------------------
// ✅ Video + MediaPipe 実行
// -----------------------------
async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
    });
    video.srcObject = stream;

    const camera = new CameraUtils.Camera(video, {
        onFrame: async () => {
            await pose.send({ image: video });
            await hands.send({ image: video });
        },
    });

    camera.start();

    setInterval(() => {
        if (!lastPose || !lastHands) return;

        const vec225 = extract225(lastPose, lastHands);

        ws.send(
            JSON.stringify({
                landmark: vec225,
            })
        );
    }, 33); // 30 FPS
}

start();
