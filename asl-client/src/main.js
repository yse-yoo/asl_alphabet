import {
    FilesetResolver,
    PoseLandmarker,
    HandLandmarker
} from "@mediapipe/tasks-vision";

// ===============================
// 要素
// ===============================
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");


let lastPredictTime = 0;
// 10fps の間隔（ミリ秒）
const PREDICT_INTERVAL = 100;

let poseLandmarker;
let handLandmarker;

let lastPose = null;
let lastHands = null;

// ===============================
// ✅ WebSocket
// ===============================
const ws = new WebSocket("ws://localhost:8000/ws");
let allowSend = true;

ws.onmessage = (msg) => {
    const r = JSON.parse(msg.data);
    if (r.ready) {
        document.getElementById("res-class").textContent =
            `${r.label} (${(r.prob * 100).toFixed(1)}%)`;
    }
    allowSend = true;
};

// ===============================
// ✅ カメラ
// ===============================
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 }
    });

    video.srcObject = stream;
    await video.play();

    // canvas のサイズを video と揃える
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// ===============================
// ✅ MediaPipe Tasks モデル
// ===============================
async function setupModels() {

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    const POSE_MODEL =
        "https://storage.googleapis.com/mediapipe-tasks/pose_landmarker/pose_landmarker_full.task";

    const HAND_MODEL =
        "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task";

    // ✅ Pose Landmarker
    poseLandmarker = await PoseLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath: POSE_MODEL
            },
            runningMode: "video",
            minPoseDetectionConfidence: 0.2,
            minPosePresenceConfidence: 0.2,
            minTrackingConfidence: 0.2
        }
    );

    // ✅ Hand Landmarker
    handLandmarker = await HandLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath: HAND_MODEL
            },
            runningMode: "video",
            numHands: 2,
            minHandDetectionConfidence: 0.3,
            minTrackingConfidence: 0.3,
        }
    );
}

// ===============================
// ✅ 225次元抽出（Python と完全一致）
// ===============================
function extract225(pose, hands) {
    const POSE_DIM = 33 * 3;
    const HANDS_DIM = 21 * 2 * 3;
    const HAND_WEIGHT = 1.5;
    const Z_SCALE = 0.3;

    let poseArr = [];
    let handArr = [];

    if (pose?.keypoints) {
        pose.keypoints.forEach(pt => {
            poseArr.push(pt.x / video.videoWidth);
            poseArr.push(pt.y / video.videoHeight);
            poseArr.push((pt.z || 0) * Z_SCALE);
        });
    }
    while (poseArr.length < POSE_DIM) poseArr.push(0);

    hands.forEach(hand => {
        hand.keypoints.forEach(pt => {
            handArr.push(pt.x / video.videoWidth);
            handArr.push(pt.y / video.videoHeight);
            handArr.push((pt.z || 0) * Z_SCALE);
        });
    });
    while (handArr.length < HANDS_DIM) handArr.push(0);

    handArr = handArr.map(v => v * HAND_WEIGHT);

    return [...poseArr, ...handArr];
}

// ===============================
// ✅ 描画
// ===============================
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Pose (cyan)
    if (lastPose?.landmarks?.length > 0) {
        const poseLm = lastPose.landmarks[0];  // 33点のセット

        poseLm.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 4, 0, 2 * Math.PI);
            ctx.fillStyle = "cyan";
            ctx.fill();
        });
    }

    // Hands (red)
    if (lastHands?.landmarks) {
        lastHands.landmarks.forEach(hand => {
            hand.forEach(pt => {
                ctx.beginPath();
                ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 3, 0, 2 * Math.PI);
                ctx.fillStyle = "red";
                ctx.fill();
            });
        });
    }
}

// ===============================
// ✅ 推論ループ
// ===============================
async function loop() {
    const now = performance.now();

    // 60FPSで画面描画（軽いからOK）
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // ✅ ここで「必要なときだけ推論」する
    if (now - lastPredictTime > PREDICT_INTERVAL) {

        // Pose 推論
        lastPose = poseLandmarker.detectForVideo(canvas, now);

        // Hand 推論
        lastHands = handLandmarker.detectForVideo(video, now);

        // WebSocket送信
        if (allowSend && ws.readyState === WebSocket.OPEN) {
            const lm225 = extract225(poses[0] || null, hands || []);
            ws.send(JSON.stringify({ landmark: lm225 }));
            // 次の返信が来るまで送信しない
            allowSend = false;   
        }
        lastPredictTime = now;
    }

    // 描画（軽い）
    draw();

    requestAnimationFrame(loop);
}

// ===============================
// ✅ メイン
// ===============================
(async function main() {
    await setupCamera();
    await setupModels();
    loop();
})();
