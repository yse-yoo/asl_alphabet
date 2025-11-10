// ===============================
// 初期設定
// ===============================
const video = document.getElementById("webcam");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

const SEND_INTERVAL = 100;

const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const T = 40;

let detectorHands, detectorPose;

// ===============================
// ✅ WebSocket (Backpressure あり)
// ===============================
const ws = new WebSocket("ws://localhost:8000/ws");
let allowSend = true;

ws.onopen = () => console.log("✅ WebSocket connected");

ws.onmessage = (msg) => {
    const res = JSON.parse(msg.data);
    if (res.ready) {
        document.getElementById("res-class").textContent =
            `${res.label} (${(res.prob * 100).toFixed(1)}%)`;
    }
    allowSend = true;  // ✅ 返信が来たので次の送信を許可
};

// ===============================
// ✅ カメラ
// ===============================
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
    });

    video.srcObject = stream;
    await video.play();

    // 内部ピクセルサイズ
    canvas.width  = VIDEO_WIDTH;
    canvas.height = VIDEO_HEIGHT;

    // ✅ 見た目のサイズを video と完全一致させる（ズレ防止）
    canvas.style.width  = VIDEO_WIDTH + "px";
    canvas.style.height = VIDEO_HEIGHT + "px";
}

// ===============================
// ✅ MediaPipe Models
// ===============================
async function setupModels() {
    const modelHands = handPoseDetection.SupportedModels.MediaPipeHands;
    detectorHands = await handPoseDetection.createDetector(modelHands, {
        runtime: "mediapipe",
        modelType: "full",
        maxHands: 2,
        solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands"
    });

    const modelPose = poseDetection.SupportedModels.MoveNet;
    detectorPose = await poseDetection.createDetector(modelPose, {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    });
}

// ===============================
// ✅ 225次元変換
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
// ✅ 推論ループ（FPS 自動制御）
// ===============================
async function detect() {
    const hands = await detectorHands.estimateHands(video);
    const poses = await detectorPose.estimatePoses(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 骨格描画
    if (poses.length > 0) {
        poses[0].keypoints.forEach(pt => {
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

    // ✅ サーバから返事が来た時だけ送信
    if (allowSend && ws.readyState === WebSocket.OPEN) {
        const lm225 = extract225(poses[0] || null, hands || []);
        ws.send(JSON.stringify({ landmark: lm225 }));
        allowSend = false;   // 次の返信が来るまで送信しない
    }

    requestAnimationFrame(detect);
}

// ===============================
// ✅ メイン
// ===============================
async function main() {
    await setupCamera();
    await setupModels();
    detect();
}

main();