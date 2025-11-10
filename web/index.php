<!DOCTYPE html>
<html lang="ja">

<head>
    <meta charset="UTF-8" />
    <title>ASL Predictor with Skeleton</title>
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- TensorFlow.js + Hand Pose Detection -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/hand-pose-detection"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>
</head>

<body>
    <main class="bg-gray-100">
        <!-- 推論結果パネル -->
        <div id="inference"
            class="absolute top-0 right-0 bg-white bg-opacity-75 rounded shadow-lg z-50">
            <div id="result" class="text-sm font-mono text-gray-800 w-full">
                <p class="text-2xl font-bold text-center">
                    <span id="res-class" class="text-5xl"></span>
                </p>
                <img id="res-image" src="" alt="Uploaded hand"
                    class="mt-4 border rounded shadow w-full max-w-[100px] hidden mx-auto">
            </div>
        </div>
        <div>
            <video id="webcam" class="absolute top-0 left-0"></video>
            <canvas id="outputCanvas" class="absolute top-0 left-0"></canvas>
        </div>
    </main>

    <!-- 外部JS -->
    <script type="module" src="js/app.js"></script>
</body>

</html>