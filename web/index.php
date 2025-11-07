<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8">
    <title>ASL Realtime (WebCam + FastAPI)</title>
    <style>
        body {
            background: #111;
            color: #fff;
            font-family: sans-serif;
        }

        #video {
            transform: scaleX(-1);
        }

        #label {
            font-size: 32px;
            margin-top: 10px;
        }
    </style>

    <!-- ✅ MediaPipe CDN -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
</head>

<body>

    <h2>ASL Realtime (FastAPI)</h2>

    <video id="video" width="640" height="480" autoplay muted></video>
    <div id="label">...</div>

    <!-- ✅ defer OK -->
    <script src="js/app.js" defer></script>
</body>

</html>