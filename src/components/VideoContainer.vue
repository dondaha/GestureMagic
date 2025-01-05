<template>
    <div class="right">
        <div class="camera-container">
            <!-- 视频流显示 -->
            <video ref="video" class="camera-feed" autoplay></video>
            <!-- 画布用于绘制手势识别结果 -->
            <canvas ref="canvas" class="camera-overlay"></canvas>
            <!-- 画布用于绘制叠加在人脸上的图像 -->
            <canvas ref="imgCanvas" class="camera-overlay"></canvas>
        </div>
    </div>
</template>

<script>
import {
    GestureRecognizer,
    FilesetResolver,
    DrawingUtils,
    FaceDetector
} from "@mediapipe/tasks-vision";

export default {
    data() {
        return {
            drawUtilsLoaded: false, // drawing_utils 是否已加载
            numHands: 1, // 识别的人数
            runningMode: "VIDEO", // 运行模式
            webcamRunning: true, // 摄像头是否正在运行
            results: undefined, // 识别结果
            lasthandState: undefined, // 上一帧的手势状态
            handState: undefined, // 手势状态："PAINT", "ERASE" 或 "NONE" 分别是画笔、清屏和无状态
            pics: undefined, // 用来显示在人脸上的图像类型（'hat','mustache','glasses','nose', undefined）
            trajectory: [], // 记录轨迹的数组
            faceDetector: undefined, // 人脸识别器
            faceResults: undefined, // 人脸识别结果
        }
    },
    mounted() {
        this.loadDrawingUtils();
        this.createGestureLandmarker();
        this.createFaceDetector();
        this.enableCam();
    },
    methods: {
        async loadDrawingUtils() {
            const script = document.createElement("script");
            script.src = "/GestureMagic/drawing_utils.js";
            script.onload = () => {
                this.drawUtilsLoaded = true;
            };
            document.head.appendChild(script);
        },
        async createGestureLandmarker() {
            const vision = await FilesetResolver.forVisionTasks(
                "/GestureMagic/wasm"
            );
            this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `/GestureMagic/gesture_recognizer.task`,
                    delegate: "GPU"
                },
                runningMode: this.runningMode,
                numHands: this.numHands
            });
        },
        async createFaceDetector() {
            const vision = await FilesetResolver.forVisionTasks(
                "/GestureMagic/wasm"
            );
            this.faceDetector = await FaceDetector.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `/GestureMagic/blaze_face_short_range.tflite`,
                    delegate: "GPU"
                },
                runningMode: this.runningMode
            });
        },
        enableCam() {
            const constraints = {
                video: true
            };
            navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
                this.$refs.video.srcObject = stream;
                this.$refs.video.addEventListener("loadeddata", this.predictWebcam);
            });
        },
        async predictWebcam() {
            if (!this.drawUtilsLoaded || !this.gestureRecognizer || !this.faceDetector) {
                requestAnimationFrame(this.predictWebcam);
                return;
            }

            const video = this.$refs.video;
            // 可视化画布
            const canvas = this.$refs.canvas;
            const canvasCtx = canvas.getContext("2d");
            // 贴图画布
            const imgCanvas = this.$refs.imgCanvas;
            const imgCanvasCtx = imgCanvas.getContext("2d");

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            imgCanvas.width = video.videoWidth;
            imgCanvas.height = video.videoHeight;

            if (this.runningMode === "IMAGE") {
                this.runningMode = "VIDEO";
                await this.gestureRecognizer.setOptions({ runningMode: "VIDEO" });
                await this.faceDetector.setOptions({ runningMode: "VIDEO" });
            }

            const startTimeMs = performance.now();
            if (this.lastVideoTime !== video.currentTime) {
                this.lastVideoTime = video.currentTime;

                // 创建一个临时canvas用于水平翻转视频帧
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;

                // 水平翻转视频帧
                tempCtx.translate(tempCanvas.width, 0);
                tempCtx.scale(-1, 1);
                tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

                // 获取翻转后的图像数据
                const flippedImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);

                // 将翻转后的图像数据传递给gestureRecognizer进行检测
                this.results = await this.gestureRecognizer.recognizeForVideo(flippedImageData, startTimeMs);
                // 将翻转后的图像数据传递给faceDetector进行检测
                this.faceResults = await this.faceDetector.detectForVideo(flippedImageData, startTimeMs);
            }

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height); // 清空画布

            // 应用水平翻转变换
            canvasCtx.translate(canvas.width, 0);
            canvasCtx.scale(-1, 1);
            canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // 绘制Pose关键点和连接线
            if (this.results.landmarks) {
                this.drawGestureResults(canvasCtx);
                this.poseDetected(this.results);
            }

            // 绘制人脸关键点
            if (this.faceResults && this.faceResults.detections) {
                this.drawFaceDetections(canvasCtx, this.faceResults.detections);
            }

            canvasCtx.restore(); // 恢复上下文状态

            // Call this function again to keep predicting when the browser is ready.
            if (this.webcamRunning) {
                window.requestAnimationFrame(this.predictWebcam);
            }
        },
        drawTrajectory(ctx) {
            if (this.trajectory.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = "white";
            ctx.lineWidth = 5;
            for (let i = 1; i < this.trajectory.length; i++) {
                ctx.moveTo(this.trajectory[i - 1].x * ctx.canvas.width, this.trajectory[i - 1].y * ctx.canvas.height);
                ctx.lineTo(this.trajectory[i].x * ctx.canvas.width, this.trajectory[i].y * ctx.canvas.height);
            }
            ctx.stroke(); // 绘制轨迹
        },
        drawGestureResults(ctx) {
            const drawingUtils = new DrawingUtils(ctx);
            for (const landmarks of this.results.landmarks) {
                // 水平翻转关键点
                const mirroredLandmarks = landmarks.map(point => ({
                    ...point,
                    x: 1 - point.x
                }));
                drawingUtils.drawConnectors(mirroredLandmarks, GestureRecognizer.HAND_CONNECTIONS, {
                    color: "#00FF00",
                    lineWidth: 5
                });
                drawingUtils.drawLandmarks(mirroredLandmarks, {
                    color: "#FF0000",
                    lineWidth: 1
                });

                if (this.handState === "PAINT") {
                    const thumbTip = mirroredLandmarks[4];
                    const indexTip = mirroredLandmarks[8];
                    const midPoint = {
                        x: (thumbTip.x + indexTip.x) / 2,
                        y: (thumbTip.y + indexTip.y) / 2
                    };
                    this.trajectory.push(midPoint);
                    this.drawTrajectory(ctx);
                }
            }
        },
        drawFaceDetections(ctx, detections) {
            for (const detection of detections) {
                // const keypoints = detection.keypoints;
                // 水平翻转关键点
                const keypoints = detection.keypoints.map(point => ({
                    ...point,
                    x: 1 - point.x
                }));
                const boundingBox = detection.boundingBox;
                // 尝试放一个红色块
                // ctx.fillStyle = "red";
                // ctx.fillRect(
                //     200, 200, 40, 40
                // );
                // 绘制关键点和标号
                ctx.fillStyle = "red";
                ctx.font = "10px Arial";
                keypoints.forEach((keypoint, index) => {
                    const x = keypoint.x * ctx.canvas.width;
                    const y = keypoint.y * ctx.canvas.height;
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.fillText(index, x + 5, y - 5);
                });
            }
        },
        poseDetected(results) {
            this.inferState(results);
            if (this.handState === "NONE" || this.handState === "ERASE") {
                if (this.handState === "ERASE") {
                    this.pics = undefined;
                } else {
                    if (this.lasthandState === "PAINT") {
                        this.recognizeTrajectory(this.trajectory);
                    }
                }
                this.trajectory = [];
            }
        },
        inferState(results) {
            // 识别手势状态
            if (results.landmarks[0]) {
                this.lasthandState = this.handState;
                // 找到results.landmarks[0][i]的最大最小值
                let minX = results.landmarks[0][0].x;
                let maxX = results.landmarks[0][0].x;
                let minY = results.landmarks[0][0].y;
                let maxY = results.landmarks[0][0].y;
                for (let i = 1; i < results.landmarks[0].length; i++) {
                    if (results.landmarks[0][i].x < minX) {
                        minX = results.landmarks[0][i].x;
                    }
                    if (results.landmarks[0][i].x > maxX) {
                        maxX = results.landmarks[0][i].x;
                    }
                    if (results.landmarks[0][i].y < minY) {
                        minY = results.landmarks[0][i].y;
                    }
                    if (results.landmarks[0][i].y > maxY) {
                        maxY = results.landmarks[0][i].y;
                    }
                }
                let x1 = results.landmarks[0][4].x;
                let y1 = results.landmarks[0][4].y;
                let x2 = results.landmarks[0][8].x;
                let y2 = results.landmarks[0][8].y;
                let dis = Math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
                if (dis / (maxX - minX + maxY - minY) < 0.15) {
                    this.handState = "PAINT";
                } else {
                    if (this.calculateAngle(results.landmarks[0][8], results.landmarks[0][7], results.landmarks[0][6], results.landmarks[0][5]) > 60 &&
                        this.calculateAngle(results.landmarks[0][12], results.landmarks[0][11], results.landmarks[0][10], results.landmarks[0][9]) > 60 &&
                        this.calculateAngle(results.landmarks[0][16], results.landmarks[0][15], results.landmarks[0][14], results.landmarks[0][13]) > 60 &&
                        this.calculateAngle(results.landmarks[0][20], results.landmarks[0][19], results.landmarks[0][18], results.landmarks[0][17]) > 60) {
                        this.handState = "ERASE";
                    } else {
                        this.handState = "NONE";
                    }
                }
                // console.log(this.handState);
            }
        },
        calculateAngle(p1, p2, p3, p4) {
            // 计算P2P1和P3P4的夹角
            let x1 = p1.x - p2.x;
            let y1 = p1.y - p2.y;
            let z1 = p1.z - p2.z;
            let x2 = p3.x - p4.x;
            let y2 = p3.y - p4.y;
            let z2 = p3.z - p4.z;
            let cos = (x1 * x2 + y1 * y2 + z1 * z2) / (Math.sqrt(x1 * x1 + y1 * y1 + z1 * z1) * Math.sqrt(x2 * x2 + y2 * y2 + z2 * z2));
            return Math.acos(cos) * 180 / Math.PI;
        },
        recognizeTrajectory(trajectory) {
            // 调用KNN识别函数，暂时不写完整
            const recognizedType = this.knnRecognize(trajectory);
            if (recognizedType) {
                this.pics = recognizedType;
                this.applySticker(recognizedType);
            }
        },
        knnRecognize(trajectory) {
            // KNN识别逻辑，暂时不写完整
            return ['hat', 'mustache', 'glasses', 'nose'][Math.floor(Math.random() * 4)];
        },
        applySticker(type) {
            // 随机选择一个贴图
            const randomIndex = Math.floor(Math.random() * 3) + 1;
            const imgPath = `/GestureMagic/imgs/${type}${randomIndex}.png`;
            // 贴图逻辑，暂时不写完整
            // console.log(`Applying sticker: ${imgPath}`);
        }
    }
}

</script>

<style>
.right {
    position: absolute;
    top: 25%;
    right: 12%;
    width: 45%;
    height: 68%;
    background-color: #c0e3ff;
    border-radius: 5%;
    overflow: hidden;
    /* 内部board */
    box-shadow: 0 0 10px 5px rgba(0, 0, 0, 0.1);
}

.camera-container {
    position: relative;
    top: 0%;
    left: 0%;
    width: 100%;
    height: 100%;
}

.camera-feed {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
    /* 居中 */
    margin: auto;
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
}

.camera-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
</style>