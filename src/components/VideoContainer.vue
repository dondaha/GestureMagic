<template>
    <div class="right">
        <div class="camera-container">
            <!-- 视频流显示 -->
            <video ref="video" class="camera-feed" autoplay></video>
            <!-- 画布用于绘制手势识别结果 -->
            <canvas ref="canvas" class="camera-overlay"></canvas>
        </div>
    </div>
</template>

<script>
import {
    GestureRecognizer,
    FilesetResolver,
    DrawingUtils
} from "@mediapipe/tasks-vision";

export default {
    data() {
        return {
            drawUtilsLoaded: false, // drawing_utils 是否已加载
            numHands: 1, // 识别的人数
            runningMode: "VIDEO", // 运行模式
            webcamRunning: true, // 摄像头是否正在运行
            results: undefined, // 识别结果
            handState: undefined, // 手势状态："PAINT", "ERASE" 或 "NONE" 分别是画笔、清屏和无状态
        }
    },
    mounted() {
        this.loadDrawingUtils();
        this.createGestureLandmarker();
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
            if (!this.drawUtilsLoaded || !this.gestureRecognizer) {
                requestAnimationFrame(this.predictWebcam);
                return;
            }

            const video = this.$refs.video;
            const canvas = this.$refs.canvas;
            const canvasCtx = canvas.getContext("2d");

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            if (this.runningMode === "IMAGE") {
                this.runningMode = "VIDEO";
                await this.gestureRecognizer.setOptions({ runningMode: "VIDEO" });
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
            }

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

            // 应用水平翻转变换
            canvasCtx.translate(canvas.width, 0);
            canvasCtx.scale(-1, 1);
            canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // 绘制Pose关键点和连接线
            if (this.results.landmarks) {
                const drawingUtils = new DrawingUtils(canvasCtx);
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
                }
                this.poseDetected(this.results);
            }
            canvasCtx.restore(); // 恢复上下文状态

            // Call this function again to keep predicting when the browser is ready.
            if (this.webcamRunning) {
                window.requestAnimationFrame(this.predictWebcam);
            }
        },
        poseDetected(results) {
            // console.log("Pose detected:", results);
            this.inferState(results);
        },
        inferState(results) {
            // 识别手势状态
            if (results.landmarks[0]) {
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
                if (dis / (maxX - minX + maxY - minY) < 0.1 ){
                    this.handState = "PAINT";
                } else {
                    // 如果每个手指都伸直，则None
                    if (this.calculateAngle(results.landmarks[0][0],results.landmarks[0][1],results.landmarks[0][2],results.landmarks[0][3]) < 30 &&
                        this.calculateAngle(results.landmarks[0][0],results.landmarks[0][5],results.landmarks[0][6],results.landmarks[0][7]) < 30 &&
                        this.calculateAngle(results.landmarks[0][0],results.landmarks[0][9],results.landmarks[0][10],results.landmarks[0][11]) < 30 &&
                        this.calculateAngle(results.landmarks[0][0],results.landmarks[0][13],results.landmarks[0][14],results.landmarks[0][15]) < 30 &&
                        this.calculateAngle(results.landmarks[0][0],results.landmarks[0][17],results.landmarks[0][18],results.landmarks[0][19]) < 30){
                        this.handState = "NONE";
                    } else {
                        this.handState = "ERASE";
                    }
                }
                console.log(this.handState);
            }
        },
        calculateAngle(p1,p2,p3,p4){
            // 计算P2P1和P3P4的夹角
            let x1 = p1.x - p2.x;
            let y1 = p1.y - p2.y;
            let x2 = p3.x - p4.x;
            let y2 = p3.y - p4.y;
            let cos = (x1 * x2 + y1 * y2) / (Math.sqrt(x1 * x1 + y1 * y1) * Math.sqrt(x2 * x2 + y2 * y2));
            return Math.acos(cos) * 180 / Math.PI;
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