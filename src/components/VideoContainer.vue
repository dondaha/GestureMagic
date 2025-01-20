<template>
    <div class="temp_dev">
        <canvas ref="dev_canvas" class="canvas_dev"></canvas>
    </div>
    <div class="right">
        <div class="camera-container">
            <!-- 视频流显示 -->
            <video ref="video" class="camera-feed" autoplay style="transform: scaleX(-1);"></video>
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
            pic_num: 0, // 用来显示在人脸上的图像编号，可以是0，1，2
            trajectory: [], // 记录轨迹的数组
            faceDetector: undefined, // 人脸识别器
            faceResults: undefined, // 人脸识别结果
            // 保存贴图的变量
            glass: [],
            hat: [],
            mustache: [],
            nose: []
        }
    },
    mounted() {
        this.loadDrawingUtils();
        this.loadImage();
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
        loadImage(){
            const imagePaths = {
                glass: ['/GestureMagic/imgs/glasses1.png', '/GestureMagic/imgs/glasses2.png', '/GestureMagic/imgs/glasses3.png'],
                hat: ['/GestureMagic/imgs/hat1.png', '/GestureMagic/imgs/hat2.png', '/GestureMagic/imgs/hat3.png'],
                mustache: ['/GestureMagic/imgs/mustache1.png', '/GestureMagic/imgs/mustache2.png', '/GestureMagic/imgs/mustache3.png'],
                nose: ['/GestureMagic/imgs/nose1.png', '/GestureMagic/imgs/nose2.png', '/GestureMagic/imgs/nose3.png']
            };
            for (const [key, paths] of Object.entries(imagePaths)) {
                paths.forEach(path => {
                    const img = new Image();
                    img.src = path;
                    img.onload = () => {
                        this[key].push(img);
                    };
                });
            }
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
                // 获取翻转后的图像数据
                canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvasCtx.getImageData(0, 0, canvas.width, canvas.height);

                // 将翻转后的图像数据传递给gestureRecognizer进行检测
                this.results = await this.gestureRecognizer.recognizeForVideo(imageData, startTimeMs);
                // 将翻转后的图像数据传递给faceDetector进行检测
                this.faceResults = await this.faceDetector.detectForVideo(imageData, startTimeMs);
            }

            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height); // 清空画布

            // 绘制Pose关键点和连接线
            if (this.results.landmarks) {
                this.drawGestureResults(canvasCtx);
                this.poseDetected(this.results);
            }

            

            canvasCtx.restore(); // 恢复上下文状态

            // 根据情况绘制贴图
            if (this.pics && this[this.pics].length > 0 && this.faceResults && this.faceResults.detections && this.faceResults.detections.length > 0) {
                const img = this[this.pics][this.pic_num];
                let x,y,w,h=0;
                // 考虑glass的情况
                if (this.pics === 'glass') {
                    const eyes_distance = Math.sqrt(
                        (canvas.width*this.faceResults.detections[0].keypoints[1].x - canvas.width*this.faceResults.detections[0].keypoints[0].x) ** 2 +
                        (canvas.height*this.faceResults.detections[0].keypoints[1].y - canvas.height*this.faceResults.detections[0].keypoints[0].y) ** 2
                    );
                    const eyes_avg_x = (canvas.width*this.faceResults.detections[0].keypoints[1].x + canvas.width*this.faceResults.detections[0].keypoints[0].x) / 2;
                    const eyes_avg_y = (canvas.height*this.faceResults.detections[0].keypoints[1].y + canvas.height*this.faceResults.detections[0].keypoints[0].y) / 2;
                    x = eyes_avg_x + eyes_distance*1.0;
                    y = eyes_avg_y - eyes_distance*0.5;
                    w = eyes_distance * 2.0;
                    h = eyes_distance * 1.0;
                }
                // 考虑hat的情况
                if (this.pics == 'hat'){
                    // 先求出0，1关键点的中点
                    const head_mid_x = (canvas.width*this.faceResults.detections[0].keypoints[0].x + canvas.width*this.faceResults.detections[0].keypoints[1].x) / 2;
                    const head_mid_y = (canvas.height*this.faceResults.detections[0].keypoints[0].y + canvas.height*this.faceResults.detections[0].keypoints[1].y) / 2;
                    // 眼睛中点到嘴巴中点的向量
                    const eye_to_mouth_x = canvas.width*this.faceResults.detections[0].keypoints[3].x - head_mid_x;
                    const eye_to_mouth_y = canvas.height*this.faceResults.detections[0].keypoints[3].y - head_mid_y;
                    // 眼睛中点为起点，向上移动一段距离就是头
                    x = head_mid_x - eye_to_mouth_x * 1.0;
                    y = head_mid_y - eye_to_mouth_y * 1.8;
                    // 计算头的宽度（4，5关键点的距离）
                    const head_width = Math.sqrt(
                        (canvas.width*this.faceResults.detections[0].keypoints[4].x - canvas.width*this.faceResults.detections[0].keypoints[5].x) ** 2 +
                        (canvas.height*this.faceResults.detections[0].keypoints[4].y - canvas.height*this.faceResults.detections[0].keypoints[5].y) ** 2
                    );
                    w = head_width * 1.5;
                    h = head_width * 1.5;
                    x = x + w * 0.5
                    y = y - h * 0.5
                }
                // 考虑mustache的情况
                if (this.pics == 'mustache'){
                    // 计算2,3关键点的中点
                    const mouth_mid_x = (canvas.width*this.faceResults.detections[0].keypoints[2].x + canvas.width*this.faceResults.detections[0].keypoints[3].x) / 2;
                    const mouth_mid_y = (canvas.height*this.faceResults.detections[0].keypoints[2].y + canvas.height*this.faceResults.detections[0].keypoints[3].y) / 2;
                    // 计算2,3关键点的距离
                    const eyes_distance = Math.sqrt(
                        (canvas.width*this.faceResults.detections[0].keypoints[0].x - canvas.width*this.faceResults.detections[0].keypoints[1].x) ** 2 +
                        (canvas.height*this.faceResults.detections[0].keypoints[0].y - canvas.height*this.faceResults.detections[0].keypoints[1].y) ** 2
                    );
                    x = mouth_mid_x;
                    y = mouth_mid_y;
                    w = eyes_distance;
                    h = eyes_distance * 0.5;
                    x = x + w * 0.5;
                    y = y - h * 0.5;
                }
                // 考虑nose的情况
                if (this.pics == 'nose'){
                    // 计算0，1关键点间距
                    const eyes_distance = Math.sqrt(
                        (canvas.width*this.faceResults.detections[0].keypoints[0].x - canvas.width*this.faceResults.detections[0].keypoints[1].x) ** 2 +
                        (canvas.height*this.faceResults.detections[0].keypoints[0].y - canvas.height*this.faceResults.detections[0].keypoints[1].y) ** 2
                    );
                    // 位置设置为2号关键点
                    x = canvas.width*this.faceResults.detections[0].keypoints[2].x;
                    y = canvas.height*this.faceResults.detections[0].keypoints[2].y;
                    // 鼻子图片的比例为1：1，大小为眼睛间距的0.6
                    w = eyes_distance * 0.6;
                    h = eyes_distance * 0.6;
                    x = x + w * 0.5;
                    y = y - h * 0.5;
                }
                x = canvas.width - x;
                imgCanvasCtx.drawImage(img, x, y, w, h);
            }

            // 绘制人脸关键点
            if (this.faceResults && this.faceResults.detections) {
                this.drawFaceDetections(canvasCtx, this.faceResults.detections);
            }

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
            console.log("识别到类别：", recognizedType);
            if (recognizedType) {
                this.pics = recognizedType;
                this.pic_num = Math.floor(Math.random() * 3);
            }
        },
        knnRecognize(trajectory) {
            // KNN识别逻辑
            // 新建两个数组xs和ys
            let xs = [];
            let ys = [];
            // 找到trajectory中x和y的最大最小值
            let minX = trajectory[0].x;
            let maxX = trajectory[0].x;
            let minY = trajectory[0].y;
            let maxY = trajectory[0].y;
            for (let i = 1; i < trajectory.length; i++) {
                if (trajectory[i].x < minX) {
                    minX = trajectory[i].x;
                }
                if (trajectory[i].x > maxX) {
                    maxX = trajectory[i].x;
                }
                if (trajectory[i].y < minY) {
                    minY = trajectory[i].y;
                }
                if (trajectory[i].y > maxY) {
                    maxY = trajectory[i].y;
                }
            }
            console.log("minX:", minX);
            console.log("maxX:", maxX);
            console.log("minY:", minY);
            console.log("maxY:", maxY);
            // 找到较长的边长
            const side = Math.max(maxX - minX, maxY - minY);
            // 将trajectory中的点映射到[0,20]区间
            for (let i = 0; i < trajectory.length; i++) {
                xs.push((trajectory[i].x - minX) / side * 20);
                ys.push((trajectory[i].y - minY) / side * 20);
            }
            console.log("trajectory:", trajectory);
            console.log("xs:", xs); 
            console.log("ys:", ys);
            // 创建一个20x20的画布，但不用显示，然后将xs,ys中的点和其中的连线绘制到画布上（宽度为2）
            const canvas_knn = document.createElement("canvas");
            // const canvas_knn = this.$refs.dev_canvas;
            canvas_knn.width = 20;
            canvas_knn.height = 20;
            // 把画布画成全白的
            const ctx = canvas_knn.getContext("2d");
            ctx.fillStyle = "white";
            ctx.fillRect(0, 0, 20, 20);
            ctx.strokeStyle = "black";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(xs[0], ys[0]);
            for (let i = 1; i < xs.length; i++) {
                ctx.lineTo(xs[i], ys[i]);
            }
            ctx.stroke();
            // 下载画布中的图片到本地
            const a = document.createElement("a");
            a.href = canvas_knn.toDataURL();
            a.download = "trajectory.png";
            a.click();


            return ['hat', 'mustache', 'glass', 'nose'][Math.floor(Math.random() * 4)];
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
    width: 100%;
    height: 100%;
    object-fit: fill; /* fill: 填满空间但可能形变，contain: 保持比例但留有空白，cover: 保持比例但填满 */
    display: block;
    position: absolute;
    top: 0;
    left: 0;
}

.camera-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}
</style>

.canvas_dev {
    position: absolute;
    top: 20%;
    left: 40%;
    width: 10%;
    height: 10%;
    z-index: 100;
}