<template>
    <div class="middle">
        <div>
            <img src="/GestureMagic/static/帽子.png" alt="帽子" />
            <span class="shape_name">帽子</span>
        </div>
        <div>
            <img src="/GestureMagic/static/胡子.png" alt="胡子" />
            <span class="shape_name">胡子</span>
        </div>
        <div>
            <img src="/GestureMagic/static/眼镜.png" alt="眼镜" />
            <span class="shape_name">眼镜</span>
        </div>
        <div>
            <img src="/GestureMagic/static/bizi.png" alt="鼻子" />
            <span class="shape_name">鼻子</span>
        </div>
    </div>
</template>

<script>
export default {
    name: 'MiddleSection',
};
</script>

<style scoped>
.middle {
    position: absolute;
    top: 50%;
    left: 5%;
    width: 24%;
    height: 50%;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
}

.middle div {
    width: 45%;
    height: 40%;
    background-color: #fff;
    border-radius: 10%;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

.middle div img {
    width: 60%;
    height: 60%;
}

.middle div span {
    font-size: 2.5vw;
    font-weight: 700;
}
</style>