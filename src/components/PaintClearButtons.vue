<template>
  <div class="top">
    <div class="paint">
      <img src="/static/绘图.png" alt="绘图" />
      <span>绘图</span>
    </div>
    <div class="clear">
      <img src="/static/清除.png" alt="清除" />
      <span>清除</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PaintClearButtons',
};
</script>

<style scoped>


.top {
  position: absolute;
  top: 22%;
  left: 7%;
  width: 20%;
  height: 8%;
  gap: 30%;
  display: flex;
  justify-content: center;
}

.paint,
.clear {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.paint img,
.clear img {
  height: 100%;
  margin-bottom: 10%;
}

span {
  font-size: 2.5vw;
  font-weight: 700;
}
</style>