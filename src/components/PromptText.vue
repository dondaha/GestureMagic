<template>
  <div class="promptText">
    <h1 :data-spotlight="title" v-cloak>{{ title }}</h1>
  </div>
</template>

<script>
export default {
  props: {
    title: {
      type: String,
      required: true
    }
  }
}
</script>

<style scoped>
.promptText {
  position: absolute;
  top: 9%;
  width: 50%;
  right: 8%;
  height: 8%;
  display: flex;
  justify-content: flex-end;
  margin: 0;
  padding: 0;
}

h1 {
  color: #333;
  font-size: 2.5vw;
  position: relative;
}

h1::after {
  content: attr(data-spotlight);
  color: transparent;
  position: absolute;
  top: 0;
  left: 0;
  clip-path: ellipse(100px 100px at 0% 50%);
  background-image: url('/GestureMagic/photo-1584840460878-3d65c4191bd6.avif');
  background-size: 150%;
  background-position: center center;
  background-clip: text;
  -webkit-background-clip: text;
  animation: identifier 3s infinite;
}

@keyframes identifier {
  0% {
    clip-path: ellipse(100px 100px at 0% 50%);
  }

  50% {
    clip-path: ellipse(100px 100px at 100% 50%);
  }

  100% {
    clip-path: ellipse(100px 100px at 0% 50%);
  }
}
</style>