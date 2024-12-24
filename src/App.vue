<template>
  <div id="app">
    <Title />
    <PaintClearButtons />
    <MiddleSection />
    <PromptText :title="title" />
    <BottomImage />
    <VideoContainer />
  </div>
</template>

<script>
import Title from './components/Title.vue';
import PaintClearButtons from './components/PaintClearButtons.vue';
import MiddleSection from './components/MiddleSection.vue';
import PromptText from './components/PromptText.vue';
import BottomImage from './components/BottomImage.vue';
import VideoContainer from './components/VideoContainer.vue';

export default {
  components: {
    Title,
    PaintClearButtons,
    MiddleSection,
    PromptText,
    BottomImage,
    VideoContainer
  },
  data() {
    return {
      title: "保持正确的手势能画的更连贯哦！",
      count: 0
    };
  },
  mounted() {
    setInterval(() => {
      this.title = this.count === 0
        ? "保持正确的手势能画的更连贯哦！"
        : "识别错误也是正常的，尝试发现其中的规律";
      this.count = this.count === 0 ? 1 : 0;
    }, 5000);
  }
};
</script>

<style>
[v-cloak] {
  display: none;
}


</style>