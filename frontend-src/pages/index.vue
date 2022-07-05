<template>
  <section class="section">
    <div class="columns is-mobile">
      <card title="anonymask" icon="cover.png">
        <b-field>
          <ui>
            <li><b>Anonymask</b> generates an image in which the company logo is hidden so naturally.</li>
          </ui>
        </b-field>
        <b-field v-if='doing'>
          <b-progress :value="percent" type="is-success" show-value format="percent"></b-progress>
        </b-field>
        <b-field>
          <b-button @click="execute" type="is-link is-light">Upload</b-button>
          <input type="file" id="file" style="display:none" v-on:change="upload">
        </b-field>
      </card>
      <card title="original" v-if="doing" :icon="image_src">
      </card>
      <card title="converted" v-if="percent == 100" :icon="image_dst">
      </card>
    </div>
  </section>
</template>

<script>
import Card from '~/components/Card'

const axios = require('axios');
const API_URL = "http://redirect.yuiga.dev/api/anonymask";
export default {
  name: 'IndexPage',
  components: {
    Card
  },
  created() {

  },
  methods: {
    execute() {
      this.doing = false
      this.percent = 0;
      document.getElementById("file").click();
    },
    async upload() {
      // get the file
      let file = document.getElementById("file").files[0];
      this.image_src = URL.createObjectURL(file);

      // upload & inference
      this.doing = true;
      this.percent = 10;
      var formData = new FormData();
      formData.append("image", file);
      const res = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        responseType: 'arraybuffer',
        onUploadProgress: (progressEvent) => {
          this.percent = Math.round(progressEvent.loaded * 32 / progressEvent.total + 10);
          if (progressEvent.loaded == 100) {
            this.$buefy.toast.open({
              message: 'Upload!',
              type: 'is-success'
            })
          }
        }
      });

      // display the result
      this.percent = 100;
      const dst = new Blob([res.data], { type: "image/png" });
      this.image_dst = URL.createObjectURL(dst);
      this.$buefy.toast.open({
        message: 'Done!',
        type: 'is-success'
      })
    }
  }, data() {
    return {
      doing: false,
      percent: 0,
      image_src: "",
      image_dst: ""
    }
  }
}
</script>
