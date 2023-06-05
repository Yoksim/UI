<template>
  <div id="app" class="container my-3">
        <h1>MIMUSA</h1>
        <div class="row w-150 my-5 mx-auto border border-secondary-subtle p-3">
            <p class="fw-bold">Enter your sentence:
                <input type="text" placeholder="Insert sentence" class="w-75 my-2" style="margin-right: 10px;" v-model="text"/>
                <button @click="generateSentiment" type="button" class="ms-3 btn btn-sm btn-primary">
                    Submit
                </button>
            </p>
            <!-- <p>
              <br/>
              You typed: <b>{{ text }}</b>
            </p> -->
            <p v-show="hasError">{{ message }}</p>
        </div>
    </div>
</template>

<script>
export default {
  name: 'HomeComponent',
  props: {
    msg: String
  },
  data() {
    return {
      text: '',
      hasError: false,
      message: 'There is a problem generating sentiment value, please try again later.'
    }
  },
  methods: {
        generateSentiment: function () {
            console.log(this.text)
        },
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
