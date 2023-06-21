const mimusa_python_code_URL = "https://mimusa-test-model.onrender.com:8000/generate"

const app = Vue.createApp({

    //=========== DATA PROPERTIES ===========
    data() {
        return {
            original_text: "",
            score: "",
        }
    },

    //=========== METHODS ===========
    methods: {
        get_score() {
            console.log(this.original_text);
            fetch(`${mimusa_python_code_URL}`,
                {
                    method: "POST",
                    headers: {
                        "Content-type": "text/plain"
                    },
                    body: this.original_text
                })
                .then(response => response.json())
                .then(data => {
                    this.score = data.data;
                })
        },

    }
})


app.mount('#app')