const mimusa_python_code_URL = "https://mimusa-test-model.onrender.com/generate"

const app = Vue.createApp({

    //=========== DATA PROPERTIES ===========
    data() {
        return {
            original_text: "",
            score: "",
            sentiment: "",
            final_text: "",
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
                    this.final_text = data.text;
                    if (this.score == -2) {
                        this.sentiment = "Strongly Negative"
                    }
                    else if (this.score == -1) {
                        this.sentiment = "Negative"
                    }
                    else if (this.score == 0) {
                        this.sentiment = "Neutral"
                    }
                    else if (this.score == 1) {
                        this.sentiment = "Positive"
                    }
                    else if (this.score == 2) {
                        this.sentiment = "Strongly Positive"
                    }
                    console.log(this.final_text);
                });
        },
    }
})


app.mount('#app')