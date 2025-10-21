const form = document.getElementById("upload-form");
const resultContainer = document.getElementById("result-container");
const result = document.getElementById("result");
const signImage = document.getElementById("sign-image");
const alphabet = document.getElementById("alphabet");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            throw new Error("Prediction request failed");
        }

        const data = await res.json();
        console.log(data);

        result.textContent =
            `📂 File: ${data.filename}\n🔤 📊 Confidence: ${data.confidence}`;

        alphabet.textContent = data.predicted_class;
        signImage.src = data.image_url + `?t=${Date.now()}`;

        resultContainer.classList.remove("hidden");
    } catch (err) {
        result.textContent = `❌ Error: ${err.message}`;
        resultContainer.classList.remove("hidden");
    }
});