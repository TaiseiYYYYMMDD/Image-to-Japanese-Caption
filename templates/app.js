const form = document.getElementById("runForm");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const fileInput = document.getElementById("image_file");

if (form && runBtn) {
  form.addEventListener("submit", () => {
    runBtn.disabled = true;
    runBtn.textContent = "Running...";
  });
}

if (clearBtn && fileInput) {
  clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    // 画面上の結果を消したいなら、リロードが一番シンプル
    // location.href = "/";
  });
}
