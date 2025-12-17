// Clear button (client-side only)
const clearBtn = document.getElementById("clearBtn");
const imageInput = document.getElementById("image_path");
if (clearBtn && imageInput) {
  clearBtn.addEventListener("click", () => {
    imageInput.value = "";
    imageInput.focus();
  });
}

// Disable button while submitting (prevents double submit)
const form = document.getElementById("runForm");
const runBtn = document.getElementById("runBtn");
if (form && runBtn) {
  form.addEventListener("submit", () => {
    runBtn.disabled = true;
    runBtn.textContent = "Running...";
  });
}
