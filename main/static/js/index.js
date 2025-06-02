const form = document.getElementById("analysisForm");
const loadingSpinner = document.getElementById("loadingSpinner");
const resultsContainer = document.getElementById("resultsContainer");
const errorContainer = document.getElementById("errorContainer");
const testBtn = document.getElementById("testBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await analyzeUrl();
});

testBtn.addEventListener("click", async () => {
  await testAnalysis();
});

async function analyzeUrl() {
  const url = document.getElementById("urlInput").value;
  const threshold = document.getElementById("thresholdInput").value;

  showLoading();
  hideResults();
  hideError();

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url, threshold }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Terjadi kesalahan");
    }

    displayResults(data);
  } catch (error) {
    showError(error.message);
  } finally {
    hideLoading();
  }
}

async function testAnalysis() {
  showLoading();
  hideResults();
  hideError();

  try {
    const response = await fetch("/test");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Terjadi kesalahan");
    }

    displayResults(data);
  } catch (error) {
    showError(error.message);
  } finally {
    hideLoading();
  }
}

function displayResults(data) {
  // Update statistics
  document.getElementById("totalComments").textContent = data.total_comments;
  document.getElementById("hateComments").textContent = data.hate_comments;
  document.getElementById("cleanComments").textContent =
    data.total_comments - data.hate_comments;
  document.getElementById("platformBadge").textContent = data.platform;

  // Display results list
  const hateCommentsList = document.getElementById("hateCommentsList");
  const noResultsMessage = document.getElementById("noResultsMessage");
  const resultsList = document.getElementById("resultsList");

  if (data.results && data.results.length > 0) {
    hateCommentsList.innerHTML = "";
    resultsList.style.display = "block";
    noResultsMessage.style.display = "none";

    data.results.forEach((result, index) => {
      const resultCard = createResultCard(result, index + 1);
      hateCommentsList.appendChild(resultCard);
    });
  } else {
    resultsList.style.display = "none";
    noResultsMessage.style.display = "block";
  }

  resultsContainer.style.display = "block";
}

function createResultCard(result, index) {
  const card = document.createElement("div");
  card.className = "card result-card mb-3";

  const severityClass = `severity-${result.severity
    .toLowerCase()
    .replace(" ", "-")}`;

  card.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <h6 class="card-title mb-0">
                            <i class="fas fa-comment text-danger me-2"></i>
                            Komentar #${index}
                        </h6>
                        <div class="d-flex gap-2">
                            <span class="badge ${severityClass} severity-badge">${
    result.severity
  }</span>
                            <span class="badge bg-secondary">Skor: ${
                              result.similarity_score
                            }</span>
                        </div>
                    </div>
                    
                    <div class="comment-text">
                        <i class="fas fa-quote-left text-muted me-2"></i>
                        ${result.original_comment}
                    </div>
                    
                    <div class="mt-3">
                        <small class="text-muted">
                            <i class="fas fa-fingerprint me-1"></i>
                            <strong>Pola yang Cocok:</strong> ${
                              result.matched_pattern
                            }
                        </small>
                    </div>
                    
                    <div class="progress mt-2" style="height: 8px;">
                        <div class="progress-bar bg-danger" role="progressbar" 
                             style="width: ${result.similarity_score * 100}%" 
                             aria-valuenow="${result.similarity_score * 100}" 
                             aria-valuemin="0" aria-valuemax="100">
                        </div>
                    </div>
                </div>
            `;

  return card;
}

function showLoading() {
  loadingSpinner.style.display = "block";
}

function hideLoading() {
  loadingSpinner.style.display = "none";
}

function showResults() {
  resultsContainer.style.display = "block";
}

function hideResults() {
  resultsContainer.style.display = "none";
}

function showError(message) {
  document.getElementById("errorMessage").textContent = message;
  errorContainer.style.display = "block";
}

function hideError() {
  errorContainer.style.display = "none";
}
