import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "",
  timeout: 300000,
});

export async function checkApiHealth() {
  const response = await api.get("/health", { timeout: 5000 });
  return response.data;
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post("/api/v1/predict", formData, {
    timeout: 300000,
  });
  return { status: response.status, data: response.data };
}

export function getApiErrorMessage(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => item?.msg).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (error?.response) {
    return `Analysis service returned HTTP ${error.response.status}.`;
  }
  if (error?.code === "ECONNABORTED" || error?.code === "ETIMEDOUT") {
    return "Analysis is taking longer than expected. Please try again.";
  }
  if (error?.code === "ERR_NETWORK") {
    return "The browser could not connect to the analysis service. Check the API connection and try again.";
  }
  return "Could not reach the analysis service. Check that the API is running and try again.";
}
