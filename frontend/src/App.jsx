import { useEffect, useRef, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import SiteShell from "./components/SiteShell.jsx";
import HomePage from "./pages/HomePage.jsx";
import DetectionPage from "./pages/DetectionPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import ExplainabilityPage from "./pages/ExplainabilityPage.jsx";
import DiseasePage from "./pages/DiseasePage.jsx";
import ResearchPage from "./pages/ResearchPage.jsx";
import { analyzeImage, checkApiHealth, getApiErrorMessage } from "./lib/api.js";
import { ALLOWED_TYPES, MAX_FILE_BYTES } from "./components/UploadPanel.jsx";
import "./styles.css";

function isValidPredictionResult(value) {
  return Boolean(value && typeof value === "object" && typeof value.predicted_class === "string" && Number.isFinite(Number(value.confidence)) && Array.isArray(value.top_k_predictions));
}

function RouteView({ state, handlers }) {
  const location = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [location.pathname]);

  return (
    <Routes>
      <Route path="/" element={<HomePage apiOnline={state.apiOnline} />} />
      <Route path="/detect" element={<DetectionPage {...state} {...handlers} />} />
      <Route path="/results" element={<ResultsPage {...state} />} />
      <Route path="/explainability" element={<ExplainabilityPage {...state} />} />
      <Route path="/disease" element={<DiseasePage {...state} />} />
      <Route path="/research" element={<ResearchPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function AppContent() {
  const navigate = useNavigate();
  const [apiOnline, setApiOnline] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [imageDimensions, setImageDimensions] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const analysisRequestRef = useRef(0);
  const currentFileRef = useRef(null);
  const [pipelineState, setPipelineState] = useState("idle");
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    let mounted = true;
    const refreshHealth = () => checkApiHealth()
      .then(() => { if (mounted) setApiOnline(true); })
      .catch(() => { if (mounted) setApiOnline(false); });
    refreshHealth();
    const interval = window.setInterval(refreshHealth, 15000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!previewUrl) return undefined;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  function selectFile(candidate) {
    analysisRequestRef.current += 1;
    currentFileRef.current = null;
    setError("");
    setResult(null);
    setPipelineState("idle");
    if (!candidate) return;
    if (!ALLOWED_TYPES.includes(candidate.type)) {
      setError("Choose a JPG, JPEG, or PNG image.");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setError("This image exceeds the 15 MB limit.");
      return;
    }
    setFile(candidate);
    currentFileRef.current = candidate;
    const objectUrl = URL.createObjectURL(candidate);
    setPreviewUrl(objectUrl);
    setImageDimensions(null);
    const image = new Image();
    image.onload = () => setImageDimensions({ width: image.naturalWidth, height: image.naturalHeight });
    image.src = objectUrl;
  }

  function removeFile() {
    analysisRequestRef.current += 1;
    currentFileRef.current = null;
    setFile(null);
    setPreviewUrl("");
    setImageDimensions(null);
    setResult(null);
    setError("");
    setPipelineState("idle");
  }

  async function runAnalysis() {
    if (!file || pipelineState === "running") return;
    const uploadedFile = file;
    const requestId = ++analysisRequestRef.current;
    setError("");
    setResult(null);
    setPipelineState("running");
    setActiveStage(0);
    try {
      const response = await analyzeImage(uploadedFile);
      if (requestId !== analysisRequestRef.current || currentFileRef.current !== uploadedFile) return;
      const completedResult = response?.data;
      const successful = response?.status === 200 && isValidPredictionResult(completedResult);
      if (!successful) {
        throw new Error("The analysis service returned an invalid prediction response.");
      }
      setResult(completedResult);
      setPipelineState("complete");
      setActiveStage(4);
      setApiOnline(true);
      navigate("/results", { replace: true, state: { analysisResult: completedResult } });
    } catch (requestError) {
      if (requestId !== analysisRequestRef.current || currentFileRef.current !== uploadedFile) return;
      setError(getApiErrorMessage(requestError));
      setPipelineState("error");
      setApiOnline(Boolean(requestError?.response));
    }
  }

  const state = {
    apiOnline,
    file,
    previewUrl,
    imageDimensions,
    result,
    error,
    pipelineState,
    activeStage,
  };
  const handlers = { onFile: selectFile, onRemove: removeFile, onAnalyze: runAnalysis };

  return (
    <SiteShell apiOnline={apiOnline}>
      <RouteView state={state} handlers={handlers} />
    </SiteShell>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
