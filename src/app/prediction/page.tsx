import { useState } from "react";
import axios from "axios";

export default function PredictPage() {
  const [features, setFeatures] = useState<string>("");
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Handle input change
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFeatures(e.target.value);
  };

  // Handle API request
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const featureList = features.split(",").map((num) => parseFloat(num.trim()));

      if (featureList.length !== 50) {
        setError("Please enter exactly 50 numerical values.");
        setLoading(false);
        return;
      }

      const response = await axios.post("http://localhost:8000/predict", {
        features: featureList,
      });

      setPrediction(response.data.prediction[0]);
    } catch (err) {
      setError("Error fetching prediction. Check input format or API.");
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-4">Stock Price Prediction</h1>

      <div className="bg-white p-6 shadow-lg rounded-lg w-96">
        <label className="block text-gray-700 mb-2">Enter last 50 stock prices (comma-separated):</label>
        <input
          type="text"
          value={features}
          onChange={handleChange}
          placeholder="e.g., 100, 101, 102, 103..."
          className="w-full p-2 border rounded"
        />

        <button
          onClick={handlePredict}
          className="w-full bg-blue-500 text-white p-2 rounded mt-4"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Get Prediction"}
        </button>

        {prediction !== null && (
          <div className="mt-4 p-3 bg-green-100 text-green-800 rounded">
            Predicted Price: <strong>{prediction.toFixed(2)}</strong>
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 bg-red-100 text-red-800 rounded">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
