"use client";

import React, { useState, useEffect } from "react";
import useStockWebSocket from "@/hooks/useStockWebSocket";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const PortfolioPage = () => {
  const [ticker, setTicker] = useState("AAPL"); // Default ticker
  const latestPrice = useStockWebSocket(ticker);
  const [stockData, setStockData] = useState<{ time: string; price: number }[]>([]);

  // Update stock data when new price is received
  useEffect(() => {
    if (latestPrice) {
      setStockData((prevData) => [
        ...prevData.slice(-49), // Keep only the last 50 data points
        { time: new Date().toLocaleTimeString(), price: latestPrice },
      ]);
    }
  }, [latestPrice]);

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Stock Portfolio</h1>

      {/* Ticker Selection */}
      <div className="mb-4">
        <label className="mr-2 font-semibold">Select Ticker:</label>
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="AAPL">Apple (AAPL)</option>
          <option value="GOOGL">Google (GOOGL)</option>
          <option value="AMZN">Amazon (AMZN)</option>
          <option value="MSFT">Microsoft (MSFT)</option>
        </select>
      </div>

      {/* Real-Time Stock Price */}
      <div className="mb-4 p-4 bg-gray-100 rounded-md text-lg font-semibold">
        <span className="mr-2">📈 Latest Price:</span>
        {latestPrice ? (
          <span className="text-green-600">${latestPrice.toFixed(2)}</span>
        ) : (
          <span className="text-gray-500">Fetching...</span>
        )}
      </div>

      {/* Stock Price Chart */}
      <div className="border p-4 rounded-lg bg-gray-100 dark:bg-gray-800">
        <h2 className="text-lg font-semibold mb-2">{ticker} Stock Price Chart</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={stockData}>
            <XAxis dataKey="time" />
            <YAxis domain={["auto", "auto"]} />
            <Tooltip />
            <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PortfolioPage;
