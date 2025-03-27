"use client";
import { useState, useEffect } from "react";

const BuySellPage = () => {
  const [ticker, setTicker] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [price, setPrice] = useState("");
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Fetch all trades
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch("/api/trades");
        const data = await response.json();
        setTrades(data.trades);
      } catch (error) {
        console.error("Error fetching trades:", error);
      }
    };

    fetchTrades();
  }, []);

  const handleTrade = async (action: "buy" | "sell") => {
    setLoading(true);
    setMessage("");

    try {
      const response = await fetch("/api/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, quantity, action, price }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.message);

      setMessage(`Trade Successful: ${data.trade.ticker} ${action} ${data.trade.quantity} shares at $${data.trade.price}`);

      // Refresh trades after successful transaction
      setTrades((prev) => [data.trade, ...prev]);
    } catch (error: any) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 bg-black text-white min-h-screen">
      <h1 className="text-2xl font-bold mb-4">Buy & Sell Stocks</h1>

      <div className="mb-4">
        <label className="block mb-2">Ticker Symbol</label>
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          className="p-2 w-full rounded bg-gray-800 text-white border border-gray-600"
        />
      </div>

      <div className="mb-4">
        <label className="block mb-2">Quantity</label>
        <input
          type="number"
          value={quantity}
          min="1"
          onChange={(e) => setQuantity(Number(e.target.value))}
          className="p-2 w-full rounded bg-gray-800 text-white border border-gray-600"
        />
      </div>

      <div className="mb-4">
        <label className="block mb-2">Price</label>
        <input
          type="text"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          className="p-2 w-full rounded bg-gray-800 text-white border border-gray-600"
        />
      </div>

      <div className="flex space-x-4">
        <button
          onClick={() => handleTrade("buy")}
          className="bg-green-500 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={loading}
        >
          Buy
        </button>
        <button
          onClick={() => handleTrade("sell")}
          className="bg-red-500 text-white px-4 py-2 rounded disabled:opacity-50"
          disabled={loading}
        >
          Sell
        </button>
      </div>

      {message && <p className="mt-4">{message}</p>}

      <h2 className="text-xl font-semibold mt-6">Trade History</h2>
      <div className="border border-gray-700 p-4 mt-2 rounded">
        {trades.length > 0 ? (
          trades.map((trade, index) => (
            <div key={index} className="border-b border-gray-600 py-2">
              {trade.timestamp}: {trade.ticker} - {trade.action} {trade.quantity} shares at ${trade.price}
            </div>
          ))
        ) : (
          <p className="text-gray-500">No trades yet.</p>
        )}
      </div>
    </div>
  );
};

export default BuySellPage;
